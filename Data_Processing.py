# =============================
# Import libraries
# =============================
import os
import numpy as np
import pandas as pd
import json
import pyproj

# =============================
# WORKING DIRECTORY SETUP
# =============================
os.chdir("C:/Users/mu128/Desktop/Manuscript Writing/Deep Learning for Flow Delay Calculation/Manuscript Code/working_directory/")

text_files_dir = "AgLeader Txt Files/"
output_dir = "Outputs"
processed_txt_dir = os.path.join(output_dir, "Processed_Txt")
os.makedirs(processed_txt_dir, exist_ok=True)

# =============================
# HEADER
# =============================
header = [
    "Longitude", "Latitude", "Grain_Flow", "GPS_Time", "Logging_Interval",
    "Distance", "Swath", "Moisture", "Header_Status", "Pass", "Serial_Number",
    "Field_ID", "Load_ID", "Grain_Type", "GPS_Status", "PDOP", "Altitude"
]

# =============================
# CROP DENSITY (lb/bu)
# =============================
CROP_DENSITY_MAP = {
    "SOYBEANS": 60,
    "CORN": 56
}

# =============================
# Add yield variable
# =============================
text_files = [f for f in os.listdir(text_files_dir) if f.lower().endswith(".txt")]

all_dfs = []

for fname in text_files:
    fpath = os.path.join(text_files_dir, fname)

    df = pd.read_csv(fpath, delimiter=",", header=None)
    if df.shape[1] != len(header):
        raise ValueError(
            f"{fname}: Expected {len(header)} columns but found {df.shape[1]}. "
            "Your header list does not match the TXT structure."
        )

    df.columns = header

    # Map crop density from Grain_Type
    df["Crop_Density"] = (
        df["Grain_Type"].astype(str).str.upper().map(CROP_DENSITY_MAP)
    )
    
    # Convert inches to feet
    df["Distance"] = df["Distance"] / 12.0
    df["Swath"] = df["Swath"] / 12.0


    # Yield equation
    numerator = df["Grain_Flow"] * df["Logging_Interval"] * 43560.0
    denominator = df["Distance"] * df["Swath"] * df["Crop_Density"]

    df["Yield"] = numerator / denominator

    # Clean up infinities / invalid values (distance=0, swath=0, unknown grain type, etc.)
    df["Yield"] = df["Yield"].replace([np.inf, -np.inf], np.nan)

    # drop rows where yield can't be computed
    df = df.dropna(subset=["Yield"]).copy()

    # Save processed TXT with header + Yield column
    out_txt = os.path.join(processed_txt_dir, fname.replace(".txt", ".csv"))
    df.to_csv(out_txt, index=False)

    # Optional: store for combined CSV
    df["Source_File"] = fname
    all_dfs.append(df)

    print(f"Processed: {fname}  ->  {out_txt}")
    
    
# ==========================================================
# Start - Rasterization and Patch Generation
# ==========================================================
RAW_DIR = os.path.join("Outputs", "Processed_Txt")
OUT_DIR = os.path.join("Outputs", "prepared_patches")

os.makedirs(OUT_DIR, exist_ok=True)

Ns = np.arange(50, 650, 25)

MAX_DELAY = 15
PATCH_SIZE = 24
PATCH_STRIDE = 12
MIN_FIELD_COVERAGE = 0.40


# ==========================================================
# PCDI STYLE HELPERS
# ==========================================================
def grid_params_for_N(x, y, N):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    width = xmax - xmin
    height = ymax - ymin

    if width > height:
        pad = (width - height) / 2
        ymin -= pad
        ymax += pad
        side = width
    else:
        pad = (height - width) / 2
        xmin -= pad
        xmax += pad
        side = height

    cell = side / N
    return xmin, ymin, cell, N


def rasterize_points(x, y, z, xmin, ymin, cell, N):
    ix = ((x - xmin) / cell).astype(int)
    iy = ((y - ymin) / cell).astype(int)

    valid = ((ix >= 0) & (ix < N) &
             (iy >= 0) & (iy < N) &
             np.isfinite(z))

    ix, iy, z = ix[valid], iy[valid], z[valid]

    raster = np.zeros((N, N), dtype=float)
    count = np.zeros((N, N), dtype=int)

    for i, j, val in zip(ix, iy, z):
        raster[j, i] += val
        count[j, i] += 1

    mask = count > 0
    raster[mask] /= count[mask]

    return raster, mask


def shift_mask(mask):
    out = np.zeros_like(mask, dtype=bool)
    out[1:, 1:] = mask[:-1, :-1]
    return out


def shift_1d_zero(z, k):
    out = np.zeros_like(z)
    if k > 0:
        out[:-k] = z[k:]
    elif k < 0:
        out[-k:] = z[:len(z) + k]
    else:
        out[:] = z
    return out


# ==========================================================
# FIND OPTIMAL RASTER PARAMS FOR FIELD
# ==========================================================
def find_optimal_raster_params_for_field(csv_path, Ns):
    df = pd.read_csv(csv_path)

    # Safer numeric conversion
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["Latitude"]  = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Yield"]     = pd.to_numeric(df["Yield"], errors="coerce")
    df = df.dropna(subset=["Longitude", "Latitude", "Yield"])

    
    # Since lat/lon already normalized 0–100,
    # treat them as Cartesian coordinates directly
    x = df["Longitude"].values
    y = df["Latitude"].values
    
    # Use correct zone for your data (keep fixed if all fields are same region)
    #proj = pyproj.Proj(proj="utm", zone=15, ellps="WGS84")
    #x, y = proj(df["Longitude"].values, df["Latitude"].values)


    z = df["Yield"].values

    valid_ratio_list = []
    coupled_ratio_list = []

    for N in Ns:
        xmin, ymin, cell, Nn = grid_params_for_N(x, y, N)
        raster, mask = rasterize_points(x, y, z, xmin, ymin, cell, Nn)

        rv = mask.sum() / len(z)
        shifted = shift_mask(mask)
        rc = (mask & shifted).sum() / mask.sum()

        valid_ratio_list.append(rv)
        coupled_ratio_list.append(rc)

    valid_ratio_list = np.array(valid_ratio_list)
    coupled_ratio_list = np.array(coupled_ratio_list)

    idx_opt = np.argmin(np.abs(valid_ratio_list - coupled_ratio_list))
    optimal_N = int(Ns[idx_opt])

    xmin_opt, ymin_opt, cell_opt, _ = grid_params_for_N(x, y, optimal_N)

    return x, y, z, xmin_opt, ymin_opt, cell_opt, optimal_N


# ==========================================================
# RASTER BUILDING FOR ALL DELAYS
# ==========================================================
def build_rasters_for_delays(x, y, z, xmin, ymin, cell, N, max_delay):
    delays = list(range(-max_delay, max_delay + 1))
    rasters = {}
    masks = {}

    for d in delays:
        z_shift = shift_1d_zero(z, d)
        raster_d, mask_d = rasterize_points(x, y, z_shift, xmin, ymin, cell, N)
        rasters[d] = raster_d
        masks[d] = mask_d

    return delays, rasters, masks


# ==========================================================
# PATCH EXTRACTION
# ==========================================================
def extract_patches(delays, rasters, masks,
                    patch_size, stride, min_field_coverage, base_delay=0):
    base_mask = masks[base_delay]
    H, W = base_mask.shape

    patches_by_delay = {d: [] for d in delays}

    for y0 in range(0, H - patch_size + 1, stride):
        for x0 in range(0, W - patch_size + 1, stride):

            y1 = y0 + patch_size
            x1 = x0 + patch_size

            base_patch_mask = base_mask[y0:y1, x0:x1]
            field_cov = base_patch_mask.mean()

            if field_cov < min_field_coverage:
                continue

            for d in delays:
                patch = rasters[d][y0:y1, x0:x1].astype(np.float32)
                patches_by_delay[d].append(patch)

    for d in delays:
        if len(patches_by_delay[d]):
            patches_by_delay[d] = np.stack(patches_by_delay[d])
        else:
            patches_by_delay[d] = np.zeros((0, patch_size, patch_size), dtype=np.float32)

    return patches_by_delay


# ==========================================================
# MAIN LOOP: PROCESS ALL FIELDS
# ==========================================================
if __name__ == "__main__":

    csv_files = [f for f in os.listdir(RAW_DIR)
                 if f.lower().endswith((".csv", ".txt"))]
    print(f"Found {len(csv_files)} files.")

    for fname in csv_files:

        print("\n========================================")
        print(f"Processing field: {fname}")
        print("========================================")

        csv_path = os.path.join(RAW_DIR, fname)
        field_name = os.path.splitext(fname)[0]

        # ---- Step 1: Optimal raster params ----
        x, y, z, xmin, ymin, cell, N = find_optimal_raster_params_for_field(csv_path, Ns)
        print(f"  Optimal N = {N}, cell = {cell:.4f}")

        # ---- Step 2: Build rasters ----
        delays, rasters, masks = build_rasters_for_delays(
            x, y, z, xmin, ymin, cell, N, MAX_DELAY
        )

        # ---- Step 3: Extract patches ----
        patches_by_delay = extract_patches(
            delays, rasters, masks,
            PATCH_SIZE, PATCH_STRIDE, MIN_FIELD_COVERAGE,
            base_delay=0
        )

        # ---- Step 4: Save patches ----
        save_dir = os.path.join(OUT_DIR, field_name)
        os.makedirs(save_dir, exist_ok=True)

        for d in delays:
            np.save(os.path.join(save_dir, f"delay_{d}.npy"),
                    patches_by_delay[d])

        # ---- Step 5: Save full rasters ----
        full_raster_dir = os.path.join(save_dir, "full_rasters")
        os.makedirs(full_raster_dir, exist_ok=True)

        for d in delays:
            np.save(os.path.join(full_raster_dir, f"raster_{d}.npy"),
                    rasters[d])

        # ---- Step 6: Metadata ----
        metadata = {
            "field": field_name,
            "optimal_N": N,
            "cell_size": float(cell),
            "xmin": float(xmin),
            "ymin": float(ymin),
            "num_patches": int(patches_by_delay[0].shape[0])
        }

        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"  Saved patches for {field_name}: {metadata['num_patches']} patches")

    print("\nAll fields processed. Patches saved to:")
    print(OUT_DIR)


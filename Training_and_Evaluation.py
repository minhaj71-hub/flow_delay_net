################### Gaussian white + Peak-centered Siamese Training ###################
import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import tensorflow as tf
tf.config.experimental.enable_op_determinism()

import json
import random
import numpy as np
import matplotlib.pyplot as plt
import keras
from scipy.ndimage import gaussian_filter
import pandas as pd
import seaborn as sns
import umap
import re
import matplotlib.pyplot as plt

os.chdir("working_directory/")

# ================================================================
# CUSTOM LAYER FOR L2 NORMALIZATION
# ================================================================
@keras.saving.register_keras_serializable(package="Custom")
class L2Norm(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=-1)


# ================================================================
# REPRODUCIBILITY
# ================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


# ================================================================
# CONFIG
# ================================================================
PATCH_DIR = r"Outputs/prepared_patches/"

print("CWD:", os.getcwd())
print("PATCH_DIR:", os.path.abspath(PATCH_DIR))
print("PATCH_DIR exists?", os.path.isdir(PATCH_DIR))
print("Field folders (first 10):", os.listdir(PATCH_DIR)[:10])


MAX_DELAY = 15
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
PATCH_CHANNELS = 1
EMBED_DIM = 64

# Patch geometry
PATCH_SIZE = 24
PATCH_STRIDE = 12
MIN_FIELD_COVERAGE = 0.40

# SAM
SAM_RHO = 0.03

# Teacher regression weight
LAMBDA_REG = 5.0


# ================================================================
# LOAD FIELDS + PER-FIELD NORMALIZATION
# ================================================================
fields = sorted(os.listdir(PATCH_DIR))
fields = [f for f in fields if os.path.isdir(os.path.join(PATCH_DIR, f))]

usable_fields = []
field_means = {}
field_stds = {}

for fld in fields:

    mpath = os.path.join(PATCH_DIR, fld, "metadata.json")
    if not os.path.exists(mpath):
        continue

    with open(mpath, "r") as f:
        meta = json.load(f)

    n_patches = meta.get("num_patches", 0)
    if n_patches == 0:
        continue

    usable_fields.append((fld, n_patches))

    # per-field normalization
    delay0_path = os.path.join(PATCH_DIR, fld, "delay_0.npy")
    if os.path.exists(delay0_path):
        arr0 = np.load(delay0_path)
        mask = arr0 != 0
        vals = arr0[mask]

        if vals.size > 100:
            m = float(vals.mean())
            s = float(vals.std())
            if s < 1e-6:
                s = 1.0
        else:
            m, s = 0.0, 1.0
    else:
        m, s = 0.0, 1.0

    field_means[fld] = m
    field_stds[fld] = s


# ================================================================
# TRAIN / VAL / TEST SPLIT (REPRODUCIBLE BUT RANDOM)
# ================================================================
rng = random.Random(42)        # independent seed
rng.shuffle(usable_fields)      # same shuffle order every run

n = len(usable_fields)
train_fields = usable_fields[:int(0.7*n)]
val_fields   = usable_fields[int(0.7*n):int(0.85*n)]
test_fields  = usable_fields[int(0.85*n):]

print(f"Train fields: {len(train_fields)}")
print(f"Val fields:   {len(val_fields)}")
print(f"Test fields:  {len(test_fields)}")


def safe_filename(s: str) -> str:
    # Replace Windows-illegal characters: \ / : * ? " < > | and also newlines/tabs
    s = re.sub(r'[\\/:*?"<>|\n\t]+', "_", s)
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s).strip("_")
    return s

# ================================================================
# LOAD PATCH
# ================================================================
def load_patch(field_name, index, delay):
    path = os.path.join(PATCH_DIR, field_name, f"delay_{delay}.npy")
    arr = np.load(path)[index]
    arr = arr[..., 0:1]

    m = field_means[field_name]
    s = field_stds[field_name]
    if s > 1e-6:
        arr = (arr - m) / s

    return arr.astype(np.float32)


# ================================================================
# GAUSSIAN SMOOTHNESS
# ================================================================
def gaussian_spatial_score(raster, sigma=2.0):
    r = raster.astype(np.float32)
    mask = r != 0
    if mask.sum() < 50:
        return 0.0

    r_masked = r.copy()
    r_masked[~mask] = 0

    num = gaussian_filter(r_masked, sigma=sigma)
    den = gaussian_filter(mask.astype(np.float32), sigma=sigma) + 1e-6

    mean = num / den
    v1 = r[mask].ravel()
    v2 = mean[mask].ravel()

    if v1.std() < 1e-6 or v2.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(v1, v2)[0, 1])


# ================================================================
# PRECOMPUTE SMOOTHNESS CURVES PER FIELD
# ================================================================
field_smoothness = {}

for fld, _ in usable_fields:
    fdir = os.path.join(PATCH_DIR, fld, "full_rasters")
    if not os.path.isdir(fdir):
        field_smoothness[fld] = {}
        continue

    scores = {}
    for d in range(-MAX_DELAY, MAX_DELAY+1):
        p = os.path.join(fdir, f"raster_{d}.npy")
        if os.path.exists(p):
            scores[d] = gaussian_spatial_score(np.load(p))
    field_smoothness[fld] = scores


def normalize_score_dict(score_dict):
    if len(score_dict) == 0:
        return {}
    arr = np.array(list(score_dict.values()), dtype=np.float32)
    mean, std = float(arr.mean()), float(arr.std())
    if std < 1e-6:
        return {d: 0.0 for d in score_dict}
    return {d: (float(v)-mean)/std for d,v in score_dict.items()}


# ================================================================
# DATASET CHARACTERISTICS (Gaussian teacher variability across fields)
# Save outputs to: Dataset_characteristics/
# Figures saved as: PNG (dpi=500, width=5000px) + SVG (editable text)
# ================================================================

DATASET_DIR = "Dataset_characteristics"
os.makedirs(DATASET_DIR, exist_ok=True)

# ----------------------------
# Global figure styling (constant font) + SVG editability
# ----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,

    # ---- KEY: keep SVG text as <text> (editable), not path outlines
    "svg.fonttype": "none",

    # Helpful if figures ever include raster images (keeps them embedded)
    "svg.image_inline": True,

    # Optional: better font embedding if you later export PDF/PS
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ----------------------------
# Helper: Save figure as PNG+SVG with required PNG size
# width=5000px, dpi=500 => figsize width = 10 inches
# ----------------------------
PNG_DPI = 500
PNG_WIDTH_PX = 5000
FIG_W_IN = PNG_WIDTH_PX / PNG_DPI  # 10 inches

PNG_WIDTH_PX_SMALL = 2000
FIG_W_SMALL = PNG_WIDTH_PX_SMALL / PNG_DPI

SMALL_FONT = 9
MED_FONT   = 10


def save_fig_dual(fig, base_name):
    png_path = os.path.join(DATASET_DIR, f"{base_name}.png")
    svg_path = os.path.join(DATASET_DIR, f"{base_name}.svg")

    # PNG for submission/preview
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight")

    # SVG with editable text (because svg.fonttype='none')
    fig.savefig(svg_path, bbox_inches="tight")

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")

# ================================================================
# Compute meaningful between-field teacher-curve metrics
# ================================================================
records = []

for fld, _ in usable_fields:
    smooth_dict = field_smoothness.get(fld, {})
    if len(smooth_dict) < 3:
        continue

    # ---- RAW curve (meaningful magnitude/contrast info)
    raw_delays = np.array(sorted(smooth_dict.keys()), dtype=int)
    raw_vals   = np.array([smooth_dict[d] for d in raw_delays], dtype=np.float32)

    # ---- Normalized curve (shape-only comparability within field)
    norm = normalize_score_dict(smooth_dict)
    norm_delays = np.array(sorted(norm.keys()), dtype=int)
    norm_vals   = np.array([norm[d] for d in norm_delays], dtype=np.float32)

    # Peak delay from normalized curve (stable for argmax)
    peak_idx = int(np.argmax(norm_vals))
    peak_delay = int(norm_delays[peak_idx])

    # Raw peak value at that delay
    raw_peak_val = float(smooth_dict[peak_delay])

    # ---- Curve-structure metrics (RAW)
    raw_max = float(np.max(raw_vals))
    raw_min = float(np.min(raw_vals))
    raw_med = float(np.median(raw_vals))

    raw_range = raw_max - raw_min
    raw_prominence = raw_max - raw_med

    # Peak-gap (max - second max): small gap => ambiguous/multi-peak curve
    if len(raw_vals) >= 2:
        sorted_raw = np.sort(raw_vals)
        raw_gap = float(sorted_raw[-1] - sorted_raw[-2])
    else:
        raw_gap = 0.0

    # Optional: peak width above half-max (on RAW curve)
    if raw_range > 1e-8:
        half_thr = raw_min + 0.5 * raw_range
        width = int(np.sum(raw_vals >= half_thr))
    else:
        width = 0

    records.append({
        "Field": fld,
        "PeakDelay": peak_delay,
        "RawPeakValue": raw_peak_val,
        "RawMax": raw_max,
        "RawMin": raw_min,
        "RawMedian": raw_med,
        "RawRange": float(raw_range),
        "RawPeakProminence": float(raw_prominence),
        "RawPeakGap": float(raw_gap),
        "PeakWidthHalfMax": int(width),
        "NumDelays": int(len(raw_delays))
    })

df_teacher_stats = pd.DataFrame(records)

# Save table + summary
csv_path = os.path.join(DATASET_DIR, "teacher_curve_stats_by_field.csv")
df_teacher_stats.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

summary_path = os.path.join(DATASET_DIR, "teacher_curve_stats_summary.txt")
with open(summary_path, "w") as f:
    f.write("Teacher curve stats (by field) summary:\n\n")
    f.write(str(df_teacher_stats.describe(include="all")))

print(f"Saved: {summary_path}")

# ================================================================
# FIGURE 1: Histogram of PeakDelay (most important)
# ================================================================
fig = plt.figure(figsize=(FIG_W_IN, 6))
ax = fig.gca()

bins = np.arange(-MAX_DELAY - 0.5, MAX_DELAY + 1.5, 1)
ax.hist(df_teacher_stats["PeakDelay"], bins=bins, edgecolor="black")

ax.set_xlabel("Teacher Peak Delay")
ax.set_ylabel("Number of Fields")

save_fig_dual(fig, "hist_teacher_peak_delay")
plt.close(fig)

# ================================================================
# FIGURE 2: RawPeakProminence distribution (how “peaky” vs flat)
# ================================================================
fig = plt.figure(figsize=(FIG_W_IN, 6))
ax = fig.gca()

ax.hist(df_teacher_stats["RawPeakProminence"], bins=30, edgecolor="black")
ax.set_xlabel("Raw Peak Prominence (max − median)")
ax.set_ylabel("Number of Fields")

save_fig_dual(fig, "hist_teacher_raw_peak_prominence")
plt.close(fig)

# ================================================================
# FIGURE 3: RawPeakGap distribution (ambiguity / multi-peak indicator)
# ================================================================
fig = plt.figure(figsize=(FIG_W_IN, 6))
ax = fig.gca()

ax.hist(df_teacher_stats["RawPeakGap"], bins=30, edgecolor="black")
ax.set_xlabel("Raw Peak Gap (max − second max)")
ax.set_ylabel("Number of Fields")

save_fig_dual(fig, "hist_teacher_raw_peak_gap")
plt.close(fig)

# ================================================================
# FIGURE 4 (optional): Peak width at half-max (sharp vs broad peaks)
# ================================================================
fig = plt.figure(figsize=(FIG_W_IN, 6))
ax = fig.gca()

ax.hist(df_teacher_stats["PeakWidthHalfMax"], bins=np.arange(-0.5, 31.5, 1), edgecolor="black")
ax.set_xlabel("Peak Width (Nos. of delays above half-max threshold)")
ax.set_ylabel("Number of Fields")

save_fig_dual(fig, "hist_teacher_peak_width_halfmax")
plt.close(fig)

print("\nDone: Dataset characteristics outputs saved in:", os.path.abspath(DATASET_DIR))





# ================================================================
# ANCHOR-CENTERED SIAMESE SAMPLING (RNG PERSISTS ACROSS EPOCHS)
# ================================================================
K_POS = 3   # positive window: |d2 - d1| <= 3
K_NEG = 8   # negative window: |d2 - d1| >= 8

def sample_siamese_pair(field_set, rng):
    """
    Uses a provided RNG object so randomness persists across epochs
    (and is reproducible across runs when the RNG seed is fixed once).
    """
    while True:
        fld, n_patches = rng.choice(field_set)

        smooth_dict = field_smoothness.get(fld, {})
        if len(smooth_dict) == 0:
            continue

        norm = normalize_score_dict(smooth_dict)
        all_delays = sorted(norm.keys())
        if len(all_delays) < 3:
            continue

        # 1) pick anchor delay anywhere
        d1 = rng.choice(all_delays)

        # 2) define positives / negatives around anchor
        pos_candidates = [d for d in all_delays if abs(d - d1) <= K_POS]
        neg_candidates = [d for d in all_delays if abs(d - d1) >= K_NEG]
        if len(pos_candidates) == 0 or len(neg_candidates) == 0:
            continue

        # random patch indices
        idx1 = rng.randint(0, n_patches - 1)
        idx2 = rng.randint(0, n_patches - 1)

        # 3) sample pos/neg half-half
        if rng.random() < 0.5:
            d2 = rng.choice(pos_candidates)
            y = 1.0
        else:
            d2 = rng.choice(neg_candidates)
            y = 0.0

        yield {
            "x1": load_patch(fld, idx1, d1),
            "x2": load_patch(fld, idx2, d2),
            "y":  np.float32(y),
            "t1": np.float32(norm[d1]),
            "t2": np.float32(norm[d2]),
        }


def make_siamese_dataset(field_set, seed=SEED):
    rng = random.Random(seed)
    pair_gen = sample_siamese_pair(field_set, rng)  # one persistent generator

    def gen():
        while True:
            yield next(pair_gen)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_types={"x1": tf.float32, "x2": tf.float32, "y": tf.float32, "t1": tf.float32, "t2": tf.float32},
        output_shapes={
            "x1": tf.TensorShape([PATCH_SIZE, PATCH_SIZE, PATCH_CHANNELS]),
            "x2": tf.TensorShape([PATCH_SIZE, PATCH_SIZE, PATCH_CHANNELS]),
            "y":  tf.TensorShape([]),
            "t1": tf.TensorShape([]),
            "t2": tf.TensorShape([]),
        },
    )

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.shuffle(2000, seed=seed, reshuffle_each_iteration=False)
    return ds.batch(BATCH_SIZE).prefetch(1)

# ================================================================
# Function for plot saving of the model prediction
# ================================================================
OUT_DIR = "gaussian_vs_model_prediction"
os.makedirs(OUT_DIR, exist_ok=True)

def get_model_curve_for_field(field_name, encoder):
    """
    For a given field:
      - For each delay d that has a teacher score
      - Run all patches at that delay through encoder
      - Average the predicted smoothness_head over patches
      - Return: dict delay -> normalized model smoothness
               dict delay -> normalized teacher (for consistent delays)
    """
    smooth_dict = field_smoothness.get(field_name, {})
    if len(smooth_dict) == 0:
        return {}, {}

    # Normalized teacher for this field
    teacher_norm = normalize_score_dict(smooth_dict)

    model_scores = {}

    for d in sorted(teacher_norm.keys()):
        patch_path = os.path.join(PATCH_DIR, field_name, f"delay_{d}.npy")
        if not os.path.exists(patch_path):
            continue

        patches = np.load(patch_path)  # (num_patches, H, W, C)
        if patches.ndim != 4 or patches.shape[-1] < 1:
            continue

        # keep yield-only channel and normalize with field stats
        patches = patches[..., 0:1]
        m = field_means[field_name]
        s = field_stds[field_name]
        if s > 1e-6:
            patches = (patches - m) / s
        patches = patches.astype(np.float32)

        # run through encoder in batches
        _, scores = encoder.predict(patches, batch_size=64, verbose=0)
        model_scores[d] = float(np.mean(scores))

    # normalize model scores per field
    model_norm = normalize_score_dict(model_scores)

    # align keys (take common delays)
    common_delays = sorted(set(teacher_norm.keys()) & set(model_norm.keys()))
    teacher_norm = {d: teacher_norm[d] for d in common_delays}
    model_norm   = {d: model_norm[d]   for d in common_delays}

    return model_norm, teacher_norm



train_ds = make_siamese_dataset(train_fields)
val_ds   = make_siamese_dataset(val_fields)


def build_encoder():
    inp = tf.keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, 1))

    # ==================================================
    # Block 1: 24 → 12
    # ==================================================
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)   # 12×12

    # ==================================================
    # Block 2: 12 → 6
    # ==================================================
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)   # 6×6

    # ==================================================
    # Block 3a: Residual block at 6×6
    # ==================================================
    res = x
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Channel matching for first residual
    if res.shape[-1] != 128:
        res = tf.keras.layers.Conv2D(128, 1, padding="same")(res)

    x = tf.keras.layers.Add()([x, res])
    x = tf.keras.layers.Activation("relu")(x)

    # ==================================================
    # Block 3b: Second residual block at 6×6 (NEW)
    # ==================================================
    res = x
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([x, res])
    x = tf.keras.layers.Activation("relu")(x)

    # ==================================================
    # Block 4: 6 → 3
    # ==================================================
    x = tf.keras.layers.MaxPool2D()(x)   # 3×3
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # ==================================================
    # Global feature aggregation
    # ==================================================
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Dense depth (important for embedding richness)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    # ==================================================
    # Embedding + regression head
    # ==================================================
    emb = tf.keras.layers.Dense(EMBED_DIM)(x)
    emb = L2Norm()(emb)

    score = tf.keras.layers.Dense(1, activation=None, name="smoothness_head")(emb)

    return tf.keras.Model(inp, [emb, score], name="encoder")


encoder = build_encoder()
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)


# ================================================================
# CONTRASTIVE LOSS
# ================================================================
def contrastive_loss(e1, e2, y, margin=1.2):

    sim  = tf.reduce_sum(e1 * e2, axis=1)
    dist = 1.0 - sim

    pos_loss = y * tf.square(dist)
    neg_loss = (1-y) * tf.square(tf.maximum(margin - dist, 0.0))

    return tf.reduce_mean(pos_loss + neg_loss)


# ================================================================
# TRAINING LOOP
# ================================================================
train_losses = []
val_losses = []
best_val = float("inf")
wait = 0
patience = 8

print("\nTraining with ANCHOR-centered + ±3 positives + ≥8 negatives ...\n")


for epoch in range(EPOCHS):

    batch_losses = []

    # -------------------------------
    # TRAIN
    # -------------------------------
    for batch in train_ds.take(400):

        with tf.GradientTape() as tape:

            e1,p1 = encoder(batch["x1"],training=True)
            e2,p2 = encoder(batch["x2"],training=True)

            Lc = contrastive_loss(e1,e2,batch["y"])
            Lr = 0.5*(tf.reduce_mean((p1[:,0]-batch["t1"])**2)+
                      tf.reduce_mean((p2[:,0]-batch["t2"])**2))

            loss = Lc + LAMBDA_REG*Lr

        grads = tape.gradient(loss, encoder.trainable_variables)

        # SAM first step
        grad_norm = tf.sqrt(tf.add_n([tf.reduce_sum(g*g) for g in grads if g is not None])+1e-12)
        scale = SAM_RHO / grad_norm

        e_ws = []
        for w,g in zip(encoder.trainable_variables,grads):
            if g is None:
                e_ws.append(tf.zeros_like(w))
                continue
            e = g * scale
            w.assign_add(e)
            e_ws.append(e)

        # SAM second step
        with tf.GradientTape() as tape2:

            e1b,p1b = encoder(batch["x1"],training=True)
            e2b,p2b = encoder(batch["x2"],training=True)

            Lc2 = contrastive_loss(e1b,e2b,batch["y"])
            Lr2 = 0.5*(tf.reduce_mean((p1b[:,0]-batch["t1"])**2)+
                       tf.reduce_mean((p2b[:,0]-batch["t2"])**2))

            final_loss = Lc2 + LAMBDA_REG*Lr2

        grads2 = tape2.gradient(final_loss, encoder.trainable_variables)

        for w,e in zip(encoder.trainable_variables, e_ws):
            w.assign_sub(e)

        optimizer.apply_gradients(zip(grads2,encoder.trainable_variables))
        batch_losses.append(float(final_loss.numpy()))

    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)

    # -------------------------------
    # VALIDATION
    # -------------------------------
    val_batch_losses = []
    for batch in val_ds.take(400):

        e1v,p1v = encoder(batch["x1"],training=False)
        e2v,p2v = encoder(batch["x2"],training=False)

        Lcv = contrastive_loss(e1v,e2v,batch["y"])
        Lrv = 0.5*(tf.reduce_mean((p1v[:,0]-batch["t1"])**2)+
                   tf.reduce_mean((p2v[:,0]-batch["t2"])**2))

        val_batch_losses.append(float((Lcv+LAMBDA_REG*Lrv).numpy()))

    val_loss = np.mean(val_batch_losses)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}  Train={train_loss:.4f}  Val={val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        encoder.save("best_encoder_tf.keras")
    else:
        wait += 1
        if wait >= patience:
            print("\nEarly stopping.\n")
            break


# ---------------------------
# Load Best Model
# ---------------------------
encoder = tf.keras.models.load_model("best_encoder_tf.keras",
                                     custom_objects={"L2Norm":L2Norm})

# ================================================================
# PLOT TRAINING CURVE
# ================================================================
output_dir = "training_performance"
os.makedirs(output_dir, exist_ok=True)

# Create a dataframe
loss_df = pd.DataFrame({
    "epoch": range(1, len(train_losses) + 1),
    "train_loss": train_losses,
    "val_loss": val_losses
})

# Save to CSV
loss_df.to_csv(
    os.path.join(output_dir, "contrastive_loss_history.csv"),
    index=False
)

plt.figure(figsize=(7,5))

plt.plot(
    train_losses,
    linestyle="--",
    linewidth=2,
    label="Train Loss"
)

plt.plot(
    val_losses,
    linestyle="-",
    linewidth=2,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "contrastive_loss_curve.png"), dpi=500)
plt.savefig(os.path.join(output_dir, "contrastive_loss_curve.svg"))
plt.show()



# ================================================================
# Loop over test fields and make plots
# ================================================================
abs_diffs = []

for fld, _ in test_fields:
    model_curve, teacher_curve = get_model_curve_for_field(fld, encoder)
    if len(model_curve) == 0 or len(teacher_curve) == 0:
        continue

    delays = sorted(model_curve.keys())
    m_vals = [model_curve[d]   for d in delays]
    t_vals = [teacher_curve[d] for d in delays]

    # argmax delays
    d_model   = delays[int(np.argmax(m_vals))]
    d_teacher = delays[int(np.argmax(t_vals))]
    abs_diffs.append(abs(d_model - d_teacher))

    # ---- plot ----
    plt.figure(figsize=(7, 4))
    plt.plot(delays, m_vals, "-o", label="Model Smoothness")
    plt.plot(delays, t_vals, "-o", label="Gaussian Teacher")
    plt.xlabel("Delay")
    plt.ylabel("Normalized Smoothness")
    plt.title(f"{fld}\nModel={d_model},  Teacher={d_teacher}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"{fld}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    
# ------------------------------------------------
# Aggregate evaluation metrics (Overall performance)
# ------------------------------------------------
model_peaks = []
teacher_peaks = []
eval_fields = []

for fld, _ in test_fields:
    model_curve, teacher_curve = get_model_curve_for_field(fld, encoder)
    if len(model_curve) == 0 or len(teacher_curve) == 0:
        continue

    delays = sorted(model_curve.keys())
    m_vals = [model_curve[d]   for d in delays]
    t_vals = [teacher_curve[d] for d in delays]

    d_model   = delays[int(np.argmax(m_vals))]
    d_teacher = delays[int(np.argmax(t_vals))]

    model_peaks.append(d_model)
    teacher_peaks.append(d_teacher)
    eval_fields.append(fld)

# Convert to numpy
model_peaks   = np.array(model_peaks, dtype=np.int32)
teacher_peaks = np.array(teacher_peaks, dtype=np.int32)

if len(model_peaks) == 0:
    print("No valid test fields for evaluation.")
else:
    abs_err = np.abs(model_peaks - teacher_peaks)

    # --- Main metrics ---
    mae = float(np.mean(abs_err))
    exact = float(np.mean(abs_err == 0) * 100.0)
    within1 = float(np.mean(abs_err <= 1) * 100.0)
    within2 = float(np.mean(abs_err <= 2) * 100.0)

    # --- Baselines ---
    # 1) Constant predictor: predict the MEDIAN teacher delay from test set
    baseline_const = int(np.median(teacher_peaks))
    abs_err_const = np.abs(baseline_const - teacher_peaks)
    mae_const = float(np.mean(abs_err_const))
    within1_const = float(np.mean(abs_err_const <= 1) * 100.0)

    # 2) Constant predictor: always predict 0
    abs_err_zero = np.abs(0 - teacher_peaks)
    mae_zero = float(np.mean(abs_err_zero))
    within1_zero = float(np.mean(abs_err_zero <= 1) * 100.0)

    # 3) Random predictor: uniformly sample one of the candidate delays
    rng_eval = np.random.RandomState(42)
    rand_preds = rng_eval.randint(-MAX_DELAY, MAX_DELAY + 1, size=len(teacher_peaks))
    abs_err_rand = np.abs(rand_preds - teacher_peaks)
    mae_rand = float(np.mean(abs_err_rand))
    within1_rand = float(np.mean(abs_err_rand <= 1) * 100.0)

    # --- Print summary ---
    print("============================================")
    print(" Overall Evaluation Summary (Test Fields)")
    print("============================================")
    print(f"Evaluated test fields: {len(teacher_peaks)}")
    print(f"MAE (samples):         {mae:.3f}")
    print(f"Exact match (%):       {exact:.2f}")
    print(f"Within ±1 (%):         {within1:.2f}")
    print(f"Within ±2 (%):         {within2:.2f}")
    print("--------------------------------------------")
    print(" Baselines")
    print("--------------------------------------------")
    print(f"Constant (median={baseline_const}) MAE: {mae_const:.3f},  Within ±1: {within1_const:.2f}%")
    print(f"Constant (0)            MAE: {mae_zero:.3f},  Within ±1: {within1_zero:.2f}%")
    print(f"Random uniform          MAE: {mae_rand:.3f},  Within ±1: {within1_rand:.2f}%")
    print("============================================")

    # Optional: save per-field predictions for appendix/supplement
    df_eval = pd.DataFrame({
        "Field": eval_fields,
        "TeacherDelay": teacher_peaks,
        "ModelDelay": model_peaks,
        "AbsError": abs_err
    })
    df_eval.to_csv(os.path.join(OUT_DIR, "test_field_delay_predictions.csv"), index=False)
    
    
# ================================================================
# 4.3 FIELD-LEVEL ERROR CHARACTERISTICS & FAILURE MODES
# Uses:
#   - df_eval (Field, TeacherDelay, ModelDelay, AbsError) from evaluation
#   - df_teacher_stats (Field, PeakWidthHalfMax, RawPeakGap, RawPeakProminence, ...)
# Saves outputs to: Field_level_error_analysis/
# Figures saved as: PNG (dpi=500, width=5000px) + SVG (editable text)
# ================================================================

ERROR_DIR = "Field_level_error_analysis"
os.makedirs(ERROR_DIR, exist_ok=True)

# ----------------------------
# Use the same plot styling and SVG editability
# ----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 20,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 12,
    "svg.fonttype": "none",
    "svg.image_inline": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_fig_dual_to(fig, out_dir, base_name):
    png_path = os.path.join(out_dir, f"{base_name}.png")
    svg_path = os.path.join(out_dir, f"{base_name}.svg")
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")

# ----------------------------
# Merge per-field errors with teacher-curve ambiguity metrics
# ----------------------------
# df_teacher_stats is created earlier in your script (Dataset_characteristics block)
df_merged = df_eval.merge(df_teacher_stats, on="Field", how="left")

# Save merged table (nice for appendix/supplement)
df_merged.to_csv(os.path.join(ERROR_DIR, "field_error_vs_teacher_metrics.csv"), index=False)

# ================================================================
# 4.3.1 Distribution of field-level errors
# ================================================================
abs_err = df_merged["AbsError"].values
N = len(abs_err)

pct0 = 100.0 * np.mean(abs_err == 0)
pct1 = 100.0 * np.mean(abs_err == 1)
pct2 = 100.0 * np.mean(abs_err == 2)
pct3p = 100.0 * np.mean(abs_err >= 3)

print("\n============================================")
print(" Field-Level Error Distribution (Test Fields)")
print("============================================")
print(f"N fields: {N}")
print(f"Error = 0:   {pct0:.2f}%")
print(f"Error = 1:   {pct1:.2f}%")
print(f"Error = 2:   {pct2:.2f}%")
print(f"Error >= 3:  {pct3p:.2f}%")
print("============================================\n")

# ------------------------------------------------
# Absolute delay error histogram (compact, discrete)
# ------------------------------------------------
fig = plt.figure(figsize=(5, 4))
ax = fig.gca()

max_e = int(np.max(abs_err)) if N > 0 else 5
bins = np.arange(-0.5, max_e + 1.5, 1)

ax.hist(abs_err, bins=bins, edgecolor="black")

ax.set_xlabel("Absolute delay error |Model − Teacher| (shifts)", fontsize=10)
ax.set_ylabel("Number of fields", fontsize=10)
ax.tick_params(axis="both", labelsize=8)
ax.set_xticks(np.arange(0, max_e + 1, 1))

save_fig_dual_to(fig, ERROR_DIR, "hist_abs_delay_error_small")
plt.close(fig)






# --- Bar chart for error = 0,1,2,>=3 (COUNTS) ---
fig = plt.figure(figsize=(FIG_W_SMALL, 3.5))
ax = fig.gca()

cnt0  = int(np.sum(abs_err == 0))
cnt1  = int(np.sum(abs_err == 1))
cnt2  = int(np.sum(abs_err == 2))
cnt3p = int(np.sum(abs_err >= 3))

cats = ["0", "1", "2", "≥3"]
vals = [cnt0, cnt1, cnt2, cnt3p]

ax.bar(cats, vals, edgecolor="black")

ax.set_xlabel(
    "Absolute error category (shifts)",
    fontsize=MED_FONT
)
ax.set_ylabel(
    "Number of fields",
    fontsize=MED_FONT
)

ax.tick_params(axis="both", labelsize=SMALL_FONT)
ax.grid(axis="y", alpha=0.3)

save_fig_dual_to(fig, ERROR_DIR, "bar_error_categories_counts")
plt.close(fig)


# ================================================================
# Relationship between error and teacher-curve ambiguity
# ================================================================
# Helper: scatter + trend line (optional)
def scatter_error_vs(metric_col, xlabel, base_name):
    x = df_merged[metric_col].values
    y = df_merged["AbsError"].values

    # remove NaNs if any
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.gca()
    ax.scatter(x, y, alpha=0.65)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Absolute delay error |Model − Teacher| (shifts)")

    save_fig_dual_to(fig, ERROR_DIR, base_name)
    plt.close(fig)

# Error vs peak width (broad plateaus -> often more ambiguity)
scatter_error_vs(
    metric_col="PeakWidthHalfMax",
    xlabel="Teacher peak width at half-max (number of delays)",
    base_name="scatter_error_vs_peak_width_halfmax"
)

# Error vs peak gap (small gap -> competing peaks -> ambiguous)
scatter_error_vs(
    metric_col="RawPeakGap",
    xlabel="Teacher peak gap (max − second max, raw)",
    base_name="scatter_error_vs_raw_peak_gap"
)

# Error vs peak prominence (weak signal -> harder)
scatter_error_vs(
    metric_col="RawPeakProminence",
    xlabel="Teacher peak prominence (max − median, raw)",
    base_name="scatter_error_vs_raw_peak_prominence"
)

# Optional: correlation summary (non-parametric is fine)
corr_out = os.path.join(ERROR_DIR, "correlations_error_vs_teacher_metrics.txt")
with open(corr_out, "w") as f:
    f.write("Spearman correlations with AbsError\n\n")
    for col in ["PeakWidthHalfMax", "RawPeakGap", "RawPeakProminence"]:
        sub = df_merged[[col, "AbsError"]].dropna()
        if len(sub) < 10:
            continue
        rho = sub[col].corr(sub["AbsError"], method="spearman")
        f.write(f"{col}: Spearman rho = {rho:.4f} (n={len(sub)})\n")
print(f"Saved: {corr_out}")



# ================================================================
# Representative success / near-miss / failure fields (MANUAL)
# Outputs:
#   Field_level_error_analysis/Representative_fields_manual/
#     - selected_fields.txt
#     - selected_fields_summary.csv
#     - curve figures (PNG+SVG via save_fig_dual_to)
#     - Rasters/ triptych figures (PNG+SVG)
#     - representative_field_raster_stats.txt
# ================================================================

REP_DIR = os.path.join(ERROR_DIR, "Representative_fields_manual")
os.makedirs(REP_DIR, exist_ok=True)

# ----------------------------
# ✅ YOU CHOOSE THESE THREE FIELD NAMES
# ----------------------------
SUCCESS_FIELD  = "Demo_success"
NEARMISS_FIELD = "Demo_near-miss"
FAIL_FIELD     = "Demo_failure"

selected = [
    ("Success (sharp, correct)",  SUCCESS_FIELD),
    ("Near-miss (ambiguous)",     NEARMISS_FIELD),
    ("Failure (ambiguous/weak)",  FAIL_FIELD),
]

# ----------------------------
# Save a record of chosen fields
# ----------------------------
with open(os.path.join(REP_DIR, "selected_fields.txt"), "w") as f:
    for label, fld in selected:
        f.write(f"{label}: {fld}\n")

# ----------------------------
# Helper: print a compact summary row for the chosen field
# ----------------------------
def print_field_summary(dfm: pd.DataFrame, field_name: str, label: str) -> None:
    row = dfm[dfm["Field"] == field_name]
    if row.empty:
        print(f"[WARNING] {label}: Field not found in df_merged -> {field_name}")
        return

    r = row.iloc[0]
    print("------------------------------------------------------------")
    print(f"{label}: {field_name}")
    print(f" TeacherDelay={int(r['TeacherDelay'])}  ModelDelay={int(r['ModelDelay'])}  AbsError={int(r['AbsError'])}")
    print(
        f" PeakWidthHalfMax={r.get('PeakWidthHalfMax', np.nan)}  "
        f"RawPeakGap={r.get('RawPeakGap', np.nan)}  "
        f"RawPeakProminence={r.get('RawPeakProminence', np.nan)}"
    )
    print("------------------------------------------------------------")

# ----------------------------
# Plot and save the curve for a selected field
# ----------------------------
def save_field_curve(field_name: str, tag: str) -> dict:
    """
    Saves curve figure and returns dict with:
      { 'd_teacher': int, 'd_model': int, 'abs_err': int, 'n_delays': int }
    """
    model_curve, teacher_curve = get_model_curve_for_field(field_name, encoder)
    if (model_curve is None) or (teacher_curve is None) or (len(model_curve) == 0) or (len(teacher_curve) == 0):
        print(f"[WARNING] Curve missing/empty for {tag}: {field_name}")
        return {"d_teacher": None, "d_model": None, "abs_err": None, "n_delays": 0}

    # ✅ Robust: use shared delay keys only
    delays = sorted(set(model_curve.keys()) & set(teacher_curve.keys()))
    if len(delays) == 0:
        print(f"[WARNING] No shared delay keys for {tag}: {field_name}")
        return {"d_teacher": None, "d_model": None, "abs_err": None, "n_delays": 0}

    m_vals = [float(model_curve[d]) for d in delays]
    t_vals = [float(teacher_curve[d]) for d in delays]

    d_model = int(delays[int(np.argmax(m_vals))])
    d_teacher = int(delays[int(np.argmax(t_vals))])
    err = int(abs(d_model - d_teacher))

    fig = plt.figure(figsize=(FIG_W_IN, 6))
    ax = fig.gca()
    ax.plot(delays, m_vals, linestyle="--", label="FlowDelayNet")
    ax.plot(delays, t_vals, label="Gaussian teacher")
    ax.axvline(d_model, linestyle="--", label=f"Pred={d_model}")
    ax.axvline(d_teacher, linestyle=":", label=f"Teacher={d_teacher}")

    ax.set_xlabel("Candidate delay (shifts)")
    ax.set_ylabel("Normalized smoothness score")
    ax.legend()
    fig.tight_layout()

    base = safe_filename(f"{tag}_{field_name}")
    save_fig_dual_to(fig, REP_DIR, base)
    plt.close(fig)

    return {"d_teacher": d_teacher, "d_model": d_model, "abs_err": err, "n_delays": len(delays)}

# ----------------------------
# Raster utilities for representative-field diagnosis
# ----------------------------
REP_RASTER_DIR = os.path.join(REP_DIR, "Rasters")
os.makedirs(REP_RASTER_DIR, exist_ok=True)

def load_full_raster(field_name: str, delay: int):
    """Loads full raster for a given field and delay from prepared_patches/<field>/full_rasters/."""
    p = os.path.join(PATCH_DIR, field_name, "full_rasters", f"raster_{delay}.npy")
    if not os.path.exists(p):
        return None
    return np.load(p).astype(np.float32)

def raster_valid_mask(r: np.ndarray):
    """Valid pixels are non-zero in your pipeline."""
    return (r != 0)

def compute_raster_stats(r: np.ndarray):
    """Simple descriptive stats on valid pixels only."""
    mask = raster_valid_mask(r)
    n_valid = int(mask.sum())
    H, W = r.shape
    if n_valid == 0:
        return {
            "n_valid": 0,
            "valid_frac": 0.0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p10": np.nan,
            "p50": np.nan,
            "p90": np.nan,
        }
    vals = r[mask]
    return {
        "n_valid": n_valid,
        "valid_frac": float(n_valid / (H * W)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
    }

def save_raster_triptych(field_name: str, tag: str, teacher_delay: int, model_delay: int, raw_delay: int = 0):
    """
    Saves a 3-panel figure:
      (1) Raw raster (d=0 by default),
      (2) Teacher peak raster (d=teacher_delay),
      (3) Model peak raster (d=model_delay).
    """
    r_raw = load_full_raster(field_name, int(raw_delay))
    r_t   = load_full_raster(field_name, int(teacher_delay))
    r_m   = load_full_raster(field_name, int(model_delay))

    if (r_raw is None) or (r_t is None) or (r_m is None):
        print(f"[WARNING] Missing full raster for {field_name}: raw={raw_delay}, teacher={teacher_delay}, model={model_delay}")
        return

    def robust_vmin_vmax(r):
        m = raster_valid_mask(r)
        if m.sum() < 50:
            return None, None
        v = r[m]
        return float(np.percentile(v, 2)), float(np.percentile(v, 98))

    # Option A (recommended): use a common vmin/vmax across all three for fair visual comparison
    vmins_vmaxs = [robust_vmin_vmax(r) for r in (r_raw, r_t, r_m)]
    vmins = [vv[0] for vv in vmins_vmaxs if vv[0] is not None]
    vmaxs = [vv[1] for vv in vmins_vmaxs if vv[1] is not None]
    vmin_common = min(vmins) if len(vmins) else None
    vmax_common = max(vmaxs) if len(vmaxs) else None

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    im1 = ax1.imshow(r_raw, vmin=vmin_common, vmax=vmax_common)
    ax1.set_title(f"Raw raster (d={raw_delay})")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(r_t, vmin=vmin_common, vmax=vmax_common)
    ax2.set_title(f"Gaussian teacher peak (d={teacher_delay})")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    im3 = ax3.imshow(r_m, vmin=vmin_common, vmax=vmax_common)
    ax3.set_title(f"Model peak (d={model_delay})")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    fig.tight_layout()

    base = safe_filename(f"{tag}_{field_name}_rasters_raw{raw_delay}_teacher{teacher_delay}_model{model_delay}")
    save_fig_dual_to(fig, REP_RASTER_DIR, base)
    plt.close(fig)


def append_stats_to_file(field_name: str, tag: str, teacher_delay: int, model_delay: int):
    """Writes simple stats for teacher and model rasters to a text file."""
    r_t = load_full_raster(field_name, int(teacher_delay))
    r_m = load_full_raster(field_name, int(model_delay))
    out_txt = os.path.join(REP_DIR, "representative_field_raster_stats.txt")

    with open(out_txt, "a") as f:
        f.write("============================================================\n")
        f.write(f"{tag}: {field_name}\n")
        f.write(f"TeacherDelay={teacher_delay}, ModelDelay={model_delay}\n")

        if r_t is None or r_m is None:
            f.write("Raster missing for one or both delays.\n")
        else:
            st_t = compute_raster_stats(r_t)
            st_m = compute_raster_stats(r_m)
            f.write(f"Teacher raster stats: {st_t}\n")
            f.write(f"Model raster stats:   {st_m}\n")
        f.write("\n")

    print(f"Updated: {out_txt}")

# ----------------------------
# Run: print summaries + save curves + save rasters + save summary CSV
# ----------------------------
dfm = df_merged.copy()

# Start fresh each run (optional; comment out if you want to append across runs)
stats_txt = os.path.join(REP_DIR, "representative_field_raster_stats.txt")
with open(stats_txt, "w") as f:
    f.write("# Representative-field raster stats\n\n")

summary_rows = []

for label, fld in selected:
    print_field_summary(dfm, fld, label)

    # Get delays from dfm (authoritative for the manuscript)
    row = dfm[dfm["Field"] == fld]
    if row.empty:
        continue
    r = row.iloc[0]
    teacher_delay = int(r["TeacherDelay"])
    model_delay   = int(r["ModelDelay"])
    abs_err       = int(r["AbsError"])

    # Save curve (also returns peaks from curves; useful for debugging)
    curve_info = save_field_curve(fld, label)

    # ✅ Raster diagnostics + stats (this was missing before)
    save_raster_triptych(fld, label, raw_delay=0, teacher_delay=teacher_delay, model_delay=model_delay)
    append_stats_to_file(fld, label, teacher_delay=teacher_delay, model_delay=model_delay)

    # ✅ Save a clean per-field summary row for writing Results/Table Sx
    summary_rows.append({
        "Label": label,
        "Field": fld,
        "TeacherDelay_df": teacher_delay,
        "ModelDelay_df": model_delay,
        "AbsError_df": abs_err,
        "TeacherDelay_curvePeak": curve_info["d_teacher"],
        "ModelDelay_curvePeak": curve_info["d_model"],
        "AbsError_curvePeak": curve_info["abs_err"],
        "n_delays_shared": curve_info["n_delays"],
        "PeakWidthHalfMax": r.get("PeakWidthHalfMax", np.nan),
        "RawPeakGap": r.get("RawPeakGap", np.nan),
        "RawPeakProminence": r.get("RawPeakProminence", np.nan),
    })

# Save summary CSV
summary_csv = os.path.join(REP_DIR, "selected_fields_summary.csv")
pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

print("\nDone: representative-field outputs saved in:", os.path.abspath(REP_DIR))
print("Summary CSV:", summary_csv)
print("Raster stats TXT:", stats_txt)



print("\nDone: Field-level error analysis outputs saved in:", os.path.abspath(ERROR_DIR))





# ================================================================
# EXTRA ANALYSIS (POST-HOC): Semivariogram (Success field only)
#  - Compares empirical semivariograms of:
#       (1) Raw raster (d=0)
#       (2) Model-aligned raster (d = ModelDelay from df_merged)
# ================================================================

# ----------------------------
# CONFIG (Success field)
# ----------------------------
RAW_DELAY = 0

# Base output folder (your ERROR_DIR should already be defined in your script)
SEMI_DIR = os.path.join(ERROR_DIR, "semivariogram_success_field")
os.makedirs(SEMI_DIR, exist_ok=True)

def _safe_name(s: str) -> str:
    return (
        s.replace(" ", "_")
         .replace("/", "_")
         .replace("\\", "_")
         .replace(":", "_")
         .replace("|", "_")
         .replace("—", "-")
         .replace("–", "-")
    )

def save_png_svg(fig, out_dir, base_name, dpi=500):
    fig.savefig(os.path.join(out_dir, f"{base_name}.png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{base_name}.svg"), bbox_inches="tight")

# ----------------------------
# Load model delay from df_merged
# ----------------------------
row = df_merged[df_merged["Field"] == SUCCESS_FIELD]
if row.empty:
    raise ValueError(f"[ERROR] SUCCESS_FIELD not found in df_merged: {SUCCESS_FIELD}")

MODEL_DELAY = int(row.iloc[0]["ModelDelay"])
TEACHER_DELAY = int(row.iloc[0]["TeacherDelay"])
ABS_ERR = int(row.iloc[0]["AbsError"])

print("============================================================")
print("Semivariogram (Success field)")
print("Field:", SUCCESS_FIELD)
print(f"TeacherDelay={TEACHER_DELAY}, ModelDelay={MODEL_DELAY}, AbsError={ABS_ERR}")
print("Output dir:", os.path.abspath(SEMI_DIR))
print("============================================================")





# ----------------------------
# Raster loaders (reuse your same prepared_patches structure)
# ----------------------------
def load_full_raster(field_name: str, delay: int):
    """Loads full raster for a given field and delay from prepared_patches/<field>/full_rasters/."""
    p = os.path.join(PATCH_DIR, field_name, "full_rasters", f"raster_{delay}.npy")
    if not os.path.exists(p):
        return None
    return np.load(p).astype(np.float32)

def raster_valid_mask(r: np.ndarray):
    """Valid pixels are non-zero in your pipeline."""
    return (r != 0)

# ----------------------------
# Empirical semivariogram from raster (random-pair estimator)
# ----------------------------
def empirical_semivariogram_from_raster(
    r: np.ndarray,
    mask: np.ndarray,
    cell_size: float = 1.0,      # set to your pixel size (e.g., meters) if known
    max_lag_cells: int = 30,
    n_pairs: int = 200000,
    n_bins: int = 25,
    seed: int = 42,
):
    """
    Empirical semivariogram using random pairs of valid pixels.
    Returns:
        lag_centers (cell_size units),
        gamma_mean (semivariance),
        counts (pairs per bin)
    """
    rng = np.random.default_rng(seed)

    coords = np.column_stack(np.where(mask))
    n = coords.shape[0]
    if n < 500:
        return None, None, None

    i1 = rng.integers(0, n, size=n_pairs)
    i2 = rng.integers(0, n, size=n_pairs)

    p1 = coords[i1]
    p2 = coords[i2]

    drow = (p1[:, 0] - p2[:, 0]).astype(np.float32)
    dcol = (p1[:, 1] - p2[:, 1]).astype(np.float32)
    dist_cells = np.sqrt(drow**2 + dcol**2)

    dist_cells_max = float(max_lag_cells)
    keep = dist_cells <= dist_cells_max
    if keep.sum() < 2000:
        return None, None, None

    dist_cells = dist_cells[keep]
    p1 = p1[keep]
    p2 = p2[keep]

    z1 = r[p1[:, 0], p1[:, 1]].astype(np.float32)
    z2 = r[p2[:, 0], p2[:, 1]].astype(np.float32)

    gamma = 0.5 * (z1 - z2) ** 2

    bins = np.linspace(0, dist_cells_max, n_bins + 1)
    bin_idx = np.digitize(dist_cells, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    gamma_sum = np.bincount(bin_idx, weights=gamma, minlength=n_bins)
    counts = np.bincount(bin_idx, minlength=n_bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_mean = gamma_sum / counts

    # drop bins with too few pairs
    gamma_mean[counts < 50] = np.nan

    lag_centers_cells = 0.5 * (bins[:-1] + bins[1:])
    lag_centers = lag_centers_cells * float(cell_size)

    return lag_centers, gamma_mean, counts

# ----------------------------
# Load rasters (raw vs model-aligned)
# ----------------------------
r_raw  = load_full_raster(SUCCESS_FIELD, RAW_DELAY)
r_pred = load_full_raster(SUCCESS_FIELD, MODEL_DELAY)

if r_raw is None:
    raise FileNotFoundError(f"[ERROR] Missing raw raster (delay={RAW_DELAY}) for {SUCCESS_FIELD}")
if r_pred is None:
    raise FileNotFoundError(f"[ERROR] Missing model raster (delay={MODEL_DELAY}) for {SUCCESS_FIELD}")

# Use intersection mask for fair comparison
mask = raster_valid_mask(r_raw) & raster_valid_mask(r_pred)
n_valid = int(mask.sum())
print("Valid pixels (intersection):", n_valid)

if n_valid < 800:
    raise ValueError(f"[ERROR] Too few valid pixels for variogram computation: {n_valid}")

# ----------------------------
# Compute variograms
# ----------------------------
CELL_SIZE = 1.0       # <- set to your raster pixel size if known (e.g., meters)
MAX_LAG_CELLS = 30
N_PAIRS = 200000
N_BINS = 25

lag_raw, gamma_raw, cnt_raw = empirical_semivariogram_from_raster(
    r_raw, mask, cell_size=CELL_SIZE, max_lag_cells=MAX_LAG_CELLS,
    n_pairs=N_PAIRS, n_bins=N_BINS, seed=42
)

lag_pred, gamma_pred, cnt_pred = empirical_semivariogram_from_raster(
    r_pred, mask, cell_size=CELL_SIZE, max_lag_cells=MAX_LAG_CELLS,
    n_pairs=N_PAIRS, n_bins=N_BINS, seed=43
)

if lag_raw is None or lag_pred is None:
    raise RuntimeError("[ERROR] Variogram computation failed (insufficient pairs after filtering).")

# ----------------------------
# Plot and save
# ----------------------------
fig = plt.figure(figsize=(7.5, 5))
ax = fig.gca()

ax.plot(lag_raw,  gamma_raw,  "--", label=f"Raw (d={RAW_DELAY})", linewidth=2)
ax.plot(lag_pred, gamma_pred, "-",  label=f"Model-aligned (d={MODEL_DELAY})", linewidth=2)  


x_units = "cells" if CELL_SIZE == 1.0 else "units"
ax.set_xlabel(f"Lag distance ({x_units})")
ax.set_ylabel("Semivariance")
ax.legend()
fig.tight_layout()

base = _safe_name(f"semivariogram_raw{RAW_DELAY}_model{MODEL_DELAY}_{SUCCESS_FIELD}")
save_png_svg(fig, SEMI_DIR, base, dpi=500)

plt.show()
plt.close(fig)

df_vario = pd.DataFrame({
    "lag_cells": lag_raw,          # same lag grid
    "gamma_raw": gamma_raw,
    "gamma_pred": gamma_pred,
    "count_raw": cnt_raw,
    "count_pred": cnt_pred,
})

csv_path = os.path.join(SEMI_DIR, _safe_name(f"semivariogram_values_{SUCCESS_FIELD}.csv"))
df_vario.to_csv(csv_path, index=False)
print("Saved variogram values to:", os.path.abspath(csv_path))

# ----------------------------
# Save a small text summary for traceability
# ----------------------------
summary_path = os.path.join(SEMI_DIR, "semivariogram_summary.txt")
with open(summary_path, "w") as f:
    f.write("Semivariogram comparison (Success field)\n")
    f.write(f"Field: {SUCCESS_FIELD}\n")
    f.write(f"TeacherDelay: {TEACHER_DELAY}\n")
    f.write(f"ModelDelay: {MODEL_DELAY}\n")
    f.write(f"AbsError: {ABS_ERR}\n")
    f.write(f"RAW_DELAY: {RAW_DELAY}\n")
    f.write(f"Valid pixels (intersection): {n_valid}\n")
    f.write(f"CELL_SIZE: {CELL_SIZE}\n")
    f.write(f"MAX_LAG_CELLS: {MAX_LAG_CELLS}\n")
    f.write(f"N_PAIRS: {N_PAIRS}\n")
    f.write(f"N_BINS: {N_BINS}\n")

print("Saved semivariogram plot to:", os.path.abspath(SEMI_DIR))
print("Saved summary to:", os.path.abspath(summary_path))


# ================================================================
# AP–AN DISTANCE DISTRIBUTION (ROBUST VERSION)
# ================================================================
def compute_AP_AN_distances(encoder, field_set, n_samples=2000, max_del=MAX_DELAY):
    """
    Computes distances using SAME embedding pipeline as inference.
    AP = anchor vs positive (nearby delay)
    AN = anchor vs negative (far delay)
    """
    D_AP, D_AN = [], []

    for _ in range(n_samples):
        fld, n_patches = random.choice(field_set)
        idx = random.randint(0, n_patches - 1)

        # ----------------------------
        # Choose anchor delay
        # ----------------------------
        d_anchor = random.randint(-max_del, max_del)

        # ----------------------------
        # Positive: nearby delay (±1)
        # ----------------------------
        d_positive = np.clip(
            d_anchor + random.choice([-1, 1]),
            -max_del,
            max_del
        )

        # ----------------------------
        # Negative: far delay (|Δd| ≥ 8)
        # ----------------------------
        neg_choices = [
            d for d in range(-max_del, max_del + 1)
            if abs(d - d_anchor) >= 8
        ]
        d_negative = random.choice(neg_choices)

        # ----------------------------
        # Load patches
        # ----------------------------
        pa = load_patch(fld, idx, d_anchor)[None, ...]
        pp = load_patch(fld, idx, d_positive)[None, ...]
        pn = load_patch(fld, idx, d_negative)[None, ...]

        # ----------------------------
        # Forward pass (encoder only)
        # ----------------------------
        ea = encoder(pa, training=False)[0].numpy()
        ep = encoder(pp, training=False)[0].numpy()
        en = encoder(pn, training=False)[0].numpy()

        # ----------------------------
        # L2 distances (interpretable)
        # ----------------------------
        D_AP.append(np.linalg.norm(ea - ep))
        D_AN.append(np.linalg.norm(ea - en))

    return np.array(D_AP), np.array(D_AN)


# Compute AP–AN distances
D_AP, D_AN = compute_AP_AN_distances(encoder, test_fields, n_samples=2000)

print("AP mean:", np.mean(D_AP))
print("AN mean:", np.mean(D_AN))
print("AP < AN percentage:", np.mean(D_AP < D_AN) * 100)


# -------------------------------------------------
# Create output directory
# -------------------------------------------------
EMB_DIR = "Embedding_analysis"
os.makedirs(EMB_DIR, exist_ok=True)

# -------------------------------------------------
# KDE PLOT
# -------------------------------------------------
plt.figure(figsize=(8,4))
sns.kdeplot(D_AP, label="AP (Anchor–Positive)", shade=True, linewidth=2)
sns.kdeplot(D_AN, label="AN (Anchor–Negative)", shade=True, linewidth=2)

plt.xlabel("L2 Distance")
plt.ylabel("Density")
plt.legend()
#plt.tight_layout()

# Save high-resolution raster + vector
plt.savefig(os.path.join(EMB_DIR, "kde_AP_AN.png"), dpi=500, bbox_inches="tight")
plt.savefig(os.path.join(EMB_DIR, "kde_AP_AN.svg"), bbox_inches="tight")

plt.show()



# -------------------------------------------------
# Prepare dataframe
# -------------------------------------------------
df = pd.DataFrame({
    "Distance": np.concatenate([D_AP, D_AN]),
    "Type":     ["AP"] * len(D_AP) + ["AN"] * len(D_AN)
})

# -------------------------------------------------
# VIOLIN PLOT
# -------------------------------------------------
plt.figure(figsize=(8,5))
sns.violinplot(data=df, x="Type", y="Distance", palette="Set2")
plt.xlabel("")
plt.ylabel("L2 Distance")
plt.tight_layout()

plt.savefig(os.path.join(EMB_DIR, "violin_AP_AN.png"), dpi=500, bbox_inches="tight")
plt.savefig(os.path.join(EMB_DIR, "violin_AP_AN.svg"), bbox_inches="tight")
plt.show()

# -------------------------------------------------
# BOX PLOT
# -------------------------------------------------
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Type", y="Distance", palette="Set2")
plt.xlabel("")
plt.ylabel("L2 Distance")
plt.tight_layout()

plt.savefig(os.path.join(EMB_DIR, "box_AP_AN.png"), dpi=500, bbox_inches="tight")
plt.savefig(os.path.join(EMB_DIR, "box_AP_AN.svg"), bbox_inches="tight")
plt.show()


# ================================================================
# Embedding_analysis/
# ================================================================
EMB_DIR = "Embedding_analysis"
PLOT_DIR = os.path.join(EMB_DIR, "UMAP_tSNE_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

def _save_png_svg(fig, out_dir, base_name, dpi=500):
    fig.savefig(os.path.join(out_dir, f"{base_name}.png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{base_name}.svg"), bbox_inches="tight")

def _safe_name(s: str) -> str:
    # Minimal safe filename helper (avoid spaces and problematic chars)
    return (
        s.replace(" ", "_")
         .replace("/", "_")
         .replace("\\", "_")
         .replace(":", "_")
         .replace("|", "_")
         .replace("—", "-")
         .replace("–", "-")
    )

# ================================================================
# SAMPLE EMBEDDINGS FOR t-SNE / UMAP (GLOBAL: many fields)
# ================================================================
def sample_embeddings_with_delay_and_smoothness(
    encoder,
    field_set,
    max_del=MAX_DELAY,
    k_samples=600
):
    """
    Samples embeddings across many fields.
    Each point = one patch at one delay.
    Returns:
        X:          (k_samples, embed_dim)
        delays:     (k_samples,)
        smoothness: (k_samples,)
    """
    embeddings = []
    delays = []
    smoothness = []

    for _ in range(k_samples):
        fld, n_patches = random.choice(field_set)
        idx = random.randint(0, n_patches - 1)

        d = random.randint(-max_del, max_del)

        patch = load_patch(fld, idx, d)[None, ...]
        emb, _ = encoder(patch, training=False)
        emb = emb.numpy()[0]

        embeddings.append(emb)
        delays.append(d)
        smoothness.append(field_smoothness[fld].get(d, 0.0))

    return np.array(embeddings), np.array(delays), np.array(smoothness)

X_global, delays_global, smoothness_global = sample_embeddings_with_delay_and_smoothness(
    encoder, test_fields, max_del=MAX_DELAY, k_samples=600
)

# ================================================================
# UMAP (GLOBAL)
# ================================================================
reducer = umap.UMAP(
    n_neighbors=20,
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)
X_umap = reducer.fit_transform(X_global)

# ---- colored by delay ----
fig = plt.figure(figsize=(8, 6))
sc = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=delays_global, cmap="jet", s=20, alpha=0.9)
plt.colorbar(sc, label="Delay (shifts)")
plt.tight_layout()
_save_png_svg(fig, PLOT_DIR, "umap_delay", dpi=500)
plt.show()
plt.close(fig)

# ---- colored by Gaussian smoothness ----
fig = plt.figure(figsize=(8, 6))
sc = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=smoothness_global, cmap="viridis", s=20, alpha=0.9)
plt.colorbar(sc, label="Gaussian Smoothness")
plt.tight_layout()
_save_png_svg(fig, PLOT_DIR, "umap_smoothness", dpi=500)
plt.show()
plt.close(fig)

# ================================================================
# ONE FIELD ONLY — UMAP (kept; useful for supplementary)
# ================================================================
def sample_embeddings_single_field(
    encoder,
    field_name,
    max_del=MAX_DELAY,
    k_samples=500
):
    """
    Samples embeddings from ONE FIELD ONLY.
    Each point = one patch at one delay.
    Returns:
        X (embeddings)      shape (k_samples, embed_dim)
        delays              shape (k_samples,)
        smoothness          shape (k_samples,)
    """
    embeddings = []
    delays = []
    smoothness = []

    # Load delay_0 just to count patches
    base_arr = np.load(os.path.join(PATCH_DIR, field_name, "delay_0.npy"))
    n_patches = base_arr.shape[0]

    for _ in range(k_samples):
        pidx = random.randint(0, n_patches - 1)
        d = random.randint(-max_del, max_del)

        patch = load_patch(field_name, pidx, d)[None, ...]
        emb_vec, _ = encoder(patch, training=False)
        emb = emb_vec.numpy()[0]

        embeddings.append(emb)
        delays.append(d)
        smoothness.append(field_smoothness[field_name].get(d, 0.0))

    return np.array(embeddings), np.array(delays), np.array(smoothness)

FIELD = "Demo_UMAP"

X_f, d_f, sm_f = sample_embeddings_single_field(
    encoder, FIELD, max_del=MAX_DELAY, k_samples=500
)

# ---- UMAP (one field) ----
umap_f = umap.UMAP(
    n_neighbors=20,
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)
X_umap_f = umap_f.fit_transform(X_f)

fig = plt.figure(figsize=(8, 6))
sc = plt.scatter(X_umap_f[:, 0], X_umap_f[:, 1], c=d_f, cmap="jet", s=25, alpha=0.85)
plt.colorbar(sc, label="Delay (shifts)")
plt.tight_layout()
_save_png_svg(fig, PLOT_DIR, f"umap_onefield_delay_{_safe_name(FIELD)}", dpi=500)
plt.show()
plt.close(fig)

fig = plt.figure(figsize=(8, 6))
sc = plt.scatter(X_umap_f[:, 0], X_umap_f[:, 1], c=sm_f, cmap="viridis", s=25, alpha=0.85)
plt.colorbar(sc, label="Gaussian Smoothness")
plt.tight_layout()
_save_png_svg(fig, PLOT_DIR, f"umap_onefield_smoothness_{_safe_name(FIELD)}", dpi=500)
plt.show()
plt.close(fig)

# ================================================================
# ONE-PATCH TRAJECTORY ACROSS ALL DELAYS (kept; good for supplement)
# ================================================================
def compute_per_delay_embeddings(encoder, field_name, patch_idx, max_del=MAX_DELAY):
    """
    Returns embeddings for ONE PATCH across all delays.
    """
    embeddings = []
    delays = []

    for d in range(-max_del, max_del + 1):
        patch = load_patch(field_name, patch_idx, d)[None, ...]
        emb_vec, _ = encoder(patch, training=False)
        emb = emb_vec.numpy()[0]
        embeddings.append(emb)
        delays.append(d)

    return np.array(embeddings), np.array(delays)

FIELD = "G1_Panola_P13_Soybeans2_18"
PATCH_IDX = 8

E_patch, D_patch = compute_per_delay_embeddings(
    encoder, FIELD, PATCH_IDX, max_del=MAX_DELAY
)

# ---- UMAP trajectory ----
umap_tr = umap.UMAP(
    n_neighbors=10,
    min_dist=0.05,
    metric="euclidean",
    random_state=42
)
E_umap_patch = umap_tr.fit_transform(E_patch)

fig = plt.figure(figsize=(8, 6))
sc = plt.scatter(E_umap_patch[:, 0], E_umap_patch[:, 1], c=D_patch, cmap="jet", s=60,
                 vmin=-MAX_DELAY, vmax=MAX_DELAY)
plt.plot(E_umap_patch[:, 0], E_umap_patch[:, 1], "-k", alpha=0.6)
plt.scatter(E_umap_patch[0, 0],  E_umap_patch[0, 1],  c="black", s=120, marker="s", label=f"Start ({D_patch[0]})")
plt.scatter(E_umap_patch[-1, 0], E_umap_patch[-1, 1], c="black", s=120, marker="*", label=f"End ({D_patch[-1]})")
plt.colorbar(sc, label="Delay (shifts)")
plt.legend()
plt.tight_layout()
_save_png_svg(fig, PLOT_DIR, f"umap_patchtraj_delay_{_safe_name(FIELD)}_p{PATCH_IDX}", dpi=500)
plt.show()
plt.close(fig)

print("Saved all UMAP/t-SNE figures to:", os.path.abspath(PLOT_DIR))




# ================================================================
# POST-HOC EVALUATION
# ================================================================

def soft_argmax_delay(scores_dict, tau=0.3):
    """
    scores_dict: delay -> normalized score
    Returns:
      d_int  : rounded integer delay
      d_cont : continuous expected delay
    """
    delays = np.array(sorted(scores_dict.keys()), dtype=np.float32)
    s = np.array([scores_dict[d] for d in delays], dtype=np.float32)

    # temperature-scaled softmax (stable)
    s = (s - np.max(s)) / max(tau, 1e-6)
    w = np.exp(s)
    w = w / (np.sum(w) + 1e-12)

    d_cont = float(np.sum(w * delays))
    d_int = int(np.round(d_cont))
    return d_int, d_cont


def eval_delay_picker_on_test_fields(tau=0.3, out_dir="gaussian_vs_model_prediction"):
    os.makedirs(out_dir, exist_ok=True)

    rows = []

    for fld, _ in test_fields:
        model_curve, teacher_curve = get_model_curve_for_field(fld, encoder)
        if not model_curve or not teacher_curve:
            continue

        # shared delays only (safety)
        delays = sorted(set(model_curve.keys()) & set(teacher_curve.keys()))
        if len(delays) < 3:
            continue

        # teacher peak (keep as hard argmax)
        t_vals = [teacher_curve[d] for d in delays]
        d_teacher = int(delays[int(np.argmax(t_vals))])

        # model peak: hard argmax
        m_vals = [model_curve[d] for d in delays]
        d_model_hard = int(delays[int(np.argmax(m_vals))])

        # model peak: soft-argmax
        model_dict = {d: float(model_curve[d]) for d in delays}
        d_model_soft, d_model_soft_cont = soft_argmax_delay(model_dict, tau=tau)

        # Clip (optional safety): keep within candidate delay set bounds
        d_model_soft = int(np.clip(d_model_soft, min(delays), max(delays)))

        rows.append({
            "Field": fld,
            "TeacherDelay": d_teacher,
            "ModelDelay_HardArgmax": d_model_hard,
            "ModelDelay_SoftArgmax": d_model_soft,
            "ModelDelay_SoftArgmax_Continuous": d_model_soft_cont,
            "AbsError_Hard": abs(d_model_hard - d_teacher),
            "AbsError_Soft": abs(d_model_soft - d_teacher),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[POST-HOC] No valid test fields found for evaluation.")
        return df

    # ----------------------------
    # Summary metrics
    # ----------------------------
    def summarize(abs_err, label):
        mae = float(np.mean(abs_err))
        exact = float(np.mean(abs_err == 0) * 100.0)
        within1 = float(np.mean(abs_err <= 1) * 100.0)
        within2 = float(np.mean(abs_err <= 2) * 100.0)
        print(f"{label}")
        print(f"  MAE:          {mae:.3f}")
        print(f"  Exact (%):    {exact:.2f}")
        print(f"  Within ±1 (%):{within1:.2f}")
        print(f"  Within ±2 (%):{within2:.2f}")
        return mae, exact, within1, within2

    print("\n===================================================")
    print(" POST-HOC COMPARISON: Hard argmax vs Soft-argmax")
    print("===================================================")
    print(f"Evaluated test fields: {len(df)}")
    print(f"Soft-argmax tau: {tau}")
    print("---------------------------------------------------")

    summarize(df["AbsError_Hard"].values, "Hard argmax (baseline)")
    print("---------------------------------------------------")
    summarize(df["AbsError_Soft"].values, "Soft-argmax (post-hoc)")

    # how many fields improved/worsened
    improved = int(np.sum(df["AbsError_Soft"] < df["AbsError_Hard"]))
    worsened = int(np.sum(df["AbsError_Soft"] > df["AbsError_Hard"]))
    same     = int(np.sum(df["AbsError_Soft"] == df["AbsError_Hard"]))

    print("---------------------------------------------------")
    print(f"Fields improved: {improved}")
    print(f"Fields worsened: {worsened}")
    print(f"Fields unchanged:{same}")
    print("===================================================\n")

    # save CSV
    out_csv = os.path.join(out_dir, f"softargmax_comparison_tau{str(tau).replace('.','p')}.csv")
    df.to_csv(out_csv, index=False)
    print("Saved:", os.path.abspath(out_csv))

    return df


# ---------------------------------------------------
# Run post-hoc evaluation (Soft-argmax, τ = 1.5)
# ---------------------------------------------------
df_soft_15 = eval_delay_picker_on_test_fields(
    tau=1.5,
    out_dir=OUT_DIR
)

# ---------------------------------------------------
# Optional sanity statistics (based on τ = 1.5 results)
# ---------------------------------------------------
df = df_soft_15.copy()

print("Hard argmax pred mean/std:",
      df["ModelDelay_HardArgmax"].mean(),
      df["ModelDelay_HardArgmax"].std())

print("Soft argmax pred mean/std:",
      df["ModelDelay_SoftArgmax"].mean(),
      df["ModelDelay_SoftArgmax"].std())

print("Teacher mean/std:",
      df["TeacherDelay"].mean(),
      df["TeacherDelay"].std())

print("\nSoft argmax prediction counts:")
print(df["ModelDelay_SoftArgmax"].value_counts().sort_index())

abs_err_soft = np.abs(df["ModelDelay_SoftArgmax"] - df["TeacherDelay"])
abs_err_zero = np.abs(0 - df["TeacherDelay"])

print("MAE soft:", abs_err_soft.mean())
print("MAE always-0:", abs_err_zero.mean())
print("Within ±2 soft:", (abs_err_soft <= 2).mean() * 100)
print("Within ±2 always-0:", (abs_err_zero <= 2).mean() * 100)


# ================================================================
# PLOTS (PNG 500 dpi + SVG): Hard argmax vs Soft-argmax (τ = 1.5)
#   Standalone diagnostic visualization (NO effect on inference)
# ================================================================
TAU = 1.5  # must match evaluation above

PLOT_DIR = os.path.join(
    OUT_DIR,
    f"softargmax_plots_tau{str(TAU).replace('.', 'p')}"
)
os.makedirs(PLOT_DIR, exist_ok=True)


def save_png_svg(fig, out_dir, base_name, dpi=500):
    fig.savefig(
        os.path.join(out_dir, f"{base_name}.png"),
        dpi=dpi,
        bbox_inches="tight"
    )
    fig.savefig(
        os.path.join(out_dir, f"{base_name}.svg"),
        bbox_inches="tight"
    )


# ----------------------------
# Data
# ----------------------------
abs_hard = df["AbsError_Hard"].values
abs_soft = df["AbsError_Soft"].values
max_e = int(max(abs_hard.max(), abs_soft.max()))


# ----------------------------
# 1) Histogram of absolute errors
# ----------------------------
fig = plt.figure(figsize=(7.5, 4.8))
ax = fig.gca()

bins = np.arange(-0.5, max_e + 1.5, 1)

ax.hist(abs_hard, bins=bins,
        edgecolor="black", label="Hard argmax")
ax.hist(abs_soft, bins=bins,
        edgecolor="black", label=f"Soft-argmax (τ={TAU})")

ax.set_xlabel("Absolute delay error |Model − Teacher| (shifts)")
ax.set_ylabel("Number of fields")
ax.legend()

fig.tight_layout()
save_png_svg(fig, PLOT_DIR,
              f"hist_abs_error_hard_vs_soft_tau{str(TAU).replace('.', 'p')}")
plt.close(fig)


# ----------------------------
# 2) Error category bar chart
# ----------------------------
def err_counts(arr):
    return [
        int(np.sum(arr == 0)),
        int(np.sum(arr == 1)),
        int(np.sum(arr == 2)),
        int(np.sum(arr >= 3)),
    ]


cats = ["0", "1", "2", "≥3"]
hard_counts = err_counts(abs_hard)
soft_counts = err_counts(abs_soft)

x = np.arange(len(cats))
w = 0.38

fig = plt.figure(figsize=(7.5, 4.8))
ax = fig.gca()

ax.bar(x - w/2, hard_counts, width=w,
       edgecolor="black", label="Hard argmax")
ax.bar(x + w/2, soft_counts, width=w,
       edgecolor="black", label=f"Soft-argmax (τ={TAU})")

ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_xlabel("Absolute error category (shifts)")
ax.set_ylabel("Number of fields")
ax.legend()

fig.tight_layout()
save_png_svg(fig, PLOT_DIR,
              f"bar_error_categories_hard_vs_soft_tau{str(TAU).replace('.', 'p')}")
plt.close(fig)

# ----------------------------
# 3) Scatter: Hard error vs Soft error (WITH JITTER)
# ----------------------------
rng = np.random.default_rng(42)

jitter = 0.15  # small, does NOT change interpretation
x_jit = abs_hard + rng.uniform(-jitter, jitter, size=len(abs_hard))
y_jit = abs_soft + rng.uniform(-jitter, jitter, size=len(abs_soft))

fig = plt.figure(figsize=(7.5, 4.8))
ax = fig.gca()

ax.scatter(x_jit, y_jit, alpha=0.6)
ax.plot([0, max_e], [0, max_e], linestyle="--")

ax.set_xlabel("Hard argmax absolute error")
ax.set_ylabel("Soft-argmax absolute error")

fig.tight_layout()
save_png_svg(fig, PLOT_DIR,
             f"scatter_hard_vs_soft_error_jitter_tau{str(TAU).replace('.', 'p')}")
plt.close(fig)



# ================================================================
# SAVE TEST FIELD LIST (CSV) — add at END of script
# ================================================================
# Where to save (keep inside your existing Field_level_error_analysis)
TESTLIST_DIR = os.path.join(ERROR_DIR, "test_field_list")
os.makedirs(TESTLIST_DIR, exist_ok=True)

# test_fields is a list of tuples: [(field_name, n_patches), ...]
df_testlist = pd.DataFrame(test_fields, columns=["Field", "NumPatches"])

# Optional: also store which split it is
df_testlist["Split"] = "test"

out_csv = os.path.join(TESTLIST_DIR, "test_fields.csv")
df_testlist.to_csv(out_csv, index=False)

print("Saved test field list CSV to:", os.path.abspath(out_csv))
print("N test fields:", len(df_testlist))


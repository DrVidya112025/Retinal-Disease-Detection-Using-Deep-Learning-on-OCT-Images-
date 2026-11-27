"""
Retinal OCT Disease Detection - Streamlit dashboard (Steps 1-9)
EDUCATIONAL PROTOTYPE. NOT A CLINICAL DIAGNOSTIC TOOL.

Run:  streamlit run app.py
Needs: step6_best_model.pt (from Step 6) next to this file, plus optimization_config.json if it is a separate file.
"""
import io
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torchvision import models

APP_DIR = Path(__file__).resolve().parent
MODEL_FILE, CONFIG_FILE = APP_DIR / "step6_best_model.pt", APP_DIR / "optimization_config.json"
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
DISCLAIMER = ("<b>Educational prototype - not a clinical diagnostic tool.</b> Predictions come from a research model trained on a "
              "public dataset. They must not be used to diagnose, treat or make decisions about any patient. "
              "Always consult a qualified eye-care professional.")
INFO = {
    "CNV": ("Choroidal neovascularization", "New, leaky blood vessels under the retina; on OCT often fluid or a bulge near the macula."),
    "DME": ("Diabetic macular edema", "Fluid-filled cysts and thickening inside the retina caused by diabetes."),
    "DRUSEN": ("Drusen", "Small yellow deposits under the retinal pigment layer, an early sign of macular degeneration."),
    "NORMAL": ("Normal retina", "Smooth foveal dip with regular layers and no fluid or deposits."),
}
COLORS = {"CNV": "#c8553d", "DME": "#e0a030", "DRUSEN": "#3f7d6b", "NORMAL": "#3b6ea5"}

st.set_page_config(page_title="OCT Disease Detection (educational prototype)", page_icon="\U0001F441", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1150px;}
.disc {border-left: 5px solid #c8553d; background: rgba(200,85,61,.10); padding: .7rem 1rem; border-radius: 4px; margin-bottom: 1rem; font-size: .93rem;}
.pred {border: 1px solid rgba(128,128,128,.35); border-radius: 8px; padding: 1rem 1.2rem;}
.pred h2 {margin: 0 0 .2rem 0;}
.small {font-size: .85rem; opacity: .8;}
</style>""", unsafe_allow_html=True)


def banner():
    st.markdown(f'<div class="disc">{DISCLAIMER}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------- model
def _act(name):
    return {"relu": nn.ReLU(), "leaky_relu": nn.LeakyReLU(0.1), "gelu": nn.GELU(), "tanh": nn.Tanh()}[name]


class OCTCNN(nn.Module):
    """Custom CNN from Steps 4 and 6 (Step 4: Tanh + average pooling, 64 px; Step 6 variants: ReLU + max pooling)."""
    def __init__(self, n_classes=4, in_channels=1, width=16, bn=True, dropout=0.4, conv_dropout=0.0, fc=128, act="relu", pool="max"):
        super().__init__()
        channels, layers, c_in = (width, 2 * width, 4 * width, 8 * width), [], in_channels
        for c_out in channels:
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=not bn)]
            if bn: layers += [nn.BatchNorm2d(c_out)]
            layers += [_act(act), nn.MaxPool2d(2) if pool == "max" else nn.AvgPool2d(2)]
            if conv_dropout > 0: layers += [nn.Dropout2d(conv_dropout)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c_in, fc), _act(act), nn.Dropout(dropout), nn.Linear(fc, n_classes))
    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


def _head(m, arch):
    return {"resnet18": lambda: m.fc, "vgg16": lambda: m.classifier[6], "mobilenet_v2": lambda: m.classifier[1],
            "densenet121": lambda: m.classifier}[arch]()


def _set_head(m, arch, new):
    if arch == "resnet18": m.fc = new
    elif arch == "vgg16": m.classifier[6] = new
    elif arch == "mobilenet_v2": m.classifier[1] = new
    else: m.classifier = new


def infer_config(sd):
    """Used when no config is available: guess the architecture from the weight names."""
    keys = list(sd)
    cfg = {"best_model": "model (config inferred)", "img_size": 112, "class_names": CLASSES, "pixel_mean": 0.19, "pixel_std": 0.21,
           "hyperparameters": {}, "inferred": True}
    if any(k.startswith("layer4") for k in keys):
        cfg.update(kind="pt", architecture="resnet18"); cfg["hyperparameters"]["dropout"] = 0.5 if "fc.1.weight" in sd else 0.0
    elif any(k.startswith("features.denseblock") for k in keys):
        cfg.update(kind="pt", architecture="densenet121"); cfg["hyperparameters"]["dropout"] = 0.5 if "classifier.1.weight" in sd else 0.0
    elif "classifier.6.weight" in sd or "classifier.6.1.weight" in sd:
        cfg.update(kind="pt", architecture="vgg16"); cfg["hyperparameters"]["dropout"] = 0.5 if "classifier.6.1.weight" in sd else 0.0
    elif any(k.startswith("features.18") for k in keys):
        cfg.update(kind="pt", architecture="mobilenet_v2"); cfg["hyperparameters"]["dropout"] = 0.5 if "classifier.1.1.weight" in sd else 0.0
    elif "features.0.weight" in sd and sd["features.0.weight"].shape[1] == 1:
        cfg.update(kind="custom", architecture="custom_cnn")
        cfg["hyperparameters"] = {"width": int(sd["features.0.weight"].shape[0]), "fc": int(sd["classifier.1.weight"].shape[0]),
                                  "bn": "features.1.running_mean" in sd}
    else:
        raise ValueError("Could not recognise this checkpoint's architecture.")
    return cfg


def build_model(cfg, n_classes):
    hp = cfg.get("hyperparameters", {})
    if cfg["kind"] == "custom":
        m = OCTCNN(n_classes, 1, hp.get("width", 16), hp.get("bn", True), hp.get("dropout", 0.4), hp.get("conv_dropout", 0.0), hp.get("fc", 128),
                   hp.get("act", "relu"), hp.get("pool", "max"))
    else:
        arch = cfg["architecture"]
        m = getattr(models, arch)(weights=None)
        lin = nn.Linear(_head(m, arch).in_features, n_classes)
        d = hp.get("dropout", 0.0)
        _set_head(m, arch, nn.Sequential(nn.Dropout(d), lin) if d > 0 else lin)
    for mod in m.modules():
        if isinstance(mod, (nn.ReLU, nn.ReLU6)): mod.inplace = False
    return m


# Recorded settings of the Step 4 checkpoint (from the Step 4 notebook: Tanh, average pooling, 64 px, train-pixel statistics).
STEP4_CFG = {"best_model": "Step 4: custom CNN (64 px)", "kind": "custom", "architecture": "custom_cnn", "img_size": 64,
             "hyperparameters": {"width": 16, "fc": 128, "dropout": 0.4, "bn": True, "act": "tanh", "pool": "avg"},
             "pixel_mean": 0.1904, "pixel_std": 0.2081, "class_names": CLASSES}
MODEL_SPECS = {
    "Step 6: ResNet18 (tuned)": {"file": "step6_best_model.pt", "cfg": "optimization_config.json", "default": None},
    "Step 4: custom CNN": {"file": "step4_custom_cnn.pt", "cfg": "custom_cnn_config.json", "default": STEP4_CFG},
}


def normalize_cfg(cfg):
    """Accept the Step 4 notebook's custom_cnn_config.json layout as well as the Step 6 layout."""
    a = cfg.get("architecture")
    if isinstance(a, dict):
        return {"best_model": "Step 4: custom CNN", "kind": "custom", "architecture": "custom_cnn", "img_size": int(cfg.get("img_size", 64)),
                "hyperparameters": {"width": a["channels"][0], "fc": a["fc_units"], "dropout": a["dropout"], "bn": True, "act": a["act"], "pool": a["pool"]},
                "pixel_mean": cfg["pixel_mean"], "pixel_std": cfg["pixel_std"], "class_names": cfg.get("class_names", CLASSES)}
    return cfg


@st.cache_resource(show_spinner="Loading model ...")
def load_bundle(ckpt_bytes: bytes, cfg_bytes, default_cfg_json):
    ckpt = torch.load(io.BytesIO(ckpt_bytes), map_location="cpu", weights_only=True)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if cfg_bytes:
        cfg = normalize_cfg(json.loads(cfg_bytes))
    elif isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
        cfg = normalize_cfg(ckpt["config"])
    elif default_cfg_json:
        cfg = json.loads(default_cfg_json)
    else:
        cfg = infer_config(sd)
    classes = cfg.get("class_names", CLASSES)
    model = build_model(cfg, len(classes))
    model.load_state_dict(sd)
    model.eval()
    return {"model": model, "cfg": cfg, "classes": classes, "size": int(cfg.get("img_size", 112)),
            "rgb": cfg["kind"] != "custom", "mean": float(cfg.get("pixel_mean", 0.19)), "std": float(cfg.get("pixel_std", 0.21)),
            "arch": cfg["architecture"], "name": cfg.get("best_model", cfg["architecture"]),
            "params": sum(p.numel() for p in model.parameters())}


def get_bundles():
    """Load every model that is available: files next to the app, or uploaded in the sidebar. Returns {label: bundle}."""
    found = {}
    with st.sidebar.expander("Model files", expanded=not any((APP_DIR / v["file"]).exists() for v in MODEL_SPECS.values())):
        for label, spec in MODEL_SPECS.items():
            path, cpath = APP_DIR / spec["file"], APP_DIR / spec["cfg"]
            ckpt_bytes = path.read_bytes() if path.exists() else None
            cfg_bytes = cpath.read_bytes() if cpath.exists() else None
            if ckpt_bytes is None:
                up = st.file_uploader(f"{spec['file']} ({label})", type=["pt", "pth"], key="ck_" + spec["file"])
                upc = st.file_uploader(f"{spec['cfg']} (optional)", type=["json"], key="cf_" + spec["file"])
                if up is not None:
                    ckpt_bytes, cfg_bytes = up.getvalue(), (upc.getvalue() if upc else None)
            else:
                st.caption(f"Found {spec['file']}")
            if ckpt_bytes is None:
                continue
            try:
                found[label] = load_bundle(ckpt_bytes, cfg_bytes, json.dumps(spec["default"]) if spec["default"] else None)
            except Exception as e:                            # show the reason instead of crashing
                st.error(f"{label}: could not load ({e})")
    return found


def benchmark(b, reps=20):
    """Latency of one model on this machine's CPU (after a short warm-up), for batch 1 and batch 32."""
    m = b["model"].eval()
    out = {}
    with torch.no_grad():
        for bs in (1, 32):
            x = torch.randn(bs, 3 if b["rgb"] else 1, b["size"], b["size"])
            for _ in range(3): m(x)
            t0 = time.perf_counter()
            for _ in range(reps): m(x)
            out[bs] = (time.perf_counter() - t0) / reps / bs * 1000
    return out



# --------------------------------------------------------------------------------------------- inference
IM_MEAN, IM_STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preprocess(pil, b):
    """Same as training: grey-level, square resize (bilinear), then the model's normalisation."""
    g = ImageOps.exif_transpose(pil).convert("L").resize((b["size"], b["size"]), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(g, dtype=np.uint8).copy()).float().div(255.0)[None, None]
    x = (x.repeat(1, 3, 1, 1) - IM_MEAN) / IM_STD if b["rgb"] else (x - b["mean"]) / b["std"]
    return g, x


def target_layer(model, arch):
    return {"custom_cnn": lambda: model.features, "resnet18": lambda: model.layer4[-1], "mobilenet_v2": lambda: model.features[-1],
            "vgg16": lambda: model.features[29], "densenet121": lambda: model.features.denseblock4}[arch]()


def gradcam(b, x, class_idx):
    """Grad-CAM for one image: where the evidence for `class_idx` sits (0-1 map at the model's input size)."""
    store = {}
    layer = target_layer(b["model"], b["arch"])
    def fwd(_m, _i, out):
        if out.requires_grad:
            store["act"] = out
            out.register_hook(lambda g: store.__setitem__("grad", g))
    handle = layer.register_forward_hook(fwd)
    try:
        with torch.enable_grad():
            b["model"].zero_grad(set_to_none=True)
            out = b["model"](x)
            out[0, class_idx].backward()
    finally:
        handle.remove()
    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def analyze(pil, b, want_cam=True):
    g, x = preprocess(pil, b)
    with torch.no_grad():
        probs = b["model"](x).softmax(1)[0].numpy()
    k = int(probs.argmax())
    return {"gray": g, "probs": probs, "pred": k, "conf": float(probs[k]), "cam": gradcam(b, x, k) if want_cam else None}


def overlay_fig(gray, cam, size=300):
    base = np.asarray(gray.resize((size, size), Image.BILINEAR))
    heat = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)) / 255.0
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(base, cmap="gray", vmin=0, vmax=255); ax.imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1); ax.axis("off")
    fig.tight_layout(pad=0.1)
    return fig


def prob_fig(classes, probs, pred):
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    ax.barh(classes[::-1], probs[::-1], color=[COLORS.get(c, "gray") if i == pred else "#b9bfc7" for i, c in list(enumerate(classes))[::-1]])
    for i, p in enumerate(probs[::-1]):
        ax.text(min(p + 0.02, 0.86), i, f"{p:.1%}", va="center", fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("Probability"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def confidence_band(c):
    if c >= 0.90: return "high", "The model is confident, but it is mildly overconfident on held-out data (Step 8), so this is not a guarantee."
    if c >= 0.70: return "moderate", "Treat this prediction with care; wrong predictions were common in this range during testing."
    return "low", "The model is unsure. Do not rely on this prediction. The image may also be unlike the OCT scans it was trained on."


def synthetic_demo_image(seed=3, w=512, h=400):
    """A drawn stand-in for an OCT-like scan (NOT a real scan), used only to demonstrate preprocessing."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    centre = h * 0.5 + 40 * np.exp(-((xx - w / 2) ** 2) / (2 * 60 ** 2)) - 25 * np.sin(xx / w * np.pi)
    img = rng.normal(25, 12, (h, w))
    for off, amp in [(-30, 110), (-12, 150), (0, 90), (14, 200), (32, 120)]:
        img += amp * np.exp(-((yy - centre - off) ** 2) / (2 * 4.5 ** 2))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


SCORE_COLS = {"Precision", "Recall", "F1", "Val macro-F1", "Test acc", "Test macro-F1", "Val acc"}


def table(df):
    df = df.copy()
    for c in df.columns:
        if c in SCORE_COLS and df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: f"{v:.3f}")                # keep 1.000 instead of 1
    st.dataframe(df, hide_index=True, width="stretch")

# --------------------------------------------------------------------------------------------- recorded results
# Every number below is copied from the attached notebooks' outputs (Steps 1-8). Where a notebook text cell and a later
# re-run disagree, both are shown and labelled.
TRAIN_COUNTS = {"NORMAL": 51140, "CNV": 37205, "DME": 11348, "DRUSEN": 8616}
STEP1_PROPS = pd.DataFrame({"Class": ["CNV", "DME", "DRUSEN", "NORMAL"], "Mean pixel": [46.53, 46.58, 51.68, 44.15], "Pixel std": [46.48, 49.70, 54.65, 49.18]})
STEP2_WIDTHS = pd.DataFrame({"Width (px)": [512, 768, 1536, 1024, 384], "Scans (of 1,200 sampled)": [759, 326, 111, 3, 1]})
STEP4_CLASS = pd.DataFrame({"Class": ["CNV", "DME", "DRUSEN", "NORMAL"], "Recall": [1.000, 0.516, 0.000, 0.872], "Precision": [0.484, 0.921, 0.000, 0.634], "F1": [0.653, 0.662, 0.000, 0.734]})
STEP4_SCREEN = pd.DataFrame([
    ("Activation", "Tanh", 0.393, 0.716), ("Activation", "ReLU", 0.389, 0.706), ("Activation", "LeakyReLU", 0.383, 0.697), ("Activation", "GELU", 0.369, 0.674),
    ("Pooling", "Average", 0.395, 0.717), ("Pooling", "Max", 0.389, 0.706),
    ("Loss", "Label smoothing", 0.389, 0.707), ("Loss", "Cross-entropy", 0.389, 0.706), ("Loss", "Focal", 0.385, 0.567), ("Loss", "Class-weighted CE", 0.350, 0.433),
    ("Batch size", "32", 0.389, 0.706), ("Batch size", "64", 0.374, 0.682), ("Batch size", "128", 0.360, 0.656),
    ("Learning rate", "0.01", 0.393, 0.714), ("Learning rate", "0.001", 0.389, 0.706), ("Learning rate", "0.0001", 0.335, 0.617)],
    columns=["Experiment", "Setting", "Val macro-F1", "Val acc"])
STEP4_CURVE = pd.DataFrame({"Epoch": range(1, 11), "Val macro-F1": [0.193, 0.290, 0.168, 0.422, 0.377, 0.298, 0.418, 0.481, 0.529, 0.515],
                            "Val acc": [0.487, 0.532, 0.368, 0.722, 0.634, 0.564, 0.696, 0.758, 0.783, 0.783]})
STEP5 = pd.DataFrame([
    ("MobileNetV2", "Partial fine-tuning", 1.69, 0.822, 0.888, 0.885), ("DenseNet121", "Feature extraction", 0.004, 0.800, 0.847, 0.845),
    ("VGG16", "Partial fine-tuning", 7.10, 0.822, 0.835, 0.834), ("MobileNetV2", "Feature extraction", 0.005, 0.727, 0.819, 0.818),
    ("ResNet18", "Partial fine-tuning", 8.40, 0.765, 0.814, 0.810), ("ResNet18", "Feature extraction", 0.002, 0.730, 0.774, 0.770),
    ("DenseNet121", "Partial fine-tuning", 2.16, 0.744, 0.767, 0.757), ("VGG16", "Feature extraction", 0.016, 0.658, 0.677, 0.677)],
    columns=["Model", "Strategy", "Trainable params (M)", "Val macro-F1", "Test acc", "Test macro-F1"])
STEP5_SIZE = pd.DataFrame([("MobileNetV2", 2.229, 8.9, 0.205), ("ResNet18", 11.179, 44.7, 0.439), ("DenseNet121", 6.958, 27.8, 0.750), ("VGG16", 134.277, 537.1, 1.648)],
                          columns=["Model", "Parameters (M)", "Size (MB)", "Inference (ms / scan, Step 5 timing)"])
STEP5_FINAL = pd.DataFrame({"Class": ["CNV", "DME", "DRUSEN", "NORMAL"], "Recall": [0.972, 1.000, 0.884, 0.992], "Precision": [0.927, 0.954, 0.995, 0.976], "F1": [0.949, 0.977, 0.936, 0.984]})

# Step 6 experiments (validation data, 112 px, T4 GPU). Columns: train acc is measured on a fixed training probe in evaluation mode.
S6_FIT = pd.DataFrame([("Underfit", 10, 10, 0.390, 0.319, 0.071, 0.300, "underfitting"), ("Good fit", 10, 9, 0.547, 0.586, -0.039, 0.561, "underfitting"),
                       ("Overfit", 25, 25, 0.982, 0.858, 0.123, 0.857, "overfitting")],
                      columns=["Setup", "Epochs", "Best epoch", "Train acc (probe)", "Val acc", "Gap", "Val macro-F1", "Notebook diagnosis"])
S6_L2 = pd.DataFrame([("L2 = 0 (baseline)", 0.982, 0.858, 0.123, "0.468 -> 0.468", 0.857, "overfitting"), ("L2 = 1e-5", 0.980, 0.836, 0.144, "0.535 -> 0.535", 0.831, "overfitting"),
                      ("L2 = 1e-4", 0.915, 0.744, 0.171, "0.446 -> 0.665", 0.742, "overfitting"), ("L2 = 1e-3", 0.975, 0.861, 0.114, "0.409 -> 0.409", 0.861, "overfitting"),
                      ("L2 = 1e-2", 0.752, 0.711, 0.041, "0.437 -> 0.741", 0.702, "underfitting")],
                     columns=["Setup", "Train acc (probe)", "Val acc", "Gap", "Val loss (min -> final)", "Val macro-F1", "Notebook diagnosis"])
S6_DROP = pd.DataFrame([("Dropout 0.0", 0.770, 0.700, 0.070, 0.680, "underfitting"), ("Dropout 0.2", 0.782, 0.722, 0.059, 0.709, "underfitting"),
                        ("Dropout 0.4", 0.748, 0.678, 0.071, 0.648, "underfitting"), ("Dropout 0.6", 0.807, 0.781, 0.026, 0.777, "good fit")],
                       columns=["Setup", "Train acc (probe)", "Val acc", "Gap", "Val macro-F1", "Notebook diagnosis"])
S6_BN = pd.DataFrame([("With BN, lr 0.001", 0.748, 0.678, 0.648), ("With BN, lr 0.01", 0.490, 0.464, 0.393), ("No BN, lr 0.001", 0.783, 0.700, 0.695), ("No BN, lr 0.01", 0.250, 0.250, 0.100)],
                     columns=["Setup", "Train acc (probe)", "Val acc", "Val macro-F1"])
S6_ES = pd.DataFrame([("No early stopping (last epoch)", 25, 25, 0.858, 0.857), ("Early stop on val loss", 14, 9, 0.533, 0.794), ("Early stop on val macro-F1", 14, 9, 0.581, 0.786)],
                     columns=["Setup", "Epochs run", "Epoch kept", "Val acc (final epoch)", "Val macro-F1 (kept weights)"])
S6_AUG = pd.DataFrame([("Without augmentation", 0.748, 0.678, 0.071, 0.648), ("With augmentation", 0.713, 0.692, 0.022, 0.639)], columns=["Setup", "Train acc (probe)", "Val acc", "Gap", "Val macro-F1"])
S6_SEARCH = pd.DataFrame([("Trial 1", 5e-4, 16, 16, 0.758, 0.754), ("Trial 2", 5e-4, 16, 32, 0.736, 0.717), ("Trial 3", 5e-4, 32, 16, 0.658, 0.660), ("Trial 4", 5e-4, 32, 32, 0.569, 0.520),
                          ("Trial 5", 1e-3, 16, 16, 0.744, 0.738), ("Trial 6", 1e-3, 16, 32, 0.722, 0.674), ("Trial 7", 1e-3, 32, 16, 0.647, 0.636), ("Trial 8", 1e-3, 32, 32, 0.678, 0.648)],
                         columns=["Trial", "Learning rate", "Batch size", "Width", "Val acc", "Val macro-F1"])
S6_PT = pd.DataFrame([("MobileNetV2", "Trial 2", 4.4e-4, 3e-3, 0.0, 1e-3, 0.1, 0.866), ("MobileNetV2", "Default", 1e-4, 1e-3, 0.0, 0.0, 0.0, 0.859), ("MobileNetV2", "Trial 4", 1.1e-4, 1e-3, 0.5, 1e-5, 0.0, 0.854),
                      ("MobileNetV2", "Trial 3", 1.9e-5, 1e-3, 0.3, 1e-5, 0.1, 0.786), ("MobileNetV2", "Trial 1", 1.6e-5, 3e-3, 0.0, 1e-4, 0.0, 0.776),
                      ("ResNet18", "Trial 2", 4.4e-4, 3e-3, 0.0, 1e-3, 0.1, 0.890), ("ResNet18", "Trial 4", 1.1e-4, 1e-3, 0.5, 1e-5, 0.0, 0.885), ("ResNet18", "Default", 1e-4, 1e-3, 0.0, 0.0, 0.0, 0.870),
                      ("ResNet18", "Trial 3", 1.9e-5, 1e-3, 0.3, 1e-5, 0.1, 0.812), ("ResNet18", "Trial 1", 1.6e-5, 3e-3, 0.0, 1e-4, 0.0, 0.811)],
                     columns=["Model", "Trial", "Body lr", "Head lr", "Dropout", "L2", "Label smoothing", "Val macro-F1"])
# Final 3-seed comparison: the table the notebook's text cells record (the run that produced the saved ResNet18 checkpoint) ...
S6_FINAL = pd.DataFrame([("ResNet18, tuned", 0.894, 0.006, 0.894), ("ResNet18, default", 0.885, 0.009, 0.887), ("MobileNetV2, tuned", 0.883, 0.011, 0.883),
                         ("MobileNetV2, default", 0.870, 0.002, 0.870), ("Custom CNN, tuned", 0.780, 0.016, 0.782), ("Custom CNN, default", 0.612, 0.152, 0.641)],
                        columns=["Configuration", "Val macro-F1", "Std", "Val acc"])
# ... and a later re-run printed in the same notebook (different tuned settings were selected, so the ranking changed).
S6_FINAL_RERUN = pd.DataFrame([("ResNet18, default", 0.895, 0.017, 0.895), ("MobileNetV2, tuned", 0.885, 0.013, 0.886), ("ResNet18, tuned", 0.880, 0.004, 0.881),
                               ("MobileNetV2, default", 0.869, 0.010, 0.870), ("Custom CNN, tuned", 0.785, 0.010, 0.785), ("Custom CNN, default", 0.767, 0.078, 0.772)],
                              columns=["Configuration", "Val macro-F1", "Std", "Val acc"])
STEP6_RANK = S6_FINAL[["Configuration", "Val macro-F1"]]

# Step 6 / Step 7 test results for the ResNet18 model (rows = true class, columns = predicted: CNV, DME, DRUSEN, NORMAL)
CM_STEP6_RECORDED = np.array([[249, 1, 0, 0], [0, 250, 0, 0], [36, 1, 196, 17], [0, 0, 5, 245]])    # 94.0% accuracy, macro-F1 0.938 (Step 6 text cell)
CM_STEP7 = np.array([[241, 9, 0, 0], [0, 246, 0, 4], [62, 1, 151, 36], [0, 0, 0, 250]])              # 88.8% accuracy, macro-F1 0.881 (Step 7: saved checkpoint)
STEP7_REPORT = pd.DataFrame({"Class": CLASSES + ["Macro average"], "Precision": [0.795, 0.961, 1.000, 0.862, 0.905], "Recall": [0.964, 0.984, 0.604, 1.000, 0.888],
                             "F1": [0.872, 0.972, 0.753, 0.926, 0.881], "Scans": [250, 250, 250, 250, 1000]})
STEP7_COMPARE = pd.DataFrame([("Step 6, results recorded in the notebook text", 0.940, 0.938, 0.784), ("Step 6, later re-run (seed 42, trial-2 settings)", 0.947, 0.946, 0.832),
                              ("Step 7, saved checkpoint re-evaluated", 0.888, 0.881, 0.604)], columns=["Source", "Test acc", "Test macro-F1", "DRUSEN recall"])
CM_STEP8 = np.array([[224, 25, 1, 0], [0, 250, 0, 0], [52, 11, 177, 10], [1, 16, 36, 197]])

# Model efficiency. Size = FP32 weights. Speed = T4 GPU, batch of 32 after warm-up (Step 6 notebook).
EFF = pd.DataFrame([("Custom CNN (default)", 0.114, 0.458, 0.087, 0.612), ("Custom CNN (tuned)", 0.522, 2.088, 0.181, 0.780),
                    ("MobileNetV2 (tuned)", 2.229, 8.916, 0.227, 0.883), ("ResNet18 (tuned)", 11.179, 44.714, 0.270, 0.894)],
                   columns=["Model", "Parameters (M)", "Size (MB)", "Inference (ms / scan, T4)", "Val macro-F1 (3 seeds)"])


def cm_fig(cm, title, size=(4.6, 3.9)):
    n = cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(cm / n, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / n[i, 0]:.0%})", ha="center", va="center", fontsize=8, color="white" if cm[i, j] / n[i, 0] > 0.55 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def hbar(labels, values, xlabel, colors=None, xmax=1.0, err=None, figsize=(7.5, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(labels, values, xerr=err, color=colors, capsize=3); ax.set_xlim(0, xmax); ax.set_xlabel(xlabel)
    for i, v in enumerate(values): ax.text(min(v + 0.01, xmax * 0.93), i, f"{v:.3f}", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    return fig
# --------------------------------------------------------------------------------------------- pages
def page_home():
    st.title("Retinal Disease Detection from OCT Images")
    banner()
    st.write("An end-to-end deep-learning project that classifies retinal OCT scans into **CNV, DME, DRUSEN or NORMAL**. "
             "Use **Try the model** to upload a scan; the other pages summarise each project step with the results from the project notebooks.")
    steps = pd.DataFrame([
        ("1", "Dataset", "Kermany2017 OCT: 108,309 training and 1,000 test scans, 4 classes", "Training split imbalanced; test split balanced (250 per class)"),
        ("2", "Preprocessing", "Grey-level, resized, normalised; augmentation on training data only", "Widths vary (384-1536 px), heights about 496 px"),
        ("3", "Neural network fundamentals", "NumPy perceptron and multi-layer network on 32 x 32 pixels", "51.3% test accuracy: pixels alone are not enough"),
        ("4", "Custom CNN", "From-scratch PyTorch CNN, 114,420 parameters", "59.7% test accuracy; never predicted DRUSEN"),
        ("5", "Transfer learning", "ResNet18, VGG16, MobileNetV2, DenseNet121 with ImageNet weights", "VGG16 fully fine-tuned: 96.2% test accuracy"),
        ("6", "Optimization", "Dropout, L2, Batch Norm, early stopping, random search, 3 seeds", "Tuned ResNet18 best (validation macro-F1 0.894)"),
        ("7", "Evaluation and efficiency", "Held-out test evaluation and size/speed comparison", "Saved ResNet18 re-evaluated: 88.8% (Step 6 recorded 94.0%)"),
        ("8", "Explainability", "Grad-CAM heatmaps and sanity checks", "Heatmaps centre-biased; weak link to the decision"),
        ("9", "This application", "Upload, predict, visualise, compare models", "Educational prototype")],
        columns=["Step", "Topic", "What was done", "Headline"])
    table(steps)
    st.subheader("Test accuracy across the project")
    fig, ax = plt.subplots(figsize=(8, 3))
    labels = ["Step 3\nNumPy network", "Step 4\ncustom CNN", "Step 5\nVGG16 full fine-tune", "Step 6\nResNet18 (recorded)", "Step 7\nsaved ResNet18"]
    vals = [51.3, 59.7, 96.2, 94.0, 88.8]
    ax.bar(labels, vals, color=["#8a8f98", "#c8553d", "#3b6ea5", "#3f7d6b", "#7a9e3f"]); ax.set_ylim(0, 108); ax.set_ylabel("Test accuracy (%)")
    for i, v in enumerate(vals): ax.text(i, v + 2, f"{v}%", ha="center", fontsize=9)
    ax.tick_params(axis="x", labelsize=8); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); st.pyplot(fig)
    st.caption("These runs used different settings (image size, training data, tuning, seeds), so they are not an exact like-for-like comparison. "
               "The Step 6 and Step 7 bars are the same model family: Step 7 re-evaluated the saved checkpoint and got a lower figure than the Step 6 run (see Step 7).")


def render_result(f, pil, bundles, names, show_cam):
    """One uploaded image: primary model in detail, optional comparison with the other selected models."""
    primary = bundles[names[0]]
    res = {n: analyze(pil, bundles[n], show_cam and n == names[0]) for n in names}
    r = res[names[0]]
    cls = primary["classes"][r["pred"]]
    band, advice = confidence_band(r["conf"])
    with st.expander(f"{f.name}: {cls} ({r['conf']:.1%})", expanded=True):
        c1, c2, c3 = st.columns([1, 1, 1.15])
        with c1:
            st.markdown("**Uploaded image**")
            st.image(ImageOps.exif_transpose(pil).convert("L"), width="stretch")
            st.caption(f"Original size {pil.size[0]} x {pil.size[1]} px")
        with c2:
            st.markdown("**Grad-CAM overlay**" if show_cam else "**What the model sees**")
            if show_cam:
                st.pyplot(overlay_fig(r["gray"], r["cam"])); st.caption("Red = strongest evidence for the predicted class.")
            else:
                st.image(r["gray"].resize((300, 300)), width="stretch"); st.caption(f"Grey-level, resized to {primary['size']} x {primary['size']} px")
        with c3:
            full, desc = INFO.get(cls, (cls, ""))
            st.markdown(f'<div class="pred"><div class="small">{names[0]}</div><h2 style="color:{COLORS.get(cls, "#444")}">{cls}</h2>'
                        f'<div>{full}</div><div class="small">{desc}</div><hr style="margin:.6rem 0">'
                        f'<b>Confidence: {r["conf"]:.1%}</b> ({band})</div>', unsafe_allow_html=True)
            st.pyplot(prob_fig(primary["classes"], r["probs"], r["pred"]))
        (st.success if band == "high" else st.warning if band == "moderate" else st.error)(f"Confidence is {band}. {advice}")
        if len(names) > 1:
            rows = []
            for n in names:
                b, x = bundles[n], res[n]
                rows.append({"Model": n, "Predicted": b["classes"][x["pred"]], "Confidence": f"{x['conf']:.1%}", **{f"P({c})": f"{p:.1%}" for c, p in zip(b["classes"], x["probs"])}})
            st.markdown("**Model comparison for this image**")
            table(pd.DataFrame(rows))
            if len({row["Predicted"] for row in rows}) > 1:
                st.info("The models disagree. The Step 4 custom CNN was never able to predict DRUSEN (recall 0.000), so its answers should be weighed accordingly.")
    row = {"File": f.name, "Model": names[0], "Predicted class": cls, "Confidence": f"{r['conf']:.1%}", "Confidence level": band,
           **{f"P({c})": f"{p:.1%}" for c, p in zip(primary["classes"], r["probs"])}}
    for n in names[1:]:
        row[f"{n} prediction"] = f"{bundles[n]['classes'][res[n]['pred']]} ({res[n]['conf']:.1%})"
    return row


def page_predict():
    st.title("Try the model: upload an OCT image")
    banner()
    bundles = get_bundles()
    if not bundles:
        st.error("No model file found. Put `step6_best_model.pt` (and/or `step4_custom_cnn.pt`) next to `app.py`, or upload one in the sidebar under **Model files**.")
        return
    default = [n for n in bundles if n.startswith("Step 6")][:1] or list(bundles)[:1]
    names = st.multiselect("Model(s) to use (the first one is shown in detail; add more to compare)", list(bundles), default=default)
    if not names:
        st.info("Select at least one model."); return
    for n in names:
        b = bundles[n]
        st.sidebar.success(f"{n}\n\nInput {b['size']} x {b['size']} px, {b['params'] / 1e6:.2f} M parameters")
        if b["cfg"].get("inferred"):
            st.sidebar.warning(f"{n}: no config found, so the architecture and input size were inferred (assumed 112 px).")
    ack = st.checkbox("I understand this is an educational prototype and not a clinical diagnostic tool.", key="ack")
    if not ack:
        st.info("Tick the box above to enable uploads."); return
    files = st.file_uploader("Upload one or more retinal OCT images (PNG, JPG, TIFF, BMP)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], accept_multiple_files=True)
    show_cam = st.toggle("Show Grad-CAM heatmap for the first model (where it looked)", value=True)
    st.caption("The models were trained only on OCT B-scans (cross-sections of the retina). Other kinds of images give meaningless results.")
    if not files:
        return
    if len(files) > 10:
        st.warning("Only the first 10 images are analysed."); files = files[:10]
    rows = []
    for f in files:
        try:
            pil = Image.open(f); pil.load()
        except Exception:
            st.error(f"{f.name}: could not be read as an image."); continue
        rows.append(render_result(f, pil, bundles, names, show_cam))
    if rows:
        df = pd.DataFrame(rows)
        if len(rows) > 1:
            st.subheader("Summary"); table(df)
        st.download_button("Download results (CSV)", df.to_csv(index=False).encode(), "oct_predictions.csv", "text/csv")
    banner()


def page_step1():
    st.title("Step 1: Dataset"); banner()
    st.write("**Kermany2017 OCT** (Hugging Face: `zacharielegault/Kermany2017-OCT`): 108,309 training scans and 1,000 test scans in four classes. "
             "The training split is strongly imbalanced; the test split is balanced (250 per class).")
    df = pd.DataFrame({"Class": list(TRAIN_COUNTS), "Meaning": [INFO[c][0] for c in TRAIN_COUNTS], "Training scans": list(TRAIN_COUNTS.values()), "Test scans": 250})
    df["Share of training"] = (df["Training scans"] / df["Training scans"].sum()).map("{:.1%}".format)
    table(df)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(df["Class"], df["Training scans"], color=[COLORS[c] for c in df["Class"]]); ax.set_ylabel("Training scans")
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); st.pyplot(fig)
    st.subheader("Image properties (300 sampled scans)")
    st.markdown("All 300 were **JPEG, grey-level (mode L)**. Heights are almost always 496-512 px; widths range from 512 to 1,536 px (mean 713, so images must be resized to one size).")
    table(STEP1_PROPS)
    st.info("Why it matters: trained on a random sample of this imbalanced data, the Step 4 model never predicted DRUSEN. From Step 5 on, the training pool was balanced by class. "
            "The Step 1 notebook itself only loaded a small 480-scan subset (120 per class) for inspection.")
    st.caption("The Hugging Face copy has no patient IDs, so a random validation split can share patients with training. The official test split is patient-independent.")


def page_step2():
    st.title("Step 2: Image exploration and preprocessing"); banner()
    st.write("In a sample of **1,200 training scans** all images were grey-level; mean brightness 48.5, contrast 51.3, sharpness varied widely (median Laplacian variance 770, max 26,397), "
             "and on average 18% of pixels were near-black.")
    c1, c2 = st.columns([1, 1.3])
    with c1: table(STEP2_WIDTHS)
    with c2:
        table(pd.DataFrame([("Resize", "224 x 224 bilinear in Step 2 (112 x 112 for the models in Steps 5-9, 64 x 64 for Step 4, for speed)"),
                            ("Channels", "3 (grey repeated) for pretrained models; 1 for the custom CNN"),
                            ("Normalization", "ImageNet mean/std for pretrained models; the dataset's own (mean 0.191, std 0.217 at 224 px) for the custom CNN"),
                            ("Augmentation (train only)", "Horizontal flip, rotation up to 10 deg, 5% shift, 0.9-1.1 zoom, brightness/contrast +/-20%"),
                            ("Not used", "Vertical flip: retinal layers have a fixed top-to-bottom order")], columns=["Decision", "Setting"]))
    st.subheader("Try it")
    up = st.file_uploader("Upload an image to preview preprocessing (or use the drawn demo image)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], key="s2up")
    pil = Image.open(up) if up else synthetic_demo_image()
    if not up:
        st.caption("Showing a drawn stand-in image, not a real scan.")
    size = st.slider("Resize to (px)", 64, 224, 112, 8)
    c1, c2, c3, c4 = st.columns(4)
    bright = c1.slider("Brightness", 0.8, 1.2, 1.0, 0.05); contrast = c2.slider("Contrast", 0.8, 1.2, 1.0, 0.05)
    angle = c3.slider("Rotation (deg)", -10, 10, 0); flip = c4.checkbox("Horizontal flip")
    g = ImageOps.exif_transpose(pil).convert("L")
    small = g.resize((size, size), Image.BILINEAR)
    aug = ImageEnhance.Contrast(ImageEnhance.Brightness(small).enhance(bright)).enhance(contrast).rotate(angle, resample=Image.BILINEAR)
    aug = ImageOps.mirror(aug) if flip else aug
    a, b_, c_ = st.columns(3)
    a.image(g, caption=f"Original grey-level {g.size[0]} x {g.size[1]}", width="stretch")
    b_.image(small.resize((300, 300)), caption=f"Resized to {size} x {size}", width="stretch")
    c_.image(aug.resize((300, 300)), caption="With augmentation (training only)", width="stretch")


@st.cache_data
def train_spiral(act, lr, epochs, seed):
    """Three interleaved spirals (the Step 3 'Experiment A' problem) learned by a 2-32-32-3 NumPy network with momentum SGD."""
    rng = np.random.default_rng(seed)
    n, K = 100, 3
    X = np.zeros((n * K, 2)); y = np.zeros(n * K, dtype=int)
    for k in range(K):
        r = np.linspace(0.05, 1, n); t = np.linspace(k * 4, (k + 1) * 4, n) + rng.normal(0, 0.2, n)
        X[k * n:(k + 1) * n] = np.c_[r * np.sin(t), r * np.cos(t)]; y[k * n:(k + 1) * n] = k
    f, df = {"ReLU": (lambda z: np.maximum(0, z), lambda z: (z > 0).astype(float)), "Tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
             "Sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda z: (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z))))}[act]
    r_ = np.random.default_rng(seed + 1); sizes = [2, 32, 32, 3]
    W = [r_.normal(0, np.sqrt((2 if act == "ReLU" else 1) / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]; B = [np.zeros((1, b)) for b in sizes[1:]]
    VW, VB = [np.zeros_like(w) for w in W], [np.zeros_like(b) for b in B]
    Y = np.eye(K)[y]; losses = []

    def forward(Xin):
        Z, A = [], [Xin]
        for i in range(3):
            z = A[-1] @ W[i] + B[i]; Z.append(z)
            A.append(f(z) if i < 2 else np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True))
        return Z, A
    for _ in range(epochs):
        Z, A = forward(X); P = A[-1]
        losses.append(float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean()))
        dZ = (P - Y) / len(y)
        for i in (2, 1, 0):
            dW, dB = A[i].T @ dZ, dZ.sum(0, keepdims=True)
            if i > 0: dZ = (dZ @ W[i].T) * df(Z[i - 1])
            VW[i] = 0.9 * VW[i] - lr * dW; VB[i] = 0.9 * VB[i] - lr * dB; W[i] += VW[i]; B[i] += VB[i]
    gx, gy = np.meshgrid(np.linspace(-1.1, 1.1, 90), np.linspace(-1.1, 1.1, 90))
    grid = forward(np.c_[gx.ravel(), gy.ravel()])[1][-1].argmax(1).reshape(gx.shape)
    return losses, float((forward(X)[1][-1].argmax(1) == y).mean()), X, y, gx, gy, grid


def page_step3():
    st.title("Step 3: Neural network fundamentals (NumPy)"); banner()
    st.write("A perceptron and a multi-layer network written with NumPy only (forward propagation, softmax cross-entropy, backpropagation), tried first on toy problems and then on "
             "OCT scans shrunk to **32 x 32 = 1,024 pixels** (1,920 training, 480 validation, 1,000 test scans).")
    t1, t2, t3 = st.tabs(["Perceptron and backprop", "Activations and optimizers", "Live demo: spirals"])
    with t1:
        table(pd.DataFrame([("AND", 4, "yes", 1.0), ("OR", 4, "yes", 1.0), ("XOR", 50, "no", 0.5)], columns=["Logic gate", "Epochs", "Converged", "Accuracy"]))
        st.markdown("One neuron separates AND and OR but **never solves XOR**, which is why hidden layers are needed. On real OCT data (NORMAL vs CNV, 1,024 pixels) the perceptron reached "
                    "69.7% training and 63.7% validation accuracy: raw pixels are not linearly separable.")
        table(pd.DataFrame([("ReLU", 1.82e-7), ("Sigmoid", 1.03e-7), ("Tanh", 7.68e-9)], columns=["Activation", "Max relative gradient error"]))
        st.caption("Gradient check: analytic backpropagation matches a numerical estimate for all three activations (error about 1e-7 or smaller), so the implementation is correct.")
    with t2:
        st.markdown("**Experiment A: three interleaved spirals (toy data)**")
        table(pd.DataFrame([("ReLU", 0.0045, 0.0010, 0.998, 1.0), ("Sigmoid", 0.0397, 0.0225, 0.998, 1.0), ("Tanh", 0.0011, 0.0005, 1.0, 1.0)],
                           columns=["Activation", "Final train loss", "Final val loss", "Train acc", "Val acc"]))
        st.markdown("**Experiment C: the same recipe on OCT pixels (139,716 parameters, 25 epochs, Adam)**")
        table(pd.DataFrame([("ReLU", 0.966, 0.479), ("Sigmoid", 0.782, 0.421), ("Tanh", 0.981, 0.481)], columns=["Activation", "Train acc", "Val acc"]))
        st.caption("The networks memorize the 1,920 training scans (train accuracy near 97-98%) but reach only about 48% on validation: strong overfitting. Random guessing is 25%.")
        st.markdown("**Optimizers, learning rate, batch size and regularization (best validation accuracy)**")
        c1, c2 = st.columns(2)
        with c1:
            table(pd.DataFrame([("SGD (lr 0.01)", 0.4375), ("SGD + momentum", 0.4667), ("RMSprop", 0.4854), ("Adam", 0.4917)], columns=["Optimizer", "Best val acc"]))
            table(pd.DataFrame([("16", 120, 0.4667), ("64", 30, 0.4917), ("256", 8, 0.4646), ("1,920 (full batch)", 1, 0.4396)], columns=["Batch size", "Updates / epoch", "Best val acc"]))
        with c2:
            table(pd.DataFrame([("Adam, no tricks", 60, 8, 0.4688), ("+ L2 0.01", 60, 8, 0.4604), ("+ L2 + lr decay", 60, 13, 0.4979), ("+ L2 + decay + early stopping", 19, 13, 0.4688)],
                               columns=["Setup", "Epochs run", "Best epoch", "Val acc"]))
        st.markdown("**Final network: 51.3% test accuracy** (validation 46.9%; random guess 25%).")
        table(pd.DataFrame({"Class": CLASSES, "Recall": [0.420, 0.672, 0.320, 0.640], "Precision": [0.553, 0.525, 0.415, 0.539]}))
        st.info("A fully connected network ignores pixel neighbourhoods, which is exactly what the convolutional layers of Step 4 fix.")
    with t3:
        st.write("A 2-32-32-3 NumPy network learns three interleaved spirals with momentum SGD. Compare how each activation bends the decision boundary. "
                 "This re-implements the notebook's toy experiment live, so exact numbers differ.")
        acts = st.multiselect("Activation(s)", ["ReLU", "Tanh", "Sigmoid"], default=["ReLU", "Tanh", "Sigmoid"])
        c1, c2, c3 = st.columns(3)
        lr = c1.select_slider("Learning rate", [0.03, 0.1, 0.3, 1.0], value=0.3); epochs = c2.slider("Epochs", 100, 2000, 800, 100); seed = c3.number_input("Seed", 0, 999, 42)
        if acts:
            fig, axes = plt.subplots(1, len(acts) + 1, figsize=(4 * (len(acts) + 1), 3.6)); rows = []
            for ax, a in zip(axes[1:], acts):
                losses, acc, X, y, gx, gy, grid = train_spiral(a, lr, epochs, int(seed))
                axes[0].plot(losses, label=a); rows.append((a, losses[-1], acc))
                ax.contourf(gx, gy, grid, alpha=0.25, cmap="viridis"); ax.scatter(X[:, 0], X[:, 1], c=y, s=8, cmap="viridis"); ax.set_title(f"{a}: accuracy {acc:.0%}", fontsize=9); ax.axis("off")
            axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss"); axes[0].legend(); axes[0].spines[["top", "right"]].set_visible(False)
            fig.tight_layout(); st.pyplot(fig)
            table(pd.DataFrame(rows, columns=["Activation", "Final training loss", "Training accuracy"]).round(4))


def page_step4():
    st.title("Step 4: Custom CNN (PyTorch)"); banner()
    st.write("Four Conv -> BatchNorm -> Activation -> Pool blocks (16, 32, 64, 128 channels), global average pooling and a two-layer classifier with dropout: "
             "**114,420 parameters**, 64 x 64 grey-level input, trained on CPU on a random (imbalanced) sample of 3,402 scans (CNV 1,172, DME 357, DRUSEN 286, NORMAL 1,587).")
    st.subheader("Screening experiments (4-epoch runs, validation)")
    ex = st.selectbox("Experiment", STEP4_SCREEN["Experiment"].unique())
    d = STEP4_SCREEN[STEP4_SCREEN["Experiment"] == ex]
    fig, ax = plt.subplots(figsize=(6.5, 2.6)); x = np.arange(len(d))
    ax.bar(x - 0.2, d["Val macro-F1"], 0.4, label="Macro-F1"); ax.bar(x + 0.2, d["Val acc"], 0.4, label="Accuracy")
    ax.set_xticks(x); ax.set_xticklabels(d["Setting"]); ax.set_ylim(0, 1); ax.legend(); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); st.pyplot(fig)
    st.caption("Activations and pooling differ by less than about 0.03 macro-F1 (within noise for one seed); loss function, batch size and learning rate mattered more.")
    st.subheader("Final model (Tanh, average pooling, label smoothing, batch 32, lr 0.01, 10 epochs)")
    c1, c2, c3 = st.columns(3); c1.metric("Test accuracy", "59.7%"); c2.metric("Test macro-F1", "0.512"); c3.metric("DRUSEN recall", "0.000")
    fig, ax = plt.subplots(figsize=(6.5, 2.6)); ax.plot(STEP4_CURVE["Epoch"], STEP4_CURVE["Val macro-F1"], "o-", label="Val macro-F1"); ax.plot(STEP4_CURVE["Epoch"], STEP4_CURVE["Val acc"], "o-", label="Val accuracy")
    ax.set_xlabel("Epoch"); ax.legend(); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); st.pyplot(fig)
    table(STEP4_CLASS)
    st.warning("No test scan was ever predicted as DRUSEN: the class was rare in the training sample. Validation accuracy (78%) overstated test accuracy (60%) because the validation split was also imbalanced. "
               "This led to the balanced training pool used from Step 5 on.")
    st.caption("This model's weights (`step4_custom_cnn.pt`) can be loaded in **Try the model** to compare it with the ResNet18.")


def page_step5():
    st.title("Step 5: Transfer learning"); banner()
    st.write("Four ImageNet-pretrained models with the final layer replaced by a 4-class layer, compared with feature extraction (new layer only) and "
             "partial fine-tuning (new layer + last block). Balanced pool: 2,040 training and 360 validation scans, 112 x 112 px, one seed each.")
    d = STEP5.sort_values("Test macro-F1")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh([f"{m} ({s.split()[0].lower()})" for m, s in zip(d["Model"], d["Strategy"])], d["Test macro-F1"], color=["#3b6ea5" if "Partial" in s else "#9bb3d1" for s in d["Strategy"]])
    ax.axvline(0.512, color="#c8553d", ls="--"); ax.text(0.515, 0.2, "custom CNN (Step 4)", color="#c8553d", fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("Test macro-F1"); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); st.pyplot(fig)
    table(STEP5)
    st.subheader("Size and speed"); table(STEP5_SIZE)
    st.subheader("VGG16 fully fine-tuned (the Step 5 final model)")
    c1, c2, c3 = st.columns(3); c1.metric("Test accuracy", "96.2%"); c2.metric("Test macro-F1", "0.962"); c3.metric("Weakest class", "DRUSEN (recall 0.884)")
    table(STEP5_FINAL)
    st.caption("Caveat: only VGG16 received full fine-tuning (6 epochs), so the jump from 0.834 to 0.962 reflects more training, not proof that VGG16 is the best architecture. One seed, small validation set.")
def exp_view(df, metric, label_col, note, err_col=None):
    d = df.iloc[::-1]
    st.pyplot(hbar(d[label_col].astype(str), d[metric].values, metric, colors="#3b6ea5", err=(d[err_col].values if err_col else None), figsize=(7.5, 0.45 * len(d) + 1.4)))
    table(df)
    st.caption(note)


def page_step6():
    st.title("Step 6: Model optimization and regularization"); banner()
    st.write("Experiments on validation data at 112 x 112 px (T4 GPU): under/overfitting diagnosis, **dropout**, **L2**, **Batch Normalization**, **early stopping**, "
             "data augmentation and random hyperparameter search, then a final comparison of default vs tuned configurations over 3 seeds. The test split was used only at the end.")
    exp = st.selectbox("Experiment", ["6.3 Underfit, good fit, overfit", "6.4 L2 regularization", "6.4 Dropout", "6.5 Batch Normalization", "6.6 Early stopping",
                                      "Data augmentation", "6.7 Custom CNN random search", "6.8 Pretrained random search", "6.9 Final comparison (3 seeds)", "Selected model"])
    if exp.startswith("6.3"):
        exp_view(S6_FIT, "Val macro-F1", "Setup", "The overfit model reaches 98% accuracy on its training probe but 86% on validation (gap 0.12). The 'good fit' and 'underfit' labels come from a rule of thumb "
                 "(training-probe accuracy below 0.80 counts as underfitting), which is why the good-fit model is still flagged: it is simply too short a run.")
    elif exp == "6.4 L2 regularization":
        exp_view(S6_L2, "Val macro-F1", "Setup", "Mild L2 (1e-3) matched the baseline (0.861 vs 0.857) and narrowed the gap slightly; L2 = 1e-2 closes the gap (0.041) by underfitting, and 1e-4 was unstable (val loss 0.446 -> 0.665).")
    elif exp == "6.4 Dropout":
        exp_view(S6_DROP, "Val macro-F1", "Setup", "Dropout 0.6 gave the best validation macro-F1 (0.777) and the smallest gap (0.026) in these 10-epoch runs. Differences of a few points are within noise for one seed.")
    elif exp.startswith("6.5"):
        exp_view(S6_BN, "Val macro-F1", "Setup", "At lr 0.001 the network without Batch Norm was slightly better here (0.695 vs 0.648). The clear benefit of Batch Norm is stability at a higher rate: at lr 0.01 it still learned (0.393) "
                 "while the network without it failed completely (0.100, chance-level accuracy).")
    elif exp.startswith("6.6"):
        exp_view(S6_ES, "Val macro-F1 (kept weights)", "Setup", "In this run early stopping (patience 5) ended after 14 epochs and kept the epoch-9 weights, but scored lower (0.794) than training all 25 epochs (0.857): "
                 "validation loss was noisy, so a longer patience would be safer.")
    elif exp == "Data augmentation":
        exp_view(S6_AUG, "Val macro-F1", "Setup", "Augmentation narrowed the gap (0.071 to 0.022) but did not raise validation macro-F1 in this experiment.")
    elif exp.startswith("6.7"):
        exp_view(S6_SEARCH, "Val macro-F1", "Trial", "Eight short random-search trials (10 epochs) over learning rate, batch size and width. The configuration used in the final comparison is width 32, 512 hidden units, "
                 "dropout 0.4, lr 0.001, batch 32, no augmentation, as printed in the notebook.")
    elif exp.startswith("6.8"):
        d = S6_PT.copy(); d["Label"] = d["Model"] + " " + d["Trial"]
        exp_view(d.drop(columns=["Label"]), "Val macro-F1", "Trial", "For both architectures the best short trial used body lr 4.4e-4, head lr 3e-3, L2 1e-3 and label smoothing 0.1; 'default' is the Step 5 style setup.")
    elif exp.startswith("6.9"):
        st.markdown("**Final comparison recorded in the notebook's text cells** (the run that produced the saved ResNet18 checkpoint; mean +/- std over 3 seeds):")
        exp_view(S6_FINAL, "Val macro-F1", "Configuration", "Tuning raised the custom CNN from 61.2% to 78.0% and cut its spread from 15.2 to 1.6 points. Pretrained models changed by only +0.9 to +1.3 points.", err_col="Std")
        st.markdown("**A later re-run in the same notebook** selected different tuned settings and gave:")
        table(S6_FINAL_RERUN)
        st.warning("The ranking of tuned vs default ResNet18 flips between the two runs (0.894 vs 0.885 in the first, 0.880 vs 0.895 in the re-run). Differences of about one point between the pretrained "
                   "configurations are within seed noise; the safe conclusion is that pretrained models beat the custom CNN by about 10 points, not which pretrained setting is best.")
    else:
        table(pd.DataFrame([("Architecture", "ResNet18, all layers fine-tuned"), ("Body / head learning rate", "1.1e-4 / 1e-3"), ("Dropout before final layer", "0.5"), ("L2 (weight decay)", "1e-5"),
                            ("Label smoothing", "0"), ("Validation macro-F1 (3-seed mean)", "0.894"), ("Image size", "112 x 112"), ("Recorded independent test", "94.0% accuracy, 0.938 macro-F1")],
                           columns=["Setting", "Value (from the saved checkpoint's config)"]))
        st.caption("These are the settings stored inside `step6_best_model.pt`. They correspond to trial 4 of the pretrained search, not to the trial-2 settings the later re-run selected (see Step 7).")


def page_step7():
    st.title("Step 7: Model evaluation and efficiency"); banner()
    t1, t2 = st.tabs(["Test evaluation", "Model efficiency"])
    with t1:
        st.write("The saved Step 6 ResNet18 checkpoint evaluated on the 1,000 held-out test scans (250 per class).")
        c1, c2, c3 = st.columns(3); c1.metric("Accuracy", "88.8%"); c2.metric("Macro-F1", "0.881"); c3.metric("Errors", "112 of 1,000")
        left, right = st.columns([1, 1])
        with left: st.pyplot(cm_fig(CM_STEP7, "Step 7: saved checkpoint (counts and recall)"))
        with right: table(STEP7_REPORT)
        st.markdown("**Where the errors are:** 99 of the 112 errors are DRUSEN scans (62 called CNV, 36 NORMAL, 1 DME). When the model says DRUSEN it is always right (precision 1.000), but it misses 40% of true DRUSEN scans.")
        st.subheader("Why Step 6 and Step 7 disagree")
        table(STEP7_COMPARE)
        st.pyplot(cm_fig(CM_STEP6_RECORDED, "Step 6: recorded result (94.0%)"))
        st.warning("The checkpoint's stored settings (body lr 1.1e-4, dropout 0.5, L2 1e-5; trial 4) differ from the settings the later Step 6 re-run selected (body lr 4.4e-4, dropout 0, L2 1e-3, label smoothing 0.1; trial 2). "
                   "So the saved file most likely comes from a different training run than the 94% result. In addition, the Step 7 notebook did not find the cached test arrays and re-created them from the dataset. "
                   "This is the likely explanation but it has not been verified; the dashboard's live ResNet18 is the saved checkpoint, whose measured accuracy is the lower 88.8%.")
    with t2:
        st.write("Accuracy is not the only criterion. Size is the FP32 weight size; speed was timed on a T4 GPU with batches of 32 after warm-up (Step 6 notebook).")
        table(EFF)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.scatter(EFF["Size (MB)"], EFF["Val macro-F1 (3 seeds)"], s=90, color=["#c8553d", "#e0a030", "#3f7d6b", "#3b6ea5"])
        for _, r in EFF.iterrows(): ax.annotate(f"{r['Model']}\n{r['Inference (ms / scan, T4)']:.2f} ms/scan", (r["Size (MB)"], r["Val macro-F1 (3 seeds)"]), textcoords="offset points", xytext=(6, -22), fontsize=8)
        ax.set_xscale("log"); ax.set_xlabel("Model size (MB, log scale)"); ax.set_ylabel("Validation macro-F1 (3 seeds)"); ax.set_ylim(0.5, 0.97)
        ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); st.pyplot(fig)
        st.markdown("- **Custom CNN (default)** is the smallest and fastest (0.46 MB, 0.087 ms/scan) but its validation macro-F1 (0.612) is far lower and very unstable across seeds.\n"
                    "- **MobileNetV2 (tuned)** reaches 0.883 with about 20% of ResNet18's size (8.9 MB vs 44.7 MB), so it is the efficient option.\n"
                    "- **ResNet18 (tuned)** is the most accurate on validation (0.894) and still fast (0.27 ms/scan on a GPU), but the largest of the four.")
        st.caption("Notes: the Step 6 notebook's efficiency table lists the tuned custom CNN as 0.422 M parameters and 1.687 MB, which contradicts its own comparison table (0.522 M); the value above is recomputed from the architecture "
                   "(521,956 parameters, 2.09 MB). Step 5 timed models separately (VGG16 1.65 ms/scan and 537 MB; DenseNet121 0.75 ms/scan and 27.8 MB), so those timings are not directly comparable. "
                   "The Step 7 notebook's 4.1 ms/image for ResNet18 covers the whole evaluation loop including data loading, not the model alone.")
        st.subheader("Measure on this machine")
        bundles = get_bundles()
        if not bundles:
            st.info("Load a model (sidebar) to benchmark it here.")
        elif st.button("Run CPU benchmark"):
            rows = []
            with st.spinner("Timing ..."):
                for n, b in bundles.items():
                    t = benchmark(b)
                    rows.append({"Model": n, "Parameters (M)": round(b["params"] / 1e6, 3), "Size (MB, FP32)": round(b["params"] * 4 / 1e6, 2), "Input": f"{b['size']} x {b['size']}",
                                 "Latency, batch 1 (ms)": round(t[1], 2), "Per image, batch 32 (ms)": round(t[32], 2)})
            table(pd.DataFrame(rows))
            st.caption("Timed on this computer's CPU, so absolute numbers differ from the GPU figures above.")


def page_step8():
    st.title("Step 8: Explainability (Grad-CAM)"); banner()
    st.write("Grad-CAM weights the last convolutional feature maps by the gradient of the class score to show **where** the evidence for a class lies. "
             "It shows where the model looked, not why it decided.")
    st.subheader("Recorded results (stand-in ResNet18, not the Step 6 model)")
    st.caption("The Step 8 notebook was run with a quick 3-epoch ResNet18 (84.8% test accuracy) because the Step 6 files were unavailable. Repeat with the real checkpoint before drawing conclusions.")
    left, right = st.columns(2)
    with left:
        st.pyplot(cm_fig(CM_STEP8, "Stand-in model: test confusion matrix"))
    with right:
        table(pd.DataFrame([("Heatmap peak on retinal tissue", "44% (chance about 20%); NORMAL 65%"), ("Focus ratio (heat on tissue / tissue share)", "0.87-1.13 (about 1 = no preference)"),
                            ("Deletion test at 20% erased", "most highlighted 0.381, random 0.393, least highlighted 0.849"),
                            ("Randomization rank correlation", "0.34 mean (median 0.45)"), ("Confidence, correct vs wrong", "0.91 vs 0.71")], columns=["Check", "Result"]))
        st.markdown("**Finding:** the heatmaps are a single coarse blob over the scan centre for every class, so they do not show class-specific lesions, "
                    "and erasing the most highlighted pixels hurts no more than erasing random ones.")
    st.subheader("Grad-CAM on your own image")
    bundles = get_bundles()
    if not bundles:
        st.info("Load a model (sidebar) to compute heatmaps here."); return
    name = st.selectbox("Model", list(bundles))
    b = bundles[name]
    up = st.file_uploader("Upload an OCT image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], key="s8up")
    if up:
        pil = Image.open(up); res = analyze(pil, b, True); cls = b["classes"][res["pred"]]
        c1, c2, c3 = st.columns(3)
        c1.image(ImageOps.exif_transpose(pil).convert("L"), caption="Original", width="stretch")
        c2.pyplot(overlay_fig(res["gray"], res["cam"])); c3.pyplot(prob_fig(b["classes"], res["probs"], res["pred"]))
        st.write(f"Predicted **{cls}** ({res['conf']:.1%}). Heatmap = evidence for the predicted class.")
        other = st.selectbox("Show the evidence for another class", list(b["classes"]), index=res["pred"])
        _, x = preprocess(pil, b)
        st.pyplot(overlay_fig(res["gray"], gradcam(b, x, b["classes"].index(other))))


def page_about():
    st.title("About and limitations"); banner()
    st.markdown("""
**Intended use.** Education and demonstration of a deep-learning workflow on public data.

**Not for clinical use.** The models have not been validated on patients, devices or hospitals beyond one public dataset, and no regulatory review has been done.

**Known limitations**
- **DRUSEN is the weakest class.** It is often called CNV or NORMAL (Steps 5, 6, 7, 8); the Step 4 custom CNN never predicted it.
- **One dataset, one source.** Other OCT machines, image quality or cropping can break the models.
- **Small images.** Scans are shrunk to 112 x 112 px (64 x 64 for the Step 4 model) for speed; subtle features can be lost.
- **Balanced test set.** Real clinics see mostly NORMAL and CNV, so precision in practice would differ.
- **Overconfidence.** Reported confidence is somewhat higher than the true hit rate in the middle ranges.
- **Explanations are coarse.** Grad-CAM heatmaps did not show class-specific lesions (Step 8).
- **Unresolved result gap.** Step 6 recorded 94.0% test accuracy while the saved checkpoint measured 88.8% in Step 7; the app uses the saved checkpoint.
- **Non-OCT images** produce meaningless predictions; the app cannot tell what kind of image it was given.

Uploaded images are processed in memory for the current session only and are not saved by this app.
""")


PAGES = {"Home": page_home, "Try the model (Step 9)": page_predict, "Step 1: Dataset": page_step1, "Step 2: Preprocessing": page_step2,
         "Step 3: NN fundamentals": page_step3, "Step 4: Custom CNN": page_step4, "Step 5: Transfer learning": page_step5,
         "Step 6: Optimization": page_step6, "Step 7: Evaluation and efficiency": page_step7, "Step 8: Explainability": page_step8, "About and limitations": page_about}

st.sidebar.title("OCT Disease Detection")
page = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
st.sidebar.markdown(f'<div class="small">{DISCLAIMER}</div>', unsafe_allow_html=True)
PAGES[page]()
st.markdown("<hr><div class='small'>Educational prototype. Not a clinical diagnostic tool. Do not use for medical decisions.</div>", unsafe_allow_html=True)

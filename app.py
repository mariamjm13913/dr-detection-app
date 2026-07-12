import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import timm
import os
import gdown
from PIL import Image
from torchvision import transforms

st.set_page_config(page_title="DR Grading — Swin V2", page_icon="🩺", layout="wide")

device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
CONFIDENCE_THRESHOLD = 45

MODEL_PATH = 'phase2_best.pt'
DRIVE_FILE_ID = '1dw4dIKaRJkrNzG9EaFuSd_ihF7rv1rrE'

# Ordinal severity palette — green (healthy) to red (advanced disease)
SEVERITY_COLORS = {
    'No DR':            '#2F9E44',
    'Mild':              '#82C91E',
    'Moderate':          '#F5A623',
    'Severe':            '#E8590C',
    'Proliferative DR':  '#C92A2A',
}

# ----------------------------------------------------------------------------
# THEME — injected once, styles native Streamlit widgets + custom components
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --bg: #F6F8F9;
    --bg-alt: #ECF1F3;
    --ink: #10242E;
    --ink-soft: #55707A;
    --teal: #0E4F5C;
    --cyan: #17A6B0;
    --border: #D8E1E4;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: var(--ink); }
.stApp { background: var(--bg); }

/* Hide default chrome clutter */
#MainMenu, footer { visibility: hidden; }

/* Hero */
.hero-wrap {
    background: linear-gradient(135deg, var(--teal) 0%, #093943 100%);
    border-radius: 16px;
    padding: 2.1rem 2.4rem;
    margin-bottom: 1.6rem;
    color: #EAF4F4;
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.05rem;
    line-height: 1.15;
    margin: 0 0 0.5rem 0;
}
.hero-sub {
    font-size: 0.95rem;
    color: #C7DEDE;
    max-width: 640px;
    line-height: 1.5;
}

/* Card container (wraps st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
}

/* File uploader restyle */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-alt);
    border: 1.5px dashed #9FB6BC;
    border-radius: 12px;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--cyan); }

/* Section labels */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin: 0 0 0.6rem 0;
}

/* Result header */
.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    margin: 0 0 0.15rem 0;
}
.result-conf {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--ink-soft);
}

/* Severity badge */
.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #fff;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    margin-bottom: 0.5rem;
}

/* Confidence gauge (conic-gradient ring) */
.gauge {
    width: 108px; height: 108px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.gauge-inner {
    width: 84px; height: 84px;
    border-radius: 50%;
    background: #fff;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.gauge-num { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1.15rem; }
.gauge-lbl { font-size: 0.6rem; color: var(--ink-soft); letter-spacing: 0.06em; text-transform: uppercase; }

/* Severity spectrum — signature element */
.spectrum-wrap { margin-top: 0.3rem; }
.spectrum-track {
    display: flex;
    width: 100%;
    height: 10px;
    border-radius: 6px;
    overflow: hidden;
}
.spectrum-seg { flex: 1; }
.spectrum-labels {
    display: flex;
    width: 100%;
    margin-top: 6px;
}
.spectrum-labels span {
    flex: 1;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: var(--ink-soft);
    text-align: center;
}
.spectrum-labels span.active { color: var(--ink); font-weight: 600; }
.spectrum-pointer-row { display: flex; width: 100%; }
.spectrum-pointer-cell { flex: 1; display: flex; justify-content: center; }
.spectrum-pointer {
    width: 0; height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-bottom: 7px solid var(--ink);
    margin-bottom: 2px;
}

/* Probability bars */
.prob-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.55rem; }
.prob-name { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; width: 128px; flex-shrink: 0; color: var(--ink-soft); }
.prob-track { flex: 1; background: var(--bg-alt); border-radius: 5px; height: 12px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 5px; }
.prob-pct { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; width: 44px; text-align: right; flex-shrink: 0; }

/* Low-confidence notice */
.notice {
    background: #FFF4E5;
    border: 1px solid #F5C177;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #7A4A00;
    margin-top: 0.9rem;
}

/* About footer */
.about-card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.6rem;
}
.about-card h4 {
    font-family: 'Space Grotesk', sans-serif;
    margin-top: 0;
}
.about-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--ink-soft);
    line-height: 1.9;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights (first launch only)..."):
            url = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)

    model = timm.create_model('swinv2_tiny_window8_256', pretrained=False, num_classes=5, drop_rate=0.3)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

gradients, activations = None, None

def save_gradient(grad):
    global gradients
    gradients = grad

def forward_hook(module, input, output):
    global activations
    activations = output
    output.register_hook(save_gradient)

target_layer = model.layers[-1].blocks[-1].norm2
target_layer.register_forward_hook(forward_hook)

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def predict(image):
    img = image.convert('RGB')
    x = eval_transform(img).unsqueeze(0).to(device)
    x.requires_grad_()

    out = model(x)
    probs = F.softmax(out, dim=1).cpu().detach().numpy()[0]
    pred_class = int(probs.argmax())
    confidence = float(probs[pred_class] * 100)

    model.zero_grad()
    out[0, pred_class].backward()
    grads = gradients[0].detach().cpu()
    acts = activations[0].detach().cpu()
    weights = grads.mean(dim=0)
    cam = (weights * acts).sum(dim=-1)
    side = int(cam.shape[0] ** 0.5)
    cam = cam.reshape(side, side).numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    img_resized = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET) / 255.0
    heatmap = heatmap[..., ::-1]
    overlay = (0.5 * img_resized + 0.5 * heatmap)
    overlay_img = (overlay * 255).astype(np.uint8)

    return pred_class, confidence, probs, overlay_img


# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">Swin V2 Tiny · Fundus Image Classifier</div>
    <div class="hero-title">Diabetic Retinopathy Grading</div>
    <div class="hero-sub">Upload a retinal fundus photograph to estimate DR severity on the
    standard 5-stage scale, with a Grad-CAM overlay showing which regions of the retina
    drove the prediction.</div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# UPLOAD
# ----------------------------------------------------------------------------
upload_card = st.container(border=True)
with upload_card:
    st.markdown('<div class="section-label">01 · Upload fundus image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a fundus image",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed",
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    pred_class, confidence, probs, overlay_img = predict(image)
    pred_label = class_names[pred_class]
    pred_color = SEVERITY_COLORS[pred_label]

    st.write("")

    # --- Images ---
    img_card = st.container(border=True)
    with img_card:
        st.markdown('<div class="section-label">02 · Image & model focus</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.image(overlay_img, caption="Grad-CAM — Model Focus Area", use_container_width=True)

    st.write("")

    # --- Result ---
    result_card = st.container(border=True)
    with result_card:
        st.markdown('<div class="section-label">03 · Prediction</div>', unsafe_allow_html=True)

        gauge_pct = min(max(confidence, 0), 100)
        head_l, head_r = st.columns([3, 1])
        with head_l:
            st.markdown(f"""
            <span class="badge" style="background:{pred_color};">{pred_label.upper()}</span>
            <div class="result-title">{pred_label}</div>
            <div class="result-conf">Model confidence for this image</div>
            """, unsafe_allow_html=True)
        with head_r:
            st.markdown(f"""
            <div class="gauge" style="background: conic-gradient({pred_color} {gauge_pct*3.6}deg, #E4EBED 0deg);">
                <div class="gauge-inner">
                    <div class="gauge-num">{confidence:.0f}%</div>
                    <div class="gauge-lbl">confidence</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Signature element: ordinal severity spectrum with pointer
        seg_html = "".join(
            f'<div class="spectrum-seg" style="background:{SEVERITY_COLORS[c]};"></div>'
            for c in class_names
        )
        label_html = "".join(
            f'<span class="{"active" if c == pred_label else ""}">{c}</span>'
            for c in class_names
        )
        pointer_mark = '<div class="spectrum-pointer"></div>'
        pointer_html = "".join(
            f'<div class="spectrum-pointer-cell">{pointer_mark if c == pred_label else ""}</div>'
            for c in class_names
        )
        st.markdown(f"""
        <div class="spectrum-wrap">
            <div class="spectrum-pointer-row">{pointer_html}</div>
            <div class="spectrum-track">{seg_html}</div>
            <div class="spectrum-labels">{label_html}</div>
        </div>
        """, unsafe_allow_html=True)

        if confidence < CONFIDENCE_THRESHOLD:
            st.markdown(f"""
            <div class="notice">⚠️ Low confidence ({confidence:.1f}%) — this image may not be a
            valid retinal fundus photo, or quality may be too low for reliable grading.</div>
            """, unsafe_allow_html=True)

    st.write("")

    # --- Probabilities ---
    prob_card = st.container(border=True)
    with prob_card:
        st.markdown('<div class="section-label">04 · Probability by class</div>', unsafe_allow_html=True)
        for i, cname in enumerate(class_names):
            pct = float(probs[i]) * 100
            color = SEVERITY_COLORS[cname]
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-name">{cname}</div>
                <div class="prob-track"><div class="prob-fill" style="width:{pct:.1f}%; background:{color};"></div></div>
                <div class="prob-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("Upload a fundus image above to run a grading.")


# ----------------------------------------------------------------------------
# ABOUT
# ----------------------------------------------------------------------------
st.markdown("""
<div class="about-card">
    <h4>About this model</h4>
    <div class="about-meta">
        ARCHITECTURE &nbsp;·&nbsp; Swin Transformer V2 (Tiny)<br>
        TRAINING &nbsp;·&nbsp; Leak-free, class-balanced split of a diabetic retinopathy fundus image dataset<br>
        TEST SET &nbsp;·&nbsp; QWK 0.7972 &nbsp;|&nbsp; macro-F1 0.6369 &nbsp;|&nbsp; accuracy 0.72
    </div>
    <p style="margin-top:0.9rem; font-size:0.85rem; color:var(--ink-soft); line-height:1.6;">
        This is an academic prototype — not validated for clinical use. Prediction confidence
        reflects the model's certainty for this specific image, not a guarantee of correctness.
        See the full per-class precision/recall report for overall model reliability.
    </p>
</div>
""", unsafe_allow_html=True)

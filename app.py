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

# Ordinal severity palette — tuned to glow against a dark background
SEVERITY_COLORS = {
    'No DR':            '#3DDC84',
    'Mild':              '#A8E62F',
    'Moderate':          '#FFC93C',
    'Severe':            '#FF8A3D',
    'Proliferative DR':  '#FF3D6E',
}

# ----------------------------------------------------------------------------
# THEME — "Ophthalmoscope" : deep clinical navy, fundus-amber signature glow,
# stacked/shingled cards, slow aurora drift, pulsing scan rings.
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --amber: #ff2d78;
    --amber-soft: #ffa8cf;
    --teal: #a855f7;
    --rose: #4f7bff;
    --bg-deep: #1a0f2e;
    --panel: rgba(28, 16, 48, 0.62);
    --panel-border: rgba(255, 61, 145, 0.35);
    --ink: #F2E9FB;
    --ink-soft: #B9A9CE;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }

/* ---------------- Ambient living background ---------------- */
.stApp {
    background:
        radial-gradient(circle at 12% 20%, rgba(180,30,90,0.35) 0%, transparent 42%),
        radial-gradient(circle at 88% 15%, rgba(90,40,180,0.30) 0%, transparent 42%),
        radial-gradient(circle at 20% 88%, rgba(200,20,90,0.28) 0%, transparent 40%),
        radial-gradient(circle at 85% 82%, rgba(60,50,200,0.30) 0%, transparent 42%),
        linear-gradient(160deg, #1a0f2e 0%, #140f28 45%, #0c1830 100%);
    background-size: 180% 180%, 180% 180%, 180% 180%, 180% 180%, 100% 100%;
    animation: aurora-drift 26s ease-in-out infinite alternate;
    background-attachment: fixed;
}
@keyframes aurora-drift {
    0%   { background-position: 0% 0%, 100% 0%, 0% 100%, 100% 100%, 0 0; }
    100% { background-position: 20% 15%, 80% 20%, 15% 85%, 85% 80%, 0 0; }
}

.block-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    padding-top: 3rem;
    padding-bottom: 4rem;
}
.block-container > div { width: 100%; }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

/* ---------------- Centered text, everywhere ---------------- */
.stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown h1, .stMarkdown h2,
.stMarkdown h3, .stMarkdown h4, .stCaption, [data-testid="stImageCaption"],
[data-testid="stCaptionContainer"] {
    text-align: center !important;
}

/* ---------------- Hero ---------------- */
.hero-wrap { text-align: center; padding: 0.4rem 1rem 0.2rem 1rem; position: relative; }

.scan-rings {
    position: relative;
    width: 84px; height: 84px;
    margin: 0 auto 1.1rem auto;
    display: flex; align-items: center; justify-content: center;
}
.scan-rings::before, .scan-rings::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    border: 1.5px solid rgba(245, 162, 75, 0.55);
    width: 100%; height: 100%;
    animation: pulse-ring 3.2s cubic-bezier(0.4,0,0.3,1) infinite;
}
.scan-rings::after { animation-delay: 1.6s; border-color: rgba(63, 209, 192, 0.5); }
@keyframes pulse-ring {
    0%   { transform: scale(0.55); opacity: 0.9; }
    80%  { transform: scale(1.5); opacity: 0; }
    100% { transform: scale(1.5); opacity: 0; }
}
.hero-icon {
    width: 60px; height: 60px; border-radius: 50%;
    background: linear-gradient(135deg, var(--amber), var(--teal));
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 30px rgba(245, 162, 75, 0.55), inset 0 0 12px rgba(255,255,255,0.25);
    position: relative; z-index: 2;
}
.hero-title {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 2.15rem;
    letter-spacing: -0.01em;
    background: linear-gradient(100deg, var(--amber-soft), var(--teal), var(--amber-soft));
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: sheen 7s linear infinite;
    margin-bottom: 0.4rem;
}
@keyframes sheen { to { background-position: 300% center; } }
.hero-sub {
    color: var(--ink-soft); font-size: 0.98rem; margin: 0 auto 1.6rem auto;
    max-width: 460px; line-height: 1.55;
}

/* ---------------- Glass cards, shingled into a stack ---------------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 26px !important;
    border: 1px solid var(--panel-border) !important;
    box-shadow: 0 18px 50px rgba(0,0,0,0.45), 0 0 40px rgba(245, 162, 75, 0.06);
    max-width: 660px;
    margin: -30px auto 0 auto !important;
    position: relative;
    transition: transform 0.35s ease, box-shadow 0.35s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:first-of-type { margin-top: 0 !important; }
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 24px 60px rgba(0,0,0,0.5), 0 0 46px rgba(245, 162, 75, 0.10);
}
/* the little tab that shows a card is tucked under the one before it */
div[data-testid="stVerticalBlockBorderWrapper"]:not(:first-of-type)::before {
    content: "";
    position: absolute;
    top: 16px; left: 50%; transform: translateX(-50%);
    width: 46px; height: 5px; border-radius: 4px;
    background: linear-gradient(90deg, var(--amber), var(--teal));
    opacity: 0.65;
}
div[data-testid="stVerticalBlockBorderWrapper"]:not(:first-of-type) > div > div {
    padding-top: 14px;
}

.hero-upload-wrap { max-width: 640px; margin: 0 auto; }

.free-upload {
    max-width: 600px;
    margin: 1.4rem auto 1.8rem auto;
    position: relative;
    z-index: 5;
}

.upload-caption {
    text-align: center; font-size: 0.76rem; color: var(--ink-soft);
    margin-top: 0.8rem; letter-spacing: 0.02em;
}

/* File uploader — glowing amber dropzone, centered */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(245, 162, 75, 0.06);
    border: 1.5px dashed rgba(245, 162, 75, 0.5) !important;
    border-radius: 18px;
    justify-content: center;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--amber) !important; }
[data-testid="stFileUploaderDropzone"] section { color: var(--ink-soft); justify-content: center; }
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(90deg, var(--amber), var(--teal)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    box-shadow: 0 0 20px rgba(245, 162, 75, 0.4);
}

/* ---------------- Result block ---------------- */
.result-center { display: flex; flex-direction: column; align-items: center; text-align: center; }

.result-title {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 1.65rem;
    margin: 0.5rem 0 0.2rem 0;
    color: var(--ink);
}
.result-conf { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: var(--ink-soft); margin-bottom: 1.1rem; }

.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #1a0f1e;
    padding: 0.32rem 0.85rem;
    border-radius: 999px;
    margin-bottom: 0.7rem;
}

/* Confidence gauge, centered under the title */
.gauge {
    width: 116px; height: 116px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    margin: 0 auto 1.4rem auto;
}
.gauge-inner {
    width: 90px; height: 90px; border-radius: 50%;
    background: #170F26;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.gauge-num { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1.2rem; color: var(--ink); }
.gauge-lbl { font-size: 0.6rem; color: var(--ink-soft); letter-spacing: 0.07em; text-transform: uppercase; }

/* Severity spectrum */
.spectrum-wrap { margin-top: 0.2rem; width: 100%; max-width: 460px; margin-left: auto; margin-right: auto; }
.spectrum-track { display: flex; width: 100%; height: 10px; border-radius: 6px; overflow: hidden; }
.spectrum-seg { flex: 1; }
.spectrum-labels { display: flex; width: 100%; margin-top: 6px; }
.spectrum-labels span {
    flex: 1; font-family: 'IBM Plex Mono', monospace; font-size: 0.58rem;
    color: var(--ink-soft); text-align: center;
}
.spectrum-labels span.active { color: var(--ink); font-weight: 700; }
.spectrum-pointer-row { display: flex; width: 100%; }
.spectrum-pointer-cell { flex: 1; display: flex; justify-content: center; }
.spectrum-pointer {
    width: 0; height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-bottom: 7px solid var(--ink);
    margin-bottom: 2px;
}

/* Probability bars — centered as a block, labels still legible */
.prob-wrap { display: flex; flex-direction: column; align-items: center; }
.prob-row {
    display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.6rem;
    width: 100%; max-width: 420px;
}
.prob-name { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; width: 128px; flex-shrink: 0; color: var(--ink-soft); text-align: right; }
.prob-track { flex: 1; background: rgba(255,255,255,0.06); border-radius: 5px; height: 12px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 5px; transition: width 0.6s ease; }
.prob-pct { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; width: 44px; text-align: left; flex-shrink: 0; color: var(--ink); }

/* Low-confidence notice */
.notice {
    background: rgba(242, 121, 58, 0.12);
    border: 1px solid rgba(242, 121, 58, 0.45);
    border-radius: 14px;
    padding: 0.8rem 1.1rem;
    font-size: 0.85rem;
    color: #FFD8B0;
    margin-top: 1.1rem;
    text-align: center;
}

/* About card */
.about-card h4, .about-inline h4 {
    font-family: 'Sora', sans-serif; margin-top: 0; color: var(--ink); font-weight: 700;
    text-align: center;
}
.chip-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.8rem 0 1rem 0; justify-content: center; }
.status-line {
    text-align: center; font-size: 0.85rem; color: var(--ink-soft);
    margin: 1.2rem 0 0.2rem 0; letter-spacing: 0.01em;
}
.chip {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--ink);
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px; padding: 0.32rem 0.8rem;
}
.about-copy {
    font-size: 0.85rem; color: var(--ink-soft); line-height: 1.65; margin: 0 auto;
    max-width: 480px; text-align: center;
}

/* Image captions centered too */
[data-testid="stImage"] { display: flex; justify-content: center; }

.stAlert { background: var(--panel) !important; border-radius: 16px !important; }
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
# HERO + UPLOAD — one centered card, everything shingled beneath it
# ----------------------------------------------------------------------------
left, mid, right = st.columns([1, 3, 1])
with mid:
    hero_card = st.container(border=True)
    with hero_card:
        st.markdown("""
        <div class="hero-wrap">
            <div class="scan-rings">
                <div class="hero-icon">
                    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M2 12C2 12 5.5 5 12 5C18.5 5 22 12 22 12C22 12 18.5 19 12 19C5.5 19 2 12 2 12Z" stroke="white" stroke-width="1.6" stroke-linejoin="round"/>
                        <circle cx="12" cy="12" r="3.2" stroke="white" stroke-width="1.6"/>
                        <circle cx="12" cy="12" r="1" fill="white"/>
                    </svg>
                </div>
            </div>
            <div class="hero-title">Diabetic Retinopathy Grading</div>
            <div class="hero-sub">Upload a fundus photo to grade DR severity, with a Grad-CAM view of the model's focus.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="free-upload">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a fundus image",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="upload-caption">JPG · PNG supported &nbsp;•&nbsp; Instant grading</div></div>',
        unsafe_allow_html=True,
    )

    if uploaded_file is None:
        st.markdown(
            '<div class="status-line">Upload a fundus image above to run a grading.</div>',
            unsafe_allow_html=True,
        )

    about_card = st.container(border=True)
    with about_card:
        st.markdown("""
        <h4>About this model</h4>
        <div class="chip-row">
            <span class="chip">Swin V2 Tiny</span>
            <span class="chip">QWK 0.797</span>
            <span class="chip">Accuracy 72%</span>
        </div>
        <p class="about-copy">
            Academic prototype — not validated for clinical use. Confidence reflects certainty
            on this image only, not overall accuracy.
        </p>
        """, unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    pred_class, confidence, probs, overlay_img = predict(image)
    pred_label = class_names[pred_class]
    pred_color = SEVERITY_COLORS[pred_label]

    # --- Images ---
    img_card = st.container(border=True)
    with img_card:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.image(overlay_img, caption="Grad-CAM — Model Focus Area", use_container_width=True)

    # --- Result ---
    result_card = st.container(border=True)
    with result_card:
        gauge_pct = min(max(confidence, 0), 100)
        st.markdown(f"""
        <div class="result-center">
            <span class="badge" style="background:{pred_color};">{pred_label.upper()}</span>
            <div class="result-title">{pred_label}</div>
            <div class="result-conf">Model confidence for this image</div>
            <div class="gauge" style="background: conic-gradient({pred_color} {gauge_pct*3.6}deg, rgba(255,255,255,0.08) 0deg);">
                <div class="gauge-inner">
                    <div class="gauge-num">{confidence:.0f}%</div>
                    <div class="gauge-lbl">confidence</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Severity spectrum with pointer
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

    # --- Probabilities ---
    prob_card = st.container(border=True)
    with prob_card:
        st.markdown('<h4 style="text-align:center;">Class probabilities</h4>', unsafe_allow_html=True)
        rows_html = ""
        for i, cname in enumerate(class_names):
            pct = float(probs[i]) * 100
            color = SEVERITY_COLORS[cname]
            rows_html += f"""
            <div class="prob-row">
                <div class="prob-name">{cname}</div>
                <div class="prob-track"><div class="prob-fill" style="width:{pct:.1f}%; background:{color};"></div></div>
                <div class="prob-pct">{pct:.1f}%</div>
            </div>
            """
        st.markdown(f'<div class="prob-wrap">{rows_html}</div>', unsafe_allow_html=True)

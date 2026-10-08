"""
PulmoVision AI — Clinical Pulmonary Diagnostic Platform
Professional Medical-Grade UI with PDF Report Generation
Architecture: CLAHE → U-Net Segmentation → DenseNet-121 → Grad-CAM XAI
"""

import io
import os
import sys
import time
import base64
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PulmoVision AI — Clinical Diagnostic Platform",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"]
CLASS_COLORS = {
    "Normal":              "#0D9488",   # teal-600
    "Bacterial Pneumonia": "#DC2626",   # red-600
    "Viral Pneumonia":     "#D97706",   # amber-600
}
CLASS_HEX_LIGHT = {
    "Normal":              "#F0FDFA",
    "Bacterial Pneumonia": "#FEF2F2",
    "Viral Pneumonia":     "#FFFBEB",
}

DISEASE_INFO = {
    "Normal": {
        "icon": "✓",
        "tagline": "Clear lung fields with no abnormal opacities, consolidation, or pleural effusion.",
        "key_markers": ["Clear Lung Fields", "Normal Cardiac Size", "Sharp Margins"],
        "description": (
            "A normal chest X-ray shows clear lung fields with no signs of consolidation, "
            "infiltrates, or abnormal opacities. The lung parenchyma appears homogeneously "
            "translucent, cardiac silhouette is within normal limits, and diaphragmatic contours "
            "are sharp and well-defined."
        ),
        "findings": [
            "Clear lung fields bilaterally",
            "No consolidation or infiltrates",
            "Normal cardiac silhouette",
            "Sharp costophrenic angles",
            "No pleural effusion",
        ],
        "recommendation": "Routine follow-up. No immediate intervention required.",
    },
    "Bacterial Pneumonia": {
        "icon": "⚑",
        "tagline": "Focal or lobar opacity typically caused by bacterial pathogens like S. pneumoniae.",
        "key_markers": ["Lobar Consolidation", "Air Bronchograms", "Unilateral Pattern"],
        "description": (
            "Bacterial pneumonia produces focal or lobar consolidation visible as areas of "
            "increased opacity on chest X-ray. It is typically caused by organisms such as "
            "Streptococcus pneumoniae, Klebsiella, or Staphylococcus aureus. Early diagnosis "
            "and targeted antibiotic therapy are critical to prevent complications such as "
            "empyema or respiratory failure."
        ),
        "findings": [
            "Focal lobar or segmental consolidation",
            "Air bronchograms within opacities",
            "Silhouette sign of adjacent structures",
            "Possible pleural effusion",
            "Predominantly unilateral distribution",
        ],
        "recommendation": (
            "Urgent radiologist review recommended. Initiate sputum culture, "
            "CBC with differential, and empiric antibiotic therapy per local guidelines."
        ),
    },
    "Viral Pneumonia": {
        "icon": "⚑",
        "tagline": "Diffuse bilateral ground-glass opacities & interstitial thickening.",
        "key_markers": ["Bilateral Opacities", "Ground-Glass Pattern", "Interstitial Infiltrates"],
        "description": (
            "Viral pneumonia is characterized by diffuse bilateral ground-glass opacities "
            "and interstitial infiltrates. Common causative agents include Influenza A/B, "
            "RSV, COVID-19 (SARS-CoV-2), and Adenovirus. The diffuse pattern reflects "
            "alveolar and interstitial inflammation rather than the focal consolidation "
            "typical of bacterial disease."
        ),
        "findings": [
            "Bilateral ground-glass opacities",
            "Peribronchovascular interstitial thickening",
            "Diffuse bilateral distribution",
            "Possible crazy-paving pattern (COVID-19)",
            "Lower lobe predominance common",
        ],
        "recommendation": (
            "Viral panel and PCR testing recommended. Supportive care with oxygen monitoring. "
            "Consider antiviral therapy if clinically indicated."
        ),
    },
}

# ─────────────────────────────────────────────────────────────
# CSS — Professional Medical Website
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    box-sizing: border-box;
}

/* ── Global Reset & Comfortable Soft Theme ── */
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }
.stApp { background: #F0F4F8; }
[data-testid="stFileUploader"] label { font-size: 0.9rem !important; }

/* ── Nav Bar ── */
.pv-nav {
    background: #0F172A;
    padding: 0 2.5rem;
    height: 62px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 1000;
}
.pv-nav-logo {
    display: flex; align-items: center; gap: 10px;
    font-size: 1.05rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.2px;
}
.pv-nav-logo em { color: #38BDF8; font-style: normal; }
.pv-nav-tag {
    background: rgba(56,189,248,0.15); color: #38BDF8;
    border: 1px solid rgba(56,189,248,0.35);
    font-size: 0.68rem; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; padding: 2px 9px; border-radius: 20px;
}
.pv-nav-right {
    display: flex; align-items: center; gap: 1.8rem;
    font-size: 0.82rem; color: #94A3B8; font-weight: 500;
}
.pv-nav-right span { cursor: default; }

/* ── Hero ── */
.pv-hero {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 55%, #0F172A 100%);
    padding: 4rem 2.5rem 3.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.pv-hero::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(56,189,248,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.pv-eyebrow {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #38BDF8; margin-bottom: 1rem;
}
.pv-hero-title {
    font-size: 3rem; font-weight: 900; color: #F8FAFC;
    letter-spacing: -1.5px; line-height: 1.1; margin: 0 auto 1rem;
    max-width: 700px;
}
.pv-hero-title em { color: #38BDF8; font-style: normal; }
.pv-hero-sub {
    font-size: 1.05rem; color: #94A3B8; line-height: 1.7;
    max-width: 560px; margin: 0 auto 1.8rem;
}
.pv-hero-chips {
    display: flex; justify-content: center; gap: 0.6rem; flex-wrap: wrap;
}
.pv-chip {
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);
    color: #CBD5E1; font-size: 0.78rem; font-weight: 500;
    padding: 5px 14px; border-radius: 20px;
}

/* ── Metrics Strip ── */
.pv-metrics {
    display: flex; background: #E6EDF5;
    border-bottom: 1px solid #D5E0EB; border-top: 1px solid #D5E0EB;
}
.pv-metric {
    flex: 1; text-align: center; padding: 1.3rem 1rem;
    border-right: 1px solid #D5E0EB;
}
.pv-metric:last-child { border-right: none; }
.pv-metric-val { font-size: 1.65rem; font-weight: 800; color: #0F172A; letter-spacing: -0.5px; }
.pv-metric-lbl { font-size: 0.72rem; color: #64748B; font-weight: 600; margin-top: 3px;
    text-transform: uppercase; letter-spacing: 0.6px; }

/* ── Content Wrapper ── */
.pv-wrap { max-width: 1160px; margin: 0 auto; padding: 2.5rem 2rem; }

/* ── Section Headers ── */
.pv-section-eyebrow {
    font-size: 0.68rem; font-weight: 700; color: #0284C7; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 0.3rem;
}
.pv-section-title {
    font-size: 1.5rem; font-weight: 800; color: #0F172A;
    letter-spacing: -0.4px; margin-bottom: 1.3rem;
}

/* ── Divider ── */
.pv-divider { border: none; border-top: 1px solid #CBD5E1; margin: 2.2rem 0; }

/* ── Info Cards (Disease Info) ── */
.pv-info-card {
    background: #F8FAFC; border: 1px solid #D5E0EB; border-radius: 14px;
    padding: 1.6rem; height: 100%;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
    transition: box-shadow 0.2s, border-color 0.2s;
}
.pv-info-card:hover { box-shadow: 0 4px 16px rgba(15,23,42,0.08); border-color: #CBD5E1; }
.pv-info-card-title {
    font-size: 1.05rem; font-weight: 700; color: #0F172A;
    margin-bottom: 0.6rem; display: flex; align-items: center; gap: 8px;
}
.pv-info-card-body {
    font-size: 0.85rem; color: #334155; line-height: 1.7; margin-bottom: 0.9rem;
}
.pv-finding-list {
    list-style: none; padding: 0; margin: 0 0 0.9rem;
}
.pv-finding-list li {
    font-size: 0.82rem; color: #334155; padding: 3px 0;
    display: flex; align-items: flex-start; gap: 6px;
}
.pv-finding-list li::before { content: "—"; color: #94A3B8; }
.pv-reco {
    font-size: 0.82rem; font-weight: 500; color: #0F172A;
    background: #EAEFF5; border-left: 3px solid;
    padding: 0.6rem 0.85rem; border-radius: 0 6px 6px 0;
}

/* ── Upload Section ── */
.pv-upload-box {
    background: #F8FAFC; border: 1.5px dashed #94A3B8;
    border-radius: 12px; padding: 1.8rem; text-align: center;
    transition: border-color 0.2s;
}

/* ── File Meta Table ── */
.pv-meta-table { width: 100%; border-collapse: collapse; }
.pv-meta-table tr { border-bottom: 1px solid #E2E8F0; }
.pv-meta-table tr:last-child { border-bottom: none; }
.pv-meta-table td { padding: 7px 0; font-size: 0.85rem; }
.pv-meta-table td:first-child { color: #64748B; font-weight: 500; width: 130px; }
.pv-meta-table td:last-child { color: #0F172A; font-weight: 500; }

/* ── Primary Button ── */
div[data-testid="stButton"] > button {
    background: #0F172A !important;
    color: #F8FAFC !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 0.72rem 1.8rem !important;
    border-radius: 8px !important;
    border: none !important;
    letter-spacing: 0.2px !important;
    transition: background 0.2s !important;
    box-shadow: 0 2px 4px rgba(15,23,42,0.12) !important;
}
div[data-testid="stButton"] > button:hover {
    background: #1E3A5F !important;
}

/* ── Pipeline Panel Labels ── */
.pv-panel-step {
    font-size: 0.68rem; font-weight: 700; color: #64748B;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 5px;
}
.pv-panel-cap {
    font-size: 0.78rem; color: #64748B; text-align: center; margin-top: 5px;
}

/* ── Diagnosis Result Card ── */
.pv-dx-card {
    background: #F8FAFC; border: 1px solid #D5E0EB;
    border-radius: 14px; padding: 1.8rem 2rem; border-top: 4px solid;
    box-shadow: 0 2px 8px rgba(15,23,42,0.04);
}
.pv-dx-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; margin-bottom: 0.5rem;
}
.pv-dx-name {
    font-size: 2.2rem; font-weight: 900;
    letter-spacing: -0.8px; line-height: 1.1; margin-bottom: 0.3rem;
}
.pv-dx-conf {
    font-size: 0.88rem; color: #334155; font-weight: 500; margin-bottom: 1rem;
}

/* ── Clinical Note ── */
.pv-clinical-note {
    border-radius: 8px; padding: 0.95rem 1.1rem;
    font-size: 0.84rem; line-height: 1.65; border: 1px solid;
}
.pv-cn-teal   { background:#E6FFFA; border-color:#99F6E4; color:#134E4A; }
.pv-cn-red    { background:#FFEAEB; border-color:#FECDD3; color:#9F1239; }
.pv-cn-amber  { background:#FEF3C7; border-color:#FDE68A; color:#92400E; }

/* ── Probability Bars ── */
.pv-prob-row { display:flex; align-items:center; gap:0.75rem; margin-bottom:0.8rem; }
.pv-prob-cls { font-size:0.84rem; color:#1E293B; font-weight:500; width:160px; flex-shrink:0; }
.pv-prob-bg  { flex:1; background:#E2E8F0; border-radius:6px; height:9px; }
.pv-prob-bar { border-radius:6px; height:9px; }
.pv-prob-pct { font-size:0.82rem; color:#334155; font-weight:600; width:46px; text-align:right; flex-shrink:0; }

/* ── Download Button ── */
div[data-testid="stDownloadButton"] > button {
    background: #F8FAFC !important;
    color: #0F172A !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.5rem !important;
    border-radius: 8px !important;
    border: 1.5px solid #94A3B8 !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: #E2E8F0 !important;
    border-color: #64748B !important;
}

/* ── Footer ── */
.pv-footer {
    background: #0F172A; color: #64748B;
    font-size: 0.78rem; text-align: center;
    padding: 1.8rem 2.5rem; margin-top: 3rem;
    line-height: 1.8;
}
.pv-footer a { color: #38BDF8; text-decoration: none; }

/* Image panels */
div[data-testid="stImage"] img { border-radius: 10px; border: 1px solid #E2E8F0; }

/* Spinner */
.stSpinner > div { color: #38BDF8 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# NEURAL NETWORK MODELS
# ─────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)
        for f in features:
            self.downs.append(DoubleConv(in_channels, f)); in_channels = f
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.ups.append(DoubleConv(f*2, f))
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x); skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x); s = skips[i//2]
            if x.shape != s.shape:
                x = F.interpolate(x, s.shape[2:], mode="bilinear", align_corners=True)
            x = self.ups[i+1](torch.cat((s, x), 1))
        return self.final_conv(x)


class DenseNet121PneumoniaClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, dropout_rate=0.3):
        super().__init__()
        try:
            w = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.backbone = models.densenet121(weights=w)
        except AttributeError:
            self.backbone = models.densenet121(pretrained=pretrained)
        inf = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(inf, 512),
            nn.BatchNorm1d(512), nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate/2), nn.Linear(512, num_classes),
        )
    def forward(self, x): return self.backbone(x)


def clear_hooks(model):
    for m in model.modules():
        m._forward_hooks.clear(); m._backward_hooks.clear(); m._forward_pre_hooks.clear()


class disable_inplace_relu:
    def __enter__(self):
        self._orig = F.relu
        F.relu = lambda x, inplace=False: self._orig(x, False)
    def __exit__(self, *a): F.relu = self._orig


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model; self.target_layer = target_layer
        self.gradients = self.activations = None
        clear_hooks(model)
        for m in model.modules():
            if isinstance(m, nn.ReLU): m.inplace = False
        target_layer.register_forward_hook(self._fwd)

    def _fwd(self, _, __, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "gradients", g.detach().clone()))

    def generate_heatmap(self, inp, target_class=None):
        self.model.eval()
        t = inp.clone().detach().requires_grad_(True)
        with disable_inplace_relu(): out = self.model(t)
        tc = target_class if target_class is not None else out.argmax(1).item()
        self.model.zero_grad(); out[0, tc].backward()
        if self.gradients is None or self.activations is None:
            return np.zeros(inp.shape[2:], np.float32)
        w   = self.gradients[0].mean((1,2), keepdim=True)
        cam = F.relu(torch.sum(w * self.activations[0], 0))
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (inp.shape[3], inp.shape[2]))
        if cam.max() > 0: cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay_heatmap(self, img, heatmap, alpha=0.45):
        h, w = img.shape[:2]
        h8 = np.uint8(255 * cv2.resize(heatmap, (w,h)))
        cmap = cv2.applyColorMap(h8, cv2.COLORMAP_JET)
        base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        return cv2.addWeighted(base, 1-alpha, cmap, alpha, 0)


# ─────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────
def safe_load(path, device):
    try:    return torch.load(path, map_location=device, weights_only=False)
    except: return torch.load(path, map_location=device)


@st.cache_resource
def load_models():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = UNet(1,1).to(dev); u_ok = False
    for p in [Path("weights/lung_unet_segmentation.pth"), Path("lung_unet_segmentation.pth")]:
        if p.exists():
            try:
                sd = safe_load(p, dev)
                if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
                unet.load_state_dict(sd); unet.eval(); u_ok = True; break
            except: pass

    clf = DenseNet121PneumoniaClassifier(3).to(dev); c_ok = False
    for p in [Path("weights/best_densenet121_pneumonia_classifier.pth"), Path("best_densenet121_pneumonia_classifier.pth")]:
        if p.exists():
            try:
                ckpt = safe_load(p, dev)
                clf.load_state_dict(ckpt.get("model_state_dict", ckpt)); clf.eval()
                clear_hooks(clf); c_ok = True; break
            except: pass

    gcam = GradCAM(clf, clf.backbone.features[-1])
    return (unet, u_ok), (clf, c_ok), gcam, dev


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
def clahe_enhance(img, clip=2.0, tile=(8,8)):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(img)


def extract_roi(img_clahe, unet, dev):
    h, w = img_clahe.shape[:2]
    t = torch.from_numpy(cv2.resize(img_clahe,(256,256))).float().div(255).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        prob = torch.sigmoid(unet(t)).cpu().numpy()[0,0]
    pn = np.uint8(255*(prob-prob.min())/(prob.max()-prob.min()+1e-8))
    _, m = cv2.threshold(pn, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cx, bx = m[64:192,64:192], np.concatenate([m[:32,:],m[-32:,:],m[:,:32].T,m[:,-32:].T])
    if bx.mean() > cx.mean(): m = cv2.bitwise_not(m)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    mask = cv2.resize(m, (w,h), interpolation=cv2.INTER_NEAREST)
    if mask is None or np.sum(mask) < 0.01*mask.size*255:
        _, th = cv2.threshold(cv2.GaussianBlur(img_clahe,(5,5),0), 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cl = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
        cnts,_ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(img_clahe, np.uint8)
        if cnts: cv2.drawContours(mask, sorted(cnts, key=cv2.contourArea, reverse=True)[:2], -1, 255, cv2.FILLED)
        else:    mask[:] = 255
    return cv2.bitwise_and(img_clahe, img_clahe, mask=mask), mask


# ─────────────────────────────────────────────────────────────
# PDF REPORT GENERATOR
# ─────────────────────────────────────────────────────────────
def _np_to_tempfile(arr: np.ndarray) -> str:
    """Save numpy image to a real temp file on disk. Returns the file path.
    This prevents ReportLab's lazy-reader from hitting a GC'd BytesIO object."""
    if arr.ndim == 2:
        pil = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    else:
        pil = Image.fromarray(arr.astype(np.uint8))
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    pil.save(path, format="PNG")
    return path


def generate_pdf_report(
    filename: str, w_in: int, h_in: int, filesize_kb: float, device_str: str,
    pred_label: str, pred_conf: float, probs: np.ndarray, inference_ms: float,
    img_raw: np.ndarray, img_clahe: np.ndarray, img_roi: np.ndarray, overlay_rgb: np.ndarray,
) -> Tuple[bytes, str]:
    """Returns (pdf_bytes, error_str). error_str is empty string on success."""
    tmp_paths = []   # keep track for cleanup
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.enums import TA_CENTER, TA_RIGHT
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                         TableStyle, HRFlowable, Image as RLImage)
    except ImportError as ie:
        return b"", f"reportlab not installed: {ie}"

    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=1.8*cm, rightMargin=1.8*cm,
            topMargin=1.6*cm, bottomMargin=1.6*cm,
        )

        W, H = A4
        NAVY  = colors.HexColor("#0F172A")
        TEAL  = colors.HexColor("#0D9488")
        RED   = colors.HexColor("#DC2626")
        AMBER = colors.HexColor("#D97706")
        LGRAY = colors.HexColor("#F1F5F9")
        MGRAY = colors.HexColor("#64748B")
        WHITE = colors.white
        DXCOL = {"Normal": TEAL, "Bacterial Pneumonia": RED, "Viral Pneumonia": AMBER}
        pred_color = DXCOL[pred_label]

        SS = getSampleStyleSheet()
        # Style cloner — generates a fresh ParagraphStyle each call
        _used_style_names = {}
        def S(base, **kw):
            key = base + str(sorted(kw.items()))
            if key not in _used_style_names:
                s = SS[base].clone(f"{base}_{len(_used_style_names)}")
                for k, v in kw.items():
                    setattr(s, k, v)
                _used_style_names[key] = s
            return _used_style_names[key]

        def HR(color=None, width=1):
            c = color if color is not None else LGRAY
            return HRFlowable(width="100%", thickness=width, color=c, spaceAfter=8, spaceBefore=4)

        def rl_img(arr, w_cm):
            """Save numpy array to a temp file on disk, return RLImage.
            MUST use disk files — BytesIO objects get GC'd before doc.build() reads them."""
            p = _np_to_tempfile(arr)
            tmp_paths.append(p)
            tw = w_cm * cm
            # Get aspect ratio from PIL
            pil_tmp = Image.open(p)
            aspect = pil_tmp.height / pil_tmp.width
            pil_tmp.close()
            return RLImage(p, width=tw, height=tw * aspect)

        story = []
        ts = time.strftime("%B %d, %Y  %H:%M:%S")

        # ── HEADER BAR ──
        hdr = Table([[
            Paragraph("<b>PulmoVision AI</b>",
                      S("Normal", fontSize=16, textColor=WHITE, fontName="Helvetica-Bold")),
            Paragraph("Clinical Pulmonary Diagnostic Report",
                      S("Normal", fontSize=9, textColor=colors.HexColor("#94A3B8"), alignment=TA_RIGHT)),
        ]], colWidths=[(W-3.6*cm)*0.6, (W-3.6*cm)*0.4])
        hdr.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), NAVY),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("LEFTPADDING",   (0,0),(-1,-1), 14),
            ("RIGHTPADDING",  (0,0),(-1,-1), 14),
            ("TOPPADDING",    (0,0),(-1,-1), 12),
            ("BOTTOMPADDING", (0,0),(-1,-1), 12),
        ]))
        story += [hdr, Spacer(1, 14)]

        # ── METADATA TABLE ──
        meta_rows = [
            ["Report Generated", ts],
            ["Image File",       filename],
            ["Native Resolution",f"{w_in} x {h_in} px"],
            ["File Size",        f"{filesize_kb:.1f} KB"],
            ["Compute Device",   device_str.upper()],
            ["Inference Latency",f"{inference_ms:.1f} ms"],
        ]
        mt = Table(
            [[Paragraph(r[0], S("Normal", fontSize=8.5, textColor=MGRAY,
                                 fontName="Helvetica-Bold", leading=13)),
              Paragraph(r[1], S("Normal", fontSize=9, textColor=colors.HexColor("#334155"), leading=13))]
             for r in meta_rows],
            colWidths=[(W-3.6*cm)*0.30, (W-3.6*cm)*0.70]
        )
        mt.setStyle(TableStyle([
            ("ROWBACKGROUNDS", (0,0),(-1,-1), [WHITE, LGRAY]),
            ("BOX",            (0,0),(-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ("GRID",           (0,0),(-1,-1), 0.3, colors.HexColor("#E2E8F0")),
            ("LEFTPADDING",    (0,0),(-1,-1), 10),
            ("RIGHTPADDING",   (0,0),(-1,-1), 10),
            ("TOPPADDING",     (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
        ]))
        story += [mt, Spacer(1, 20)]

        # ── PRIMARY DIAGNOSIS ──
        story.append(HR(color=pred_color, width=2))
        story.append(Paragraph("PRIMARY DIAGNOSIS",
                                S("Normal", fontSize=8, textColor=MGRAY, fontName="Helvetica-Bold",
                                  leading=12, spaceAfter=4)))
        story.append(Paragraph(pred_label,
                                S("Heading1", fontSize=26, textColor=pred_color,
                                  fontName="Helvetica-Bold", leading=30, spaceAfter=4)))
        story.append(Paragraph(
            f"Confidence Score: <b>{pred_conf*100:.2f}%</b>  ·  Inference Latency: <b>{inference_ms:.1f} ms</b>",
            S("Normal", fontSize=10, textColor=NAVY, leading=14)
        ))
        story.append(Spacer(1, 10))

        # Probability breakdown table
        prob_header = [
            Paragraph("Diagnostic Class", S("Normal", fontSize=9, textColor=WHITE,
                                              fontName="Helvetica-Bold", leading=13)),
            Paragraph("Confidence",       S("Normal", fontSize=9, textColor=WHITE,
                                              fontName="Helvetica-Bold", leading=13)),
            Paragraph("Status",           S("Normal", fontSize=9, textColor=WHITE,
                                              fontName="Helvetica-Bold", leading=13)),
        ]
        prob_rows = [prob_header]
        for i, cls in enumerate(CLASS_NAMES):
            pct = probs[i] * 100
            is_pred = (cls == pred_label)
            clr = DXCOL[cls] if is_pred else colors.HexColor("#334155")
            prob_rows.append([
                Paragraph(f"<b>{cls}</b>" if is_pred else cls,
                           S("Normal", fontSize=9.5, textColor=clr, leading=14)),
                Paragraph(f"<b>{pct:.2f}%</b>" if is_pred else f"{pct:.2f}%",
                           S("Normal", fontSize=9.5, textColor=clr, leading=14)),
                Paragraph("PREDICTED" if is_pred else "—",
                           S("Normal", fontSize=8,
                             textColor=clr if is_pred else MGRAY,
                             fontName="Helvetica-Bold" if is_pred else "Helvetica", leading=12)),
            ])
        pt = Table(prob_rows, colWidths=[(W-3.6*cm)*0.45, (W-3.6*cm)*0.30, (W-3.6*cm)*0.25])
        pt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0),  NAVY),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LGRAY]),
            ("BOX",           (0,0),(-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#E2E8F0")),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
            ("RIGHTPADDING",  (0,0),(-1,-1), 10),
            ("TOPPADDING",    (0,0),(-1,-1), 6),
            ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ]))
        story += [pt, Spacer(1, 12)]

        # Clinical recommendation box
        info = DISEASE_INFO[pred_label]
        note_bg = {"Normal": colors.HexColor("#F0FDFA"),
                   "Bacterial Pneumonia": colors.HexColor("#FFF1F2"),
                   "Viral Pneumonia": colors.HexColor("#FFFBEB")}[pred_label]
        note_bdr = {"Normal": TEAL, "Bacterial Pneumonia": RED, "Viral Pneumonia": AMBER}[pred_label]
        reco_t = Table(
            [[Paragraph(f"<b>Clinical Recommendation:</b> {info['recommendation']}",
                        S("Normal", fontSize=9.5, textColor=colors.HexColor("#334155"), leading=15))]],
            colWidths=[W-3.6*cm]
        )
        reco_t.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,-1), note_bg),
            ("LEFTPADDING",  (0,0),(-1,-1), 14),
            ("RIGHTPADDING", (0,0),(-1,-1), 14),
            ("TOPPADDING",   (0,0),(-1,-1), 8),
            ("BOTTOMPADDING",(0,0),(-1,-1), 8),
            ("LINEBEFORE",   (0,0),(0,-1),  3, note_bdr),
            ("BOX",          (0,0),(-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ]))
        story += [reco_t, Spacer(1, 22)]

        # ── IMAGING PIPELINE ──
        story.append(HR())
        story.append(Paragraph("Diagnostic Imaging Pipeline",
                                S("Heading2", fontSize=13, textColor=NAVY,
                                  fontName="Helvetica-Bold", leading=16, spaceAfter=4)))
        story.append(Paragraph(
            "The four-stage pipeline below was executed sequentially on the submitted radiograph. "
            "Each transformation isolates and highlights pathological features for classification.",
            S("Normal", fontSize=9.5, textColor=colors.HexColor("#334155"), leading=15, spaceAfter=4)
        ))
        story.append(Spacer(1, 8))

        # Save images to disk temp files so ReportLab can read them lazily during build
        IW_cm = (W - 3.6*cm - 3*4) / 4 / cm    # 4 images across, 4pt gaps
        imgs = [img_raw, img_clahe, img_roi, overlay_rgb]
        img_titles = ["(1) Original CXR", "(2) CLAHE Enhanced",
                      "(3) U-Net ROI",    f"(4) Grad-CAM"]
        img_subs   = ["Raw PA input",          "Adaptive contrast eq.",
                      "Lung parenchyma",        pred_label]

        img_cells  = [rl_img(a, IW_cm) for a in imgs]
        title_cells = [Paragraph(t, S("Normal", fontSize=8, textColor=NAVY,
                                       fontName="Helvetica-Bold", alignment=TA_CENTER, leading=11))
                       for t in img_titles]
        sub_cells   = [Paragraph(s, S("Normal", fontSize=7.5, textColor=MGRAY,
                                       alignment=TA_CENTER, leading=10))
                       for s in img_subs]

        IW_pt = IW_cm * cm
        it = Table([img_cells, title_cells, sub_cells],
                   colWidths=[IW_pt, IW_pt, IW_pt, IW_pt])
        it.setStyle(TableStyle([
            ("VALIGN",        (0,0),(-1,-1), "TOP"),
            ("ALIGN",         (0,0),(-1,-1), "CENTER"),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 2),
            ("RIGHTPADDING",  (0,0),(-1,-1), 2),
        ]))
        story += [it, Spacer(1, 22)]

        # ── DISEASE PROFILE ──
        story.append(HR())
        story.append(Paragraph("Disease Profile & Radiological Features",
                                S("Heading2", fontSize=13, textColor=NAVY,
                                  fontName="Helvetica-Bold", leading=16, spaceAfter=4)))
        story.append(Paragraph(
            f"The following section details the radiological characteristics of "
            f"<b>{pred_label}</b> as relevant to the predicted diagnosis.",
            S("Normal", fontSize=9.5, textColor=colors.HexColor("#334155"), leading=15, spaceAfter=6)
        ))
        story.append(Paragraph(info["description"],
                                S("Normal", fontSize=9.5, textColor=colors.HexColor("#475569"),
                                  leading=15, spaceAfter=4)))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Expected Radiological Findings",
                                S("Normal", fontSize=8.5, textColor=MGRAY,
                                  fontName="Helvetica-Bold", leading=13, spaceAfter=3)))
        for f in info["findings"]:
            story.append(Paragraph(f"• {f}",
                                    S("Normal", fontSize=9.5, textColor=colors.HexColor("#334155"),
                                      leading=15, leftIndent=14)))
        story.append(Spacer(1, 18))

        # ── MODEL ARCHITECTURE ──
        story.append(HR())
        story.append(Paragraph("AI Model Architecture & Performance Metrics",
                                S("Heading2", fontSize=13, textColor=NAVY,
                                  fontName="Helvetica-Bold", leading=16, spaceAfter=4)))
        arch_rows = [
            ["Component",           "Details"],
            ["Classifier Model",    "DenseNet-121 with custom 3-class head: Dropout(0.3) → Linear(1024→512) → BN1d → SiLU → Dropout(0.15) → Linear(512→3)"],
            ["Segmentation Model",  "U-Net with encoder features [64, 128, 256, 512], skip connections, bilinear upsampling"],
            ["XAI / Explainability","Gradient-weighted Class Activation Mapping (Grad-CAM) — target: denseblock4"],
            ["Input Resolution",    "384 × 384 px, 3-channel (grayscale × 3), ImageNet mean/std normalization"],
            ["Loss Function",       "Focal Loss (gamma=2) with class-balancing weights"],
            ["Optimizer",           "AdamW: lr=1e-4, weight_decay=1e-4"],
            ["LR Scheduler",        "CosineAnnealingLR: T_max=20 epochs"],
            ["Test Accuracy",       "75.96% on 624 held-out test images"],
            ["Disease Sensitivity", "99.2% (387 / 390 pneumonia cases correctly identified)"],
            ["Normal Precision",    "98.16% (minimal false positive disease alerts)"],
            ["Bacterial Recall",    "94.63%"],
            ["Training Dataset",    "Guangzhou Women & Children Medical Center Chest X-Ray Dataset (Kermany et al., Cell 2018)"],
        ]
        col0w = (W-3.6*cm)*0.30; col1w = (W-3.6*cm)*0.70
        arch_table_rows = []
        for i, r in enumerate(arch_rows):
            is_header = (i == 0)
            arch_table_rows.append([
                Paragraph(r[0], S("Normal", fontSize=8.5 if not is_header else 9,
                                   textColor=WHITE if is_header else MGRY if False else MGRAY,
                                   fontName="Helvetica-Bold", leading=13)),
                Paragraph(r[1], S("Normal", fontSize=8.5 if not is_header else 9,
                                   textColor=WHITE if is_header else colors.HexColor("#334155"),
                                   fontName="Helvetica-Bold" if is_header else "Helvetica", leading=13)),
            ])
        at = Table(arch_table_rows, colWidths=[col0w, col1w])
        at.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0),  NAVY),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [WHITE, LGRAY]),
            ("BOX",           (0,0),(-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#E2E8F0")),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
            ("RIGHTPADDING",  (0,0),(-1,-1), 10),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ]))
        story += [at, Spacer(1, 20)]

        # ── DISCLAIMER ──
        story.append(HR())
        disc = Table(
            [[Paragraph(
                "<b>IMPORTANT DISCLAIMER</b><br/>"
                "This report is generated by an artificial intelligence research system for educational "
                "and informational purposes only. It is NOT a substitute for professional medical advice, "
                "diagnosis, or treatment by a qualified, licensed radiologist or physician. "
                "All AI-generated findings must be independently reviewed and verified by a clinician "
                "before any clinical decisions are made. The developers of PulmoVision AI assume no "
                "liability for decisions made based on this report.",
                S("Normal", fontSize=8, textColor=colors.HexColor("#92400E"), leading=13, alignment=TA_CENTER)
            )]],
            colWidths=[W-3.6*cm]
        )
        disc.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,-1), colors.HexColor("#FFFBEB")),
            ("BOX",          (0,0),(-1,-1), 0.5, colors.HexColor("#FDE68A")),
            ("LEFTPADDING",  (0,0),(-1,-1), 14),
            ("RIGHTPADDING", (0,0),(-1,-1), 14),
            ("TOPPADDING",   (0,0),(-1,-1), 10),
            ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ]))
        story += [disc, Spacer(1, 8)]
        story.append(Paragraph(
            f"PulmoVision AI  ·  Report generated {ts}  ·  For research use only",
            S("Normal", fontSize=8, textColor=MGRAY, leading=12, alignment=TA_CENTER)
        ))

        doc.build(story)
        return buf.getvalue(), ""

    except Exception as exc:
        import traceback
        return b"", traceback.format_exc()

    finally:
        # Clean up all temp image files from disk
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass





# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────
def prob_bar(label, prob, color, bold=False):
    fw = "700" if bold else "500"
    return f"""
    <div class="pv-prob-row">
        <div class="pv-prob-cls" style="font-weight:{fw};">{label}</div>
        <div class="pv-prob-bg"><div class="pv-prob-bar" style="width:{prob*100:.1f}%;background:{color};"></div></div>
        <div class="pv-prob-pct">{prob*100:.1f}%</div>
    </div>"""


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    (unet, u_ok), (clf, c_ok), gcam, dev = load_models()

    # ── NAV ──
    st.markdown("""
    <div class="pv-nav">
      <div class="pv-nav-logo">
        🫁&nbsp; Pulmo<em>Vision</em>&nbsp;AI
        <span class="pv-nav-tag">BETA</span>
      </div>
      <div class="pv-nav-right">
        <span>Diagnostics</span>
        <span>About</span>
        <span>Documentation</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── HERO ──
    st.markdown("""
    <div class="pv-hero">
      <div class="pv-eyebrow">AI-Powered Pulmonary Radiology</div>
      <h1 class="pv-hero-title">Clinical Chest X-Ray<br><em>Diagnostic Intelligence</em></h1>
      <p class="pv-hero-sub">
        Upload a PA chest radiograph for automated pneumonia classification
        across three diagnostic classes — backed by Grad-CAM explainability
        and a detailed clinical PDF report.
      </p>
      <div class="pv-hero-chips">
        <span class="pv-chip">DenseNet-121 Classifier</span>
        <span class="pv-chip">U-Net Lung Segmentation</span>
        <span class="pv-chip">Grad-CAM XAI</span>
        <span class="pv-chip">CLAHE Enhancement</span>
        <span class="pv-chip">PDF Report Export</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── METRICS ──
    st.markdown("""
    <div class="pv-metrics">
      <div class="pv-metric"><div class="pv-metric-val">75.96%</div><div class="pv-metric-lbl">Test Accuracy</div></div>
      <div class="pv-metric"><div class="pv-metric-val">99.2%</div><div class="pv-metric-lbl">Disease Sensitivity</div></div>
      <div class="pv-metric"><div class="pv-metric-val">98.16%</div><div class="pv-metric-lbl">Normal Precision</div></div>
      <div class="pv-metric"><div class="pv-metric-val">3 Classes</div><div class="pv-metric-lbl">Diagnostic Scope</div></div>
      <div class="pv-metric"><div class="pv-metric-val">384×384</div><div class="pv-metric-lbl">Input Resolution</div></div>
    </div>""", unsafe_allow_html=True)

    # ── CONTENT WRAPPER ──
    st.markdown("<div class='pv-wrap'>", unsafe_allow_html=True)

    # ── DISEASE INFORMATION ──
    st.markdown("""
    <div class="pv-section-eyebrow">Diagnostic Reference</div>
    <div class="pv-section-title">What We Detect</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    cols = [c1, c2, c3]
    border_colors = ["#0D9488", "#DC2626", "#D97706"]

    for col, (cls_name, info), bc in zip(cols, DISEASE_INFO.items(), border_colors):
        with col:
            markers_html = "".join(
                f'<span style="background:#F1F5F9; color:#475569; font-size:0.74rem; font-weight:600; padding:3px 9px; border-radius:12px; border:1px solid #E2E8F0;">{m}</span>'
                for m in info["key_markers"]
            )
            st.markdown(f"""
            <div class="pv-info-card" style="border-top: 3px solid {bc}; padding: 1.3rem;">
              <div class="pv-info-card-title" style="color:{bc}; font-size: 1.05rem; font-weight: 700; margin-bottom: 0.4rem;">
                {info['icon']}&nbsp; {cls_name}
              </div>
              <div style="font-size:0.85rem; color:#475569; line-height:1.5; margin-bottom:0.9rem; min-height:2.5rem;">
                {info['tagline']}
              </div>
              <div style="display:flex; flex-wrap:wrap; gap:6px;">
                {markers_html}
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='pv-divider'>", unsafe_allow_html=True)

    # ── UPLOAD ──
    st.markdown("""
    <div class="pv-section-eyebrow">Step 1 — Image Ingestion</div>
    <div class="pv-section-title">Upload Chest Radiograph</div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag and drop a posterior-anterior (PA) chest X-ray — JPEG or PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

    if uploaded is None:
        st.markdown("""
        <div style="background:#F8FAFC; border:1.5px dashed #CBD5E1; border-radius:10px;
                    padding:2.2rem; text-align:center; color:#94A3B8; font-size:0.88rem; margin-top:0.5rem;">
          No file selected. Accepted formats: <strong style="color:#64748B;">JPEG, PNG</strong>.
          Ensure the image is a standard PA chest radiograph for accurate results.
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        _footer(dev)
        return

    # ── FILE PREVIEW ──
    pil_img = Image.open(uploaded)
    img_raw = np.array(pil_img.convert("L"))
    h_in, w_in = img_raw.shape[:2]

    st.markdown("<hr class='pv-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="pv-section-eyebrow">Step 2 — Review & Confirm</div>
    <div class="pv-section-title">Radiograph Preview</div>
    """, unsafe_allow_html=True)

    ci, cm_ = st.columns([1, 2], gap="large")
    with ci:
        st.image(img_raw, use_container_width=True, clamp=True, caption="Uploaded radiograph")
    with cm_:
        st.markdown(f"""
        <table class="pv-meta-table">
          <tr><td>Filename</td><td><code style="background:#F1F5F9;padding:2px 7px;border-radius:4px;font-size:0.81rem;">{uploaded.name}</code></td></tr>
          <tr><td>Dimensions</td><td>{w_in} × {h_in} px</td></tr>
          <tr><td>File Size</td><td>{uploaded.size/1024:.1f} KB</td></tr>
          <tr><td>Format</td><td>{uploaded.type.upper()}</td></tr>
          <tr><td>Pipeline</td><td>CLAHE → U-Net ROI → DenseNet-121 → Grad-CAM</td></tr>
          <tr><td>Status</td><td><span style="color:#0D9488;font-weight:600;">✓ Ready for analysis</span></td></tr>
        </table>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:1.2rem;'>", unsafe_allow_html=True)
        run = st.button("Run Diagnostic Analysis →")
        st.markdown("</div>", unsafe_allow_html=True)

    if not run:
        st.markdown("</div>", unsafe_allow_html=True)
        _footer(dev)
        return

    # ── PIPELINE ──
    st.markdown("<hr class='pv-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="pv-section-eyebrow">Step 3 — AI Pipeline Execution</div>
    <div class="pv-section-title">Diagnostic Results</div>
    """, unsafe_allow_html=True)

    with st.spinner("Executing 4-stage diagnostic pipeline…"):
        t0 = time.time()
        img_clahe = clahe_enhance(img_raw)
        img_roi, mask = extract_roi(img_clahe, unet, dev)

        SZ = (384, 384)
        raw_r   = cv2.resize(img_raw,   SZ, interpolation=cv2.INTER_CUBIC)
        clahe_r = cv2.resize(img_clahe, SZ, interpolation=cv2.INTER_CUBIC)
        roi_r   = cv2.resize(img_roi,   SZ, interpolation=cv2.INTER_CUBIC)

        img_t = torch.from_numpy(np.stack([roi_r]*3, -1)).float().permute(2,0,1)/255.0
        mu = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        sd = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        inp = ((img_t - mu)/sd).unsqueeze(0).to(dev)

        with torch.no_grad():
            out   = clf(inp)
            probs = F.softmax(out, 1).cpu().numpy()[0]
            pid   = int(np.argmax(probs))
            plbl  = CLASS_NAMES[pid]
            pconf = probs[pid]

        ms = (time.time()-t0)*1000

        heatmap     = gcam.generate_heatmap(inp, pid)
        overlay_bgr = gcam.overlay_heatmap(roi_r, heatmap, 0.45)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # ── 4-PANEL GRID ──
    p1, p2, p3, p4 = st.columns(4, gap="small")
    for col, label, sub, img_arr in [
        (p1, "① Original CXR",      "Raw input",                  raw_r),
        (p2, "② CLAHE Enhanced",    "Adaptive contrast eq.",       clahe_r),
        (p3, "③ U-Net Lung ROI",    "Segmented parenchyma",        roi_r),
        (p4, f"④ Grad-CAM · {plbl}", "Saliency attention map",     overlay_rgb),
    ]:
        with col:
            st.markdown(f"<div class='pv-panel-step'>{label}</div>", unsafe_allow_html=True)
            st.image(img_arr, use_container_width=True, clamp=True)
            st.markdown(f"<div class='pv-panel-cap'>{sub}</div>", unsafe_allow_html=True)

    st.markdown("<hr class='pv-divider'>", unsafe_allow_html=True)

    # ── DIAGNOSIS CARD + PROBABILITIES ──
    dcol, pcol = st.columns([1,1], gap="large")
    pcolor = CLASS_COLORS[plbl]

    with dcol:
        st.markdown("""<div class="pv-section-eyebrow">Primary Diagnosis</div>""", unsafe_allow_html=True)
        note_cls = {"Normal":"pv-cn-teal","Bacterial Pneumonia":"pv-cn-red","Viral Pneumonia":"pv-cn-amber"}[plbl]
        reco = DISEASE_INFO[plbl]["recommendation"]
        st.markdown(f"""
        <div class="pv-dx-card" style="border-top-color:{pcolor};">
          <div class="pv-dx-label" style="color:{pcolor};">Predicted Condition</div>
          <div class="pv-dx-name" style="color:{pcolor};">{plbl}</div>
          <div class="pv-dx-conf">
            Confidence &nbsp;<strong style="color:#0F172A;">{pconf*100:.2f}%</strong>
            &nbsp;·&nbsp; Latency &nbsp;<strong style="color:#0F172A;">{ms:.0f} ms</strong>
          </div>
          <div class="pv-clinical-note {note_cls}">
            <strong>Clinical Recommendation:</strong><br>{reco}
          </div>
        </div>""", unsafe_allow_html=True)

    with pcol:
        st.markdown("""<div class="pv-section-eyebrow">Confidence Breakdown</div>""", unsafe_allow_html=True)
        bars = "".join(prob_bar(c, probs[i], CLASS_COLORS[c], bold=(i==pid)) for i, c in enumerate(CLASS_NAMES))
        st.markdown(f"<div style='margin-top:0.3rem;'>{bars}</div>", unsafe_allow_html=True)

        fig = go.Figure(go.Bar(
            x=[p*100 for p in probs], y=CLASS_NAMES, orientation='h',
            marker=dict(color=[CLASS_COLORS[c] for c in CLASS_NAMES], line=dict(width=0)),
            text=[f"{p*100:.1f}%" for p in probs], textposition='outside',
            textfont=dict(size=11, color="#374151"),
        ))
        fig.update_layout(
            xaxis=dict(range=[0,118], showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False),
            height=145, bargap=0.45,
            margin=dict(l=0,r=0,t=6,b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter, sans-serif', size=12, color='#374151'),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<hr class='pv-divider'>", unsafe_allow_html=True)

    # ── DISEASE DETAIL PANEL (focused on predicted) ──
    st.markdown("""<div class="pv-section-eyebrow">Condition Overview</div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="pv-section-title">{plbl} — Radiological Profile</div>""", unsafe_allow_html=True)

    info = DISEASE_INFO[plbl]
    d1, d2 = st.columns([3,2], gap="large")
    with d1:
        st.markdown(f"""
        <div style="font-size:0.92rem;color:#334155;line-height:1.75;">{info['description']}</div>
        """, unsafe_allow_html=True)
    with d2:
        findings_html = "".join(f"<li>{f}</li>" for f in info["findings"])
        st.markdown(f"""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:1.2rem;">
          <div style="font-size:0.72rem;font-weight:700;color:#94A3B8;text-transform:uppercase;
                      letter-spacing:0.8px;margin-bottom:0.6rem;">Radiological Findings</div>
          <ul class="pv-finding-list">{findings_html}</ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='pv-divider'>", unsafe_allow_html=True)

    # ── PDF EXPORT ──
    st.markdown("""<div class="pv-section-eyebrow">Export</div>""", unsafe_allow_html=True)
    st.markdown("""<div class="pv-section-title">Download Clinical Report</div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.88rem;color:#475569;margin-bottom:1rem;line-height:1.65;">
      The PDF report includes: primary diagnosis, class probability table, all 4 pipeline images,
      disease radiological profile, model architecture details, and a clinical disclaimer.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Generating PDF report…"):
        pdf_bytes, pdf_err = generate_pdf_report(
            filename=uploaded.name, w_in=w_in, h_in=h_in,
            filesize_kb=uploaded.size/1024, device_str=dev.type,
            pred_label=plbl, pred_conf=pconf, probs=probs, inference_ms=ms,
            img_raw=raw_r, img_clahe=clahe_r, img_roi=roi_r, overlay_rgb=overlay_rgb,
        )

    if pdf_bytes and not pdf_err:
        rc1, rc2 = st.columns([1,3])
        with rc1:
            st.download_button(
                label="↓  Download PDF Report",
                data=pdf_bytes,
                file_name=f"PulmoVision_Report_{uploaded.name.rsplit('.',1)[0]}_{int(time.time())}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with rc2:
            st.markdown("""
            <div style="padding:0.65rem 0; font-size:0.82rem; color:#94A3B8;">
              Includes: diagnosis, probabilities, imaging panels, disease profile, model specs, disclaimer.
            </div>""", unsafe_allow_html=True)
    else:
        if pdf_err:
            st.error(f"PDF generation encountered an issue:\n```\n{pdf_err}\n```")
        else:
            st.warning("PDF generation unavailable — `reportlab` may not be installed.")

        # Fallback TXT
        txt = f"""PULMOVISION AI — CLINICAL DIAGNOSTIC REPORT
============================================
File       : {uploaded.name}  |  {w_in}×{h_in}px  |  {uploaded.size/1024:.1f} KB
Timestamp  : {time.strftime("%Y-%m-%d %H:%M:%S")}
Device     : {dev.type.upper()}  |  Latency: {ms:.1f} ms

DIAGNOSIS       : {plbl}
CONFIDENCE      : {pconf*100:.2f}%

CLASS PROBABILITIES:
  Normal              : {probs[0]*100:6.2f}%
  Bacterial Pneumonia : {probs[1]*100:6.2f}%
  Viral Pneumonia     : {probs[2]*100:6.2f}%

CLINICAL RECOMMENDATION:
  {DISEASE_INFO[plbl]['recommendation']}

PIPELINE: CLAHE → U-Net Segmentation → DenseNet-121 → Grad-CAM (denseblock4)

DISCLAIMER: For research/educational use only. Not a substitute for professional medical diagnosis.
"""
        st.download_button("↓ Download TXT Report", txt,
                           file_name=f"pulmovision_report_{int(time.time())}.txt", mime="text/plain")

    st.markdown("</div>", unsafe_allow_html=True)  # close pv-wrap
    _footer(dev)


def _footer(dev):
    st.markdown(f"""
    <div class="pv-footer">
      <strong style="color:#94A3B8;">PulmoVision AI</strong> &nbsp;·&nbsp;
      DenseNet-121 + U-Net Segmentation + Grad-CAM XAI &nbsp;·&nbsp;
      Compute: <strong style="color:#38BDF8;">{dev.type.upper()}</strong><br>
      <span style="color:#334155;">
        ⚠ For research and educational use only.
        Not a substitute for professional medical diagnosis by a licensed radiologist.
      </span>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

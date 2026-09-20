"""
Python script to generate publication-grade photos of algorithms and mathematical formulas,
and assemble a comprehensive Word document (.docx) for Agenda Point 6 of the 1st Project Review PPT.

Author: Senior Medical AI Engineer & Technical Presenter
"""

import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

# ==============================================================================
# 1. PHOTO GENERATION ENGINE (Matplotlib 300 DPI - Equations & Pseudocode)
# ==============================================================================

def render_pseudocode_photo(title: str, code_text: str, output_path: Path):
    lines = [line for line in code_text.strip().split('\n')]
    num_lines = len(lines)
    
    # Calculate figure height dynamically
    fig_height = max(3.8, num_lines * 0.24 + 0.8)
    fig, ax = plt.subplots(figsize=(9.5, fig_height), dpi=300)
    ax.axis('off')
    
    # Dark IDE background style
    bg_color = '#1A202C' # Dark Slate
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Top Header Bar
    ax.text(0.03, 0.96, f"⚡ {title}", ha='left', va='top', fontsize=10.5,
            fontweight='bold', color='#63B3ED', transform=ax.transAxes, fontfamily='monospace')
    ax.plot([0.03, 0.97], [0.92, 0.92], color='#4A5568', linewidth=1.2, transform=ax.transAxes)
    
    # Render line-by-line pseudocode with syntax highlighting
    y_start = 0.88
    line_spacing = 0.84 / max(num_lines, 1)
    
    for idx, line in enumerate(lines):
        y_pos = y_start - (idx * line_spacing)
        line_str = line
        
        # Color coding logic
        color = '#E2E8F0' # Light Gray Default
        fontweight = 'normal'
        
        if line_str.startswith("ALGORITHM") or line_str.startswith("INPUT") or line_str.startswith("OUTPUT") or line_str.startswith("==="):
            color = '#ECC94B' # Soft Gold
            fontweight = 'bold'
        elif any(line_str.strip().startswith(kw) for kw in ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."]):
            color = '#63B3ED' # Soft Blue
            fontweight = 'bold'
        elif any(kw in line_str for kw in ["FOR EACH", "DO:", "IF", "THEN", "RETURN"]):
            color = '#F6AD55' # Orange Keyword
            fontweight = 'bold'
        elif any(line_str.strip().startswith(prefix) for prefix in ["a.", "b.", "c.", "d.", "e.", "f.", "g."]):
            color = '#68D391' # Soft Green Sub-step
            
        ax.text(0.04, y_pos, line_str, ha='left', va='top', fontsize=8.5,
                color=color, fontweight=fontweight, transform=ax.transAxes, fontfamily='monospace')
                
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
    plt.close()
    print(f"[PHOTO OK] Rendered Pseudocode Image: {output_path.name}")


def render_equation_photo(title: str, latex_expressions: list, output_path: Path):
    num_eqs = len(latex_expressions)
    fig_height = max(1.8, num_eqs * 0.7 + 0.6)
    fig, ax = plt.subplots(figsize=(9.0, fig_height), dpi=300)
    ax.axis('off')
    
    bg_color = '#F7FAFC' # Light Soft Gray
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Border & Header
    rect = plt.Rectangle((0.01, 0.01), 0.98, 0.98, fill=False, edgecolor='#007791', linewidth=1.8, transform=ax.transAxes)
    ax.add_patch(rect)
    
    ax.text(0.04, 0.88, f"FORMULA CARD: {title}", ha='left', va='top', fontsize=10,
            fontweight='bold', color='#0A2540', transform=ax.transAxes, fontfamily='sans-serif')
            
    # Draw LaTeX formulas
    y_start = 0.65 if num_eqs == 1 else 0.68
    y_spacing = 0.45 if num_eqs > 1 else 0
    
    for idx, eq in enumerate(latex_expressions):
        y_pos = y_start - (idx * y_spacing)
        ax.text(0.5, y_pos, f"${eq}$", ha='center', va='center', fontsize=12,
                color='#1A202C', transform=ax.transAxes)
                
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    plt.close()
    print(f"[PHOTO OK] Rendered Formula Image: {output_path.name}")


def generate_all_photos(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # Pseudocode Photos
    # --------------------------------------------------------------------------
    algo1_text = (
        "========================================================================================\n"
        "ALGORITHM 1: CLAHE Local Contrast Standardization\n"
        "========================================================================================\n"
        "INPUT : Raw Gray-Scale Radiograph I(x, y), Tile Grid Size (Nx, Ny) = (8, 8), Clip Limit beta = 2.0\n"
        "OUTPUT: Contrast-Enhanced Radiograph I_CLAHE(x, y)\n\n"
        "1. Divide radiograph I into non-overlapping contextual tile grids T_{i,j} of dimension Nx x Ny.\n"
        "2. FOR EACH tile grid T_{i,j} DO:\n"
        "     a. Compute 256-bin local gray-level histogram H(k) for k in [0, 255].\n"
        "     b. Compute clip limit threshold: N_clip = beta * (Nx * Ny) / 256\n"
        "     c. Calculate total excess pixels: Delta_N = SUM_{k=0}^{255} max(H(k) - N_clip, 0)\n"
        "     d. Clip histogram H(k) = min(H(k), N_clip).\n"
        "     e. Redistribute excess pixels uniformly: H_redist(k) = H(k) + (Delta_N / 256)\n"
        "     f. Compute local Cumulative Distribution Function (CDF) for tile T_{i,j}.\n"
        "3. FOR EACH pixel location (x, y) DO:\n"
        "     a. Identify surrounding 4 tile center nodes (T11, T12, T21, T22).\n"
        "     b. Interpolate pixel value using Bilinear CDF Interpolation:\n"
        "            S(x, y) = (1-u)(1-v)*CDF_11 + u(1-v)*CDF_21 + (1-u)v*CDF_12 + uv*CDF_22\n"
        "4. RETURN Enhanced Image I_CLAHE.\n"
        "========================================================================================"
    )
    render_pseudocode_photo("Algorithm 1: Contextual CLAHE Contrast Engine", algo1_text, output_dir / "photo_algo1_clahe.png")
    
    algo2_text = (
        "========================================================================================\n"
        "ALGORITHM 2: U-Net Semantic Lung Field Extraction & ROI Isolation\n"
        "========================================================================================\n"
        "INPUT : Enhanced Image I_CLAHE(x, y), Trained Checkpoint weights/lung_unet_segmentation.pth\n"
        "OUTPUT: Anatomically Isolated Lung Region-of-Interest I_ROI(x, y)\n\n"
        "1. Preprocess I_CLAHE: Resize to (256, 256), normalize pixel intensities to [0, 1].\n"
        "2. Pass tensor X into U-Net Contracting Encoder:\n"
        "     a. Stage 1..4: Apply DoubleConv(3x3) -> BatchNorm -> ReLU -> MaxPool(2x2).\n"
        "     b. Save intermediate feature maps as Skip Connections [F_1, F_2, F_3, F_4].\n"
        "3. Pass deep features through Bottleneck (16x16x512 -> 16x16x1024).\n"
        "4. Pass through Expanding Decoder:\n"
        "     a. Stage 4..1: Upsample via ConvTranspose2d(2x2).\n"
        "     b. Concatenate upsampled features with corresponding Skip Connection F_i.\n"
        "     c. Apply DoubleConv(3x3) -> BatchNorm -> ReLU.\n"
        "5. Compute raw output logits Z and pass through Sigmoid: P_mask = Sigmoid(Z).\n"
        "6. Threshold probability map: Binary Mask M_lung = (P_mask > 0.5) ? 1 : 0.\n"
        "7. Perform ROI Isolation via element-wise Hadamard product:\n"
        "       I_ROI(x, y) = I_CLAHE(x, y) (x) M_lung(x, y)\n"
        "8. RETURN Anatomically Isolated ROI Image I_ROI.\n"
        "========================================================================================"
    )
    render_pseudocode_photo("Algorithm 2: U-Net Lung Field Segmentation & ROI Isolation", algo2_text, output_dir / "photo_algo2_unet.png")

    algo3_text = (
        "========================================================================================\n"
        "ALGORITHM 3: Class-Weighted ResNet50 Training with MixUp Augmentation\n"
        "========================================================================================\n"
        "INPUT : Isolated Lung ROIs {I_ROI}, Ground-Truth Labels Y in {0, 1, 2}, Class Frequencies N_c\n"
        "OUTPUT: Trained ResNet50 Model Weights W_ResNet\n\n"
        "1. Compute Inverse Class Frequency Weights w_c for class c in {0, 1, 2}:\n"
        "       w_c = N_total / (K * N_c)   where K = 3, N_total = 5856\n"
        "2. FOR EACH training batch (X_i, Y_i) and (X_j, Y_j) DO:\n"
        "     a. Sample mixing ratio lambda from Beta distribution: lambda ~ Beta(0.2, 0.2).\n"
        "     b. Create MixUp composite input image: X_mixed = lambda * X_i + (1 - lambda) * X_j\n"
        "     c. Create MixUp target label vector: Y_mixed = lambda * Y_i + (1 - lambda) * Y_j\n"
        "     d. Forward pass composite image X_mixed through ResNet50 backbone to get logits Z.\n"
        "     e. Compute Softmax class probabilities: P(c) = Softmax(Z).\n"
        "     f. Compute Class-Weighted Cross-Entropy Loss:\n"
        "            Loss_CE = - SUM_{c=0}^{2} w_c * Y_mixed(c) * log(P(c))\n"
        "     g. Backpropagate gradients and update weights using AdamW optimizer (lr = 1e-4).\n"
        "3. RETURN Optimized Model Weights W_ResNet.\n"
        "========================================================================================"
    )
    render_pseudocode_photo("Algorithm 3: ResNet50 Multi-Class Training & MixUp", algo3_text, output_dir / "photo_algo3_resnet.png")

    algo4_text = (
        "========================================================================================\n"
        "ALGORITHM 4: Gradient-Weighted Class Activation Mapping (Grad-CAM)\n"
        "========================================================================================\n"
        "INPUT : Query Image I_ROI, Target Class c in {0, 1, 2}, Trained ResNet50 Model, Layer A\n"
        "OUTPUT: Explainable Diagnostic Heatmap Overlay H_overlay(x, y)\n\n"
        "1. Perform forward pass of I_ROI through ResNet50 up to final layer to obtain score Y_c.\n"
        "2. Extract activation feature maps A^k from Layer 4 bottleneck (dimensions: u x v x k).\n"
        "3. Compute backpropagated gradients of target score Y_c with respect to feature maps A^k:\n"
        "       d(Y_c) / d(A_{i,j}^k)\n"
        "4. Calculate Global Average Pooling (GAP) neuron importance weights alpha_k^c:\n"
        "       alpha_k^c = (1 / (u * v)) * SUM_{i=1}^{u} SUM_{j=1}^{v} ( d(Y_c) / d(A_{i,j}^k) )\n"
        "5. Compute weighted linear combination of feature maps:\n"
        "       L_GradCAM^c = ReLU( SUM_{k} alpha_k^c * A^k )\n"
        "6. Apply Rectified Linear Unit (ReLU) to isolate positively contributing features.\n"
        "7. Resize L_GradCAM^c to match original image resolution (256, 256) via bilinear interpolation.\n"
        "8. Apply Jet colormap and blend with original radiograph:\n"
        "       H_overlay = 0.5 * I_ROI + 0.5 * Jet_ColorMap(L_GradCAM^c)\n"
        "9. RETURN Heatmap Overlay H_overlay.\n"
        "========================================================================================"
    )
    render_pseudocode_photo("Algorithm 4: Grad-CAM Explainable AI Saliency Mapping", algo4_text, output_dir / "photo_algo4_gradcam.png")

    # --------------------------------------------------------------------------
    # Formula Photos
    # --------------------------------------------------------------------------
    eq1 = [
        r"N_{\text{clip}} = \beta \cdot \frac{N_x \times N_y}{L} \quad (\text{where } \beta=2.0, \, L=256)",
        r"S(x, y) = (1-u)(1-v)S_{11} + u(1-v)S_{21} + (1-u)vS_{12} + uvS_{22}"
    ]
    render_equation_photo("Algorithm 1: CLAHE Clipping & Bilinear Interpolation Math", eq1, output_dir / "photo_eq1_clahe.png")

    eq2 = [
        r"\mathcal{L}_{\text{DiceBCE}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}",
        r"\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i} p_i g_i + \epsilon}{\sum_{i} p_i + \sum_{i} g_i + \epsilon} \quad (\epsilon = 10^{-6})"
    ]
    render_equation_photo("Algorithm 2: U-Net Combined DiceBCELoss & Metric Formulation", eq2, output_dir / "photo_eq2_unet.png")

    eq3 = [
        r"w_c = \frac{N_{\text{total}}}{K \cdot N_c} \quad (w_{\text{Normal}}=1.23, \, w_{\text{Bacter}}=0.70, \, w_{\text{Viral}}=1.31)",
        r"\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j \quad (\lambda \sim \text{Beta}(0.2, 0.2))"
    ]
    render_equation_photo("Algorithm 3: Inverse Class Weighting & MixUp Formulation", eq3, output_dir / "photo_eq3_resnet.png")

    eq4 = [
        r"\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{u} \sum_{j=1}^{v} \frac{\partial Y^c}{\partial A_{i,j}^k}",
        r"L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)"
    ]
    render_equation_photo("Algorithm 4: Grad-CAM GAP Neuron Weights & Heatmap Activation", eq4, output_dir / "photo_eq4_gradcam.png")

# ==============================================================================
# 2. WORD DOCUMENT BUILDER ENGINE (python-docx)
# ==============================================================================

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=140, bottom=140, left=200, right=200):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_callout_borders(cell, border_color="007791"):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/>
            <w:left w:val="single" w:sz="36" w:space="0" w:color="{border_color}"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    tcPr.append(borders)

def add_slide_header_card(doc, slide_title: str, sub_title: str):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "0A2540") # Deep Navy
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    
    r1 = p.add_run(f"SLIDE PRESENTATION CARD | AGENDA POINT 6\n")
    r1.font.name = "Arial"
    r1.font.size = Pt(9.0)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(0, 119, 145) # Slate Teal
    
    r2 = p.add_run(f"{slide_title.upper()}\n")
    r2.font.name = "Arial"
    r2.font.size = Pt(14)
    r2.font.bold = True
    r2.font.color.rgb = RGBColor(255, 255, 255)
    
    r3 = p.add_run(sub_title)
    r3.font.name = "Arial"
    r3.font.size = Pt(10)
    r3.font.italic = True
    r3.font.color.rgb = RGBColor(203, 213, 224)
    
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(8)

def add_body_bullet(doc, bold_prefix: str, text: str):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    
    r_pre = p.add_run(bold_prefix)
    r_pre.font.name = "Arial"
    r_pre.font.size = Pt(10.5)
    r_pre.font.bold = True
    r_pre.font.color.rgb = RGBColor(10, 37, 64)
    
    r_t = p.add_run(text)
    r_t.font.name = "Arial"
    r_t.font.size = Pt(10.0)
    r_t.font.color.rgb = RGBColor(45, 55, 72)

def add_speaker_notes(doc, text: str):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "FFFBEB") # Light Warm Amber
    set_callout_borders(cell, "D69E2E") # Amber Border
    set_cell_margins(cell, top=100, bottom=100, left=160, right=160)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    
    r_pre = p.add_run("🎙️ SPEAKER NOTES FOR PRESENTATION: ")
    r_pre.font.name = "Arial"
    r_pre.font.size = Pt(9.5)
    r_pre.font.bold = True
    r_pre.font.color.rgb = RGBColor(180, 83, 9)
    
    r_t = p.add_run(text)
    r_t.font.name = "Arial"
    r_t.font.size = Pt(9.5)
    r_t.font.italic = True
    r_t.font.color.rgb = RGBColor(45, 55, 72)
    
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(10)

def embed_photo(doc, photo_path: Path, caption: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    
    r = p.add_run()
    r.add_picture(str(photo_path), width=Inches(6.2))
    
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(10)
    r_cap = p_cap.add_run(caption)
    r_cap.font.name = "Arial"
    r_cap.font.size = Pt(8.5)
    r_cap.font.italic = True
    r_cap.font.color.rgb = RGBColor(113, 128, 150)

# ==============================================================================
# MAIN COMPILATION ENGINE
# ==============================================================================

def build_word_document(workspace_dir: Path):
    doc = Document()
    
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Document Header
    p_top = doc.add_paragraph()
    p_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_t1 = p_top.add_run("THE OXFORD COLLEGE OF ENGINEERING (TOCE)\n")
    r_t1.font.name = "Arial"
    r_t1.font.size = Pt(14)
    r_t1.font.bold = True
    r_t1.font.color.rgb = RGBColor(10, 37, 64)
    
    r_t2 = p_top.add_run("DEPARTMENT OF INFORMATION SCIENCE & ENGINEERING\n")
    r_t2.font.name = "Arial"
    r_t2.font.size = Pt(11)
    r_t2.font.bold = True
    r_t2.font.color.rgb = RGBColor(0, 119, 145)
    
    r_t3 = p_top.add_run("1st Project Review PPT Content — Agenda Point 6: Modules, Algorithms & Formulations\n")
    r_t3.font.name = "Arial"
    r_t3.font.size = Pt(12)
    r_t3.font.bold = True
    r_t3.font.color.rgb = RGBColor(197, 48, 48)
    
    p_div = doc.add_paragraph()
    p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_d = p_div.add_run("―" * 48)
    r_d.font.color.rgb = RGBColor(0, 119, 145)
    r_d.font.bold = True
    
    # --------------------------------------------------------------------------
    # SECTION 1: SYSTEM MODULES BREAKDOWN
    # --------------------------------------------------------------------------
    add_slide_header_card(doc, "SLIDE 1: System Modules Breakdown Overview", "Agenda Point 6.1 — Core Architectural Modules")
    
    add_body_bullet(doc, "Module 1: Dataset Ingestion & 3-Class Parser (src/data/dataset_parser.py) — ", 
                    "Parses raw Kaggle binary folders into Class 0 (Normal: 1,583), Class 1 (Bacterial: 2,780), and Class 2 (Viral: 1,493) across 5,856 radiographs.")
    add_body_bullet(doc, "Module 2: OpenCV CLAHE Contrast Engine (src/data/clahe.py) — ", 
                    "Executes 8x8 tile grid adaptive histogram equalization with clip limit beta=2.0 to standardize inter-equipment lighting variations.")
    add_body_bullet(doc, "Module 3: PyTorch Biomedical U-Net Lung Segmentation (notebooks/colab_lung_segmentation_unet.py) — ", 
                    "Isolates lung parenchyma ROI and eliminates extra-pulmonary shortcut artifacts (Dice: 0.956, IoU: 0.898).")
    add_body_bullet(doc, "Module 4: 3-Class ResNet50 Deep Classifier (src/models/resnet_classifier.py) — ", 
                    "Deep residual network trained with inverse class weighting and Albumentations MixUp/CutMix regularization.")
    add_body_bullet(doc, "Module 5: Grad-CAM Explainable AI & Streamlit Web UI (src/utils/gradcam.py & app.py) — ", 
                    "Generates Layer 4 gradient heatmap overlays and presents an interactive clinical triage dashboard.")

    add_speaker_notes(doc, "Emphasize to the review panel that Modules 1, 2, and 3 are 100% completed (~60% project milestone), while Modules 4 and 5 represent the remaining implementation phase.")
    
    # --------------------------------------------------------------------------
    # ALGORITHM 1
    # --------------------------------------------------------------------------
    doc.add_page_break()
    add_slide_header_card(doc, "SLIDE 2: Algorithm 1 — CLAHE Contrast Engine", "Contextual Local Contrast Standardization & Clipping")
    
    add_body_bullet(doc, "Objective: ", "Amplify subtle ground-glass opacities without over-saturating homogeneous chest regions.")
    add_body_bullet(doc, "Grid Processing: ", "Divides CXR images into 8x8 non-overlapping contextual tiles.")
    add_body_bullet(doc, "Mathematical Clipping: ", "Limits local histogram height at beta=2.0 and redistributes excess pixels uniformly.")

    embed_photo(doc, workspace_dir / "photo_algo1_clahe.png", "Photo 1.1: High-Resolution Pseudocode Photo — Algorithm 1 (CLAHE Enhancement Engine)")
    embed_photo(doc, workspace_dir / "photo_eq1_clahe.png", "Photo 1.2: Rendered Mathematical Formula Photo — CLAHE Clip Limit & Bilinear Interpolation")
    
    add_speaker_notes(doc, "Explain that global histogram equalization destroys CXR details. CLAHE clips tile histograms at beta=2.0, spreading excess pixel counts evenly so soft-tissue infiltrates become visible.")

    # --------------------------------------------------------------------------
    # ALGORITHM 2
    # --------------------------------------------------------------------------
    doc.add_page_break()
    add_slide_header_card(doc, "SLIDE 3: Algorithm 2 — U-Net Semantic Lung Segmentation", "Anatomical ROI Isolation & Black-Box Overfitting Prevention")
    
    add_body_bullet(doc, "Objective: ", "Extract precise lung parenchyma boundaries and discard peripheral extra-pulmonary shortcut artifacts.")
    add_body_bullet(doc, "Architecture: ", "Contracting Encoder + Bottleneck + Expanding Decoder with residual skip connections.")
    add_body_bullet(doc, "Loss Optimization: ", "Trained with combined DiceBCELoss to achieve Dice Coefficient = 0.956.")

    embed_photo(doc, workspace_dir / "photo_algo2_unet.png", "Photo 2.1: High-Resolution Pseudocode Photo — Algorithm 2 (U-Net Semantic Lung Field Extraction)")
    embed_photo(doc, workspace_dir / "photo_eq2_unet.png", "Photo 2.2: Rendered Mathematical Formula Photo — Combined DiceBCELoss & Dice Similarity Metric")

    add_speaker_notes(doc, "Highlight that skip connections directly pass high-resolution spatial feature maps from encoder to decoder, preventing loss of fine costal and apical lung margins.")

    # --------------------------------------------------------------------------
    # ALGORITHM 3
    # --------------------------------------------------------------------------
    doc.add_page_break()
    add_slide_header_card(doc, "SLIDE 4: Algorithm 3 — ResNet50 3-Class Classifier", "Transfer Learning, Class Weighting & MixUp Augmentation")
    
    add_body_bullet(doc, "Objective: ", "Classify isolated lung ROIs into Normal (Class 0), Bacterial (Class 1), or Viral (Class 2).")
    add_body_bullet(doc, "Imbalance Resolution: ", "Applies inverse class frequency weighting (w_Normal=1.23, w_Bacter=0.70, w_Viral=1.31).")
    add_body_bullet(doc, "Regularization: ", "Utilizes Albumentations MixUp (lambda ~ Beta(0.2, 0.2)) to force smooth decision boundaries.")

    embed_photo(doc, workspace_dir / "photo_algo3_resnet.png", "Photo 3.1: High-Resolution Pseudocode Photo — Algorithm 3 (ResNet50 Training & MixUp)")
    embed_photo(doc, workspace_dir / "photo_eq3_resnet.png", "Photo 3.2: Rendered Mathematical Formula Photo — Inverse Class Frequency Weighting & MixUp Math")

    add_speaker_notes(doc, "Note that Bacterial cases account for 47.5% of dataset images. Class weighting ensures the neural network treats Viral and Normal cases with equal diagnostic importance.")

    # --------------------------------------------------------------------------
    # ALGORITHM 4
    # --------------------------------------------------------------------------
    doc.add_page_break()
    add_slide_header_card(doc, "SLIDE 5: Algorithm 4 — Grad-CAM Explainability", "Gradient Localization & Clinician Visual Heatmaps")
    
    add_body_bullet(doc, "Objective: ", "Provide visual proof of diagnostic features for attending radiologists to establish AI transparency.")
    add_body_bullet(doc, "Gradient Pooling: ", "Computes Global Average Pooling (GAP) of backpropagated gradients over Layer 4 feature maps.")
    add_body_bullet(doc, "ReLU Filtering: ", "Applies Rectified Linear Unit activation to discard negative gradients and retain positive lesion evidence.")

    embed_photo(doc, workspace_dir / "photo_algo4_gradcam.png", "Photo 4.1: High-Resolution Pseudocode Photo — Algorithm 4 (Grad-CAM Saliency Map Generation)")
    embed_photo(doc, workspace_dir / "photo_eq4_gradcam.png", "Photo 4.2: Rendered Mathematical Formula Photo — Grad-CAM GAP Weights & Heatmap Activation")

    add_speaker_notes(doc, "Point out that applying ReLU in Grad-CAM is crucial because it isolates only those features that positively contribute to the target pneumonia class, filtering out irrelevant features.")

    # Save Word document
    output_docx_path = workspace_dir / "Pneumonia_Detection_PPT_Agenda6_Modules_Algorithms.docx"
    doc.save(str(output_docx_path))
    print(f"\n[SUCCESS] Saved slide report document with embedded photos at: {output_docx_path}")

# ==============================================================================
# SCRIPT EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    workspace_root = Path(__file__).parent.resolve()
    print(f"Generating formula photos, algorithm photos & Word doc in: {workspace_root}")
    
    generate_all_photos(workspace_root)
    build_word_document(workspace_root)

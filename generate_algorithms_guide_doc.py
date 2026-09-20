"""
Python script to generate a publication-grade, institutional Word document:
Pneumonia_Detection_Algorithms_and_Formulas_Guide.docx

Target: Complete technical explanation of all 4 algorithms, step-by-step execution,
every formula variable definition, clinical rationale, embedded high-res photos,
and presentation pointers / viva defense questions for the 1st Project Review.

Author: Senior Software Architect & Medical AI Specialist
Institution: The Oxford College of Engineering (TOCE), Dept. of ISE
"""

import os
import sys
from pathlib import Path

import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

# ==============================================================================
# WORD DOCUMENT STYLING HELPERS
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

def add_styled_heading(doc, text: str, level: int):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.bold = True
    
    if level == 1:
        p.paragraph_format.space_before = Pt(20)
        p.paragraph_format.space_after = Pt(8)
        r.font.size = Pt(16)
        r.font.color.rgb = RGBColor(10, 37, 64) # Deep Navy
    elif level == 2:
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        r.font.size = Pt(13)
        r.font.color.rgb = RGBColor(0, 119, 145) # Slate Teal
    elif level == 3:
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        r.font.size = Pt(11)
        r.font.color.rgb = RGBColor(74, 96, 122) # Muted Steel
    return p

def add_body_p(doc, text: str, bold_prefix: str = None, space_after: int = 6):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Arial"
        r_pre.font.size = Pt(10.0)
        r_pre.font.bold = True
        r_pre.font.color.rgb = RGBColor(10, 37, 64)
        
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(10.0)
    r.font.color.rgb = RGBColor(45, 55, 72)
    return p

def add_presentation_pointers(doc, title: str, pointers: list):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "FFFBEB") # Light Warm Amber
    set_callout_borders(cell, "D69E2E") # Amber Accent Border
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    
    r_hdr = p.add_run(f"🎙️ PRESENTATION POINTERS & VIVA DEFENSE GUIDE: {title.upper()}\n")
    r_hdr.font.name = "Arial"
    r_hdr.font.size = Pt(10.0)
    r_hdr.font.bold = True
    r_hdr.font.color.rgb = RGBColor(180, 83, 9)
    
    for idx, ptr in enumerate(pointers, 1):
        p_ptr = cell.add_paragraph()
        p_ptr.paragraph_format.space_before = Pt(2)
        p_ptr.paragraph_format.space_after = Pt(3)
        p_ptr.paragraph_format.line_spacing = 1.15
        
        r_num = p_ptr.add_run(f"• {ptr[0]}: ")
        r_num.font.name = "Arial"
        r_num.font.size = Pt(9.5)
        r_num.font.bold = True
        r_num.font.color.rgb = RGBColor(45, 55, 72)
        
        r_txt = p_ptr.add_run(ptr[1])
        r_txt.font.name = "Arial"
        r_txt.font.size = Pt(9.5)
        r_txt.font.color.rgb = RGBColor(45, 55, 72)

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(0)
    p_sp.paragraph_format.space_after = Pt(8)

def embed_photo(doc, photo_path: Path, caption: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    
    if photo_path.exists():
        r = p.add_run()
        r.add_picture(str(photo_path), width=Inches(6.2))
    else:
        r = p.add_run(f"[Image file not found: {photo_path.name}]")
        r.font.color.rgb = RGBColor(197, 48, 48)
        
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(10)
    r_cap = p_cap.add_run(caption)
    r_cap.font.name = "Arial"
    r_cap.font.size = Pt(8.5)
    r_cap.font.italic = True
    r_cap.font.color.rgb = RGBColor(113, 128, 150)

def style_table(table, col_widths, headers, rows_data):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    # Header Row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        hdr_cells[i].width = Inches(col_widths[i])
        set_cell_background(hdr_cells[i], "0A2540")
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=140, right=140)
        
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        for run in p.runs:
            run.font.name = "Arial"
            run.font.size = Pt(9.0)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
            
    # Data Rows
    for r_idx, row_data in enumerate(rows_data):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F7FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            row_cells[c_idx].width = Inches(col_widths[c_idx])
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=100, bottom=100, left=140, right=140)
            
            p = row_cells[c_idx].paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            for run in p.runs:
                run.font.name = "Arial"
                run.font.size = Pt(8.5)
                run.font.color.rgb = RGBColor(45, 55, 72)

# ==============================================================================
# MAIN DOCUMENT GENERATION BUILDER
# ==============================================================================

def generate_guide_document(workspace_dir: Path):
    doc = Document()
    photos_dir = workspace_dir / "generated_photos" / "algorithm_photos"
    
    # Standard 1 Inch Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # --------------------------------------------------------------------------
    # COVER / HEADER TITLE
    # --------------------------------------------------------------------------
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p_inst.add_run("THE OXFORD COLLEGE OF ENGINEERING (TOCE)\n")
    r1.font.name = "Arial"
    r1.font.size = Pt(15)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(10, 37, 64)
    
    r2 = p_inst.add_run("DEPARTMENT OF INFORMATION SCIENCE & ENGINEERING\n")
    r2.font.name = "Arial"
    r2.font.size = Pt(11)
    r2.font.bold = True
    r2.font.color.rgb = RGBColor(0, 119, 145)
    
    p_div = doc.add_paragraph()
    p_div.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_d = p_div.add_run("―" * 50)
    r_d.font.color.rgb = RGBColor(0, 119, 145)
    r_d.font.bold = True
    
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(10)
    p_title.paragraph_format.space_after = Pt(15)
    r_t = p_title.add_run("COMPREHENSIVE ALGORITHM & MATHEMATICAL FORMULA EXPLANATION GUIDE\n(1st Project Review Presentation & Viva Defense Handbook)")
    r_t.font.name = "Arial"
    r_t.font.size = Pt(14)
    r_t.font.bold = True
    r_t.font.color.rgb = RGBColor(197, 48, 48)
    
    add_body_p(doc, "Project Title: An Enhanced CNN-Based Multi-Class Pneumonia Detection System Using Advanced Image Enhancement, Segmentation, And Data Augmentation Techniques", bold_prefix="PROJECT CONTEXT: ")
    add_body_p(doc, "Student Authors: Adithya M (1OX23IS002), Bhavani Patil (1OX23IS011), Harsha R (1OX23IS025), K. Lalitha (1OX23IS029)", bold_prefix="TEAM BATCH: ")
    add_body_p(doc, "Project Guide: Mrs. S. Visalini, Assistant Professor, Dept. of ISE, TOCE", bold_prefix="FACULTY GUIDE: ")
    
    doc.add_page_break()

    # --------------------------------------------------------------------------
    # EXECUTIVE SUMMARY: SYSTEM ALGORITHMIC PIPELINE
    # --------------------------------------------------------------------------
    add_styled_heading(doc, "SECTION 1: SYSTEM ALGORITHMIC PIPELINE OVERVIEW", level=1)
    
    add_body_p(doc, 
               "To solve the critical clinical challenge of 'shortcut learning'—where deep learning models overfit to peripheral "
               "extra-pulmonary artifacts like rib cages, clavicles, radiographer text letters, and sensor exposure variations—our system "
               "implements an anatomically isolated multi-stage computer-assisted diagnostic framework governed by four primary algorithms:")
               
    pipeline_steps = [
        ["Algorithm 1: CLAHE Engine", "Contextual local contrast enhancement (8x8 tile grid, clip limit β=2.0) to unmask subtle ground-glass opacities and loabr consolidations."],
        ["Algorithm 2: U-Net Segmenter", "Biomedical semantic lung field extraction (DiceBCELoss) to produce binary lung masks M(x,y) and crop extra-pulmonary noise."],
        ["Algorithm 3: ResNet50 Classifier", "Multi-class 3-way triage (Normal vs. Bacterial vs. Viral) with inverse class weighting (w_c) and Albumentations MixUp/CutMix."],
        ["Algorithm 4: Grad-CAM Explainer", "Gradient-weighted class activation mapping to generate Layer 4 heatmap overlays for attending radiologists."]
    ]
    for step in pipeline_steps:
        add_body_p(doc, step[1], bold_prefix=f"• {step[0]}: ")

    # --------------------------------------------------------------------------
    # ALGORITHM 1: CLAHE CONTRAST Standardization
    # --------------------------------------------------------------------------
    doc.add_page_break()
    add_styled_heading(doc, "ALGORITHM 1: LOCALIZED CLAHE CONTRAST ENGINE", level=1)
    
    add_styled_heading(doc, "1.1 Clinical Rationale & Objective", level=2)
    add_body_p(doc, 
               "Standard Global Histogram Equalization (GHE) computes a single global CDF across the entire image. On chest radiographs, "
               "this causes catastrophic over-saturation in bright bone areas (sternum, clavicles) and accentuates background sensor noise. "
               "Contrast Limited Adaptive Histogram Equalization (CLAHE) operates on localized contextual sub-regions (tile grids, default 8x8). "
               "By clipping histogram heights at clip limit threshold β = 2.0, CLAHE caps local noise amplification while boosting micro-textures "
               "(reticulonodular infiltrates in viral cases and focal lobar opacities in bacterial cases).")

    add_styled_heading(doc, "1.2 Rendered Pseudocode & Formula Photos", level=2)
    embed_photo(doc, photos_dir / "photo_algo1_clahe.png", "Photo 1.1: High-Resolution Pseudocode — Algorithm 1 (Contextual CLAHE Contrast Standardization)")
    embed_photo(doc, photos_dir / "photo_eq1_clahe.png", "Photo 1.2: Rendered Mathematical Formulas — Clip Limit Threshold & Bilinear Interpolation")

    add_styled_heading(doc, "1.3 Exhaustive Variable & Mathematical Symbol Registry", level=2)
    
    clahe_vars = [
        ["I(x, y)", "Input Grayscale Image", "Original 8-bit single-channel chest radiograph pixel intensity at spatial coordinate (x, y), ranging from 0 to 255."],
        ["(N_x, N_y)", "Tile Grid Dimensions", "Contextual sub-region size into which the radiograph is divided. Default is (8, 8), creating 64 contextual tiles."],
        ["\\beta", "Clip Limit Parameter", "Normalized contrast threshold multiplier (default β = 2.0). Controls maximum allowable slope of the local CDF."],
        ["L", "Gray Level Count", "Total available intensity levels in grayscale space (L = 256 gray bins, from 0 to 255)."],
        ["H(k)", "Original Local Histogram", "Pixel count frequency for intensity bin k within tile grid T_{i,j} for k in [0, 255]."],
        ["N_{clip}", "Clip Limit Threshold", "Maximum height cap per bin: N_{clip} = \\beta \\cdot (N_x \\times N_y) / L. Bins exceeding N_{clip} are truncated."],
        ["\\Delta N", "Excess Pixel Sum", "Total sum of pixel counts exceeding N_{clip} across all bins: \\Delta N = \\sum_{k=0}^{L-1} \\max(H(k) - N_{clip}, 0)."],
        ["H_{redist}(k)", "Redistributed Histogram", "Uniformly redistributed bin count: H_{redist}(k) = H_{clip}(k) + (\\Delta N / L)."],
        ["CDF", "Cumulative Dist. Func.", "Normalized cumulative distribution function for tile T_{i,j}: CDF(k) = \\sum_{m=0}^k H_{redist}(m) / (N_x \\times N_y)."],
        ["(u, v)", "Normalized Offsets", "Fractional spatial coordinates of pixel (x, y) relative to surrounding 4 tile center nodes, u, v in [0, 1]."],
        ["S_{11}, S_{21}, S_{12}, S_{22}", "Tile Corner CDF Nodes", "CDF transformation values computed at top-left, top-right, bottom-left, and bottom-right tile centers."],
        ["S(x, y)", "Output Equalized Pixel", "Final bilinearly interpolated gray-level intensity value at coordinate (x, y)."]
    ]
    
    tbl_c_vars = doc.add_table(rows=len(clahe_vars)+1, cols=3)
    style_table(tbl_c_vars, [1.4, 1.6, 3.5], ["Variable / Symbol", "Role Name", "Detailed Mathematical & Technical Description"], clahe_vars)

    add_styled_heading(doc, "1.4 Step-by-Step Execution Mechanics", level=2)
    add_body_p(doc, "1. Tile Decomposition: The image I is partitioned into 64 non-overlapping 8x8 tile grids T_{i,j}.")
    add_body_p(doc, "2. Histogram Truncation: For each tile, a 256-bin histogram H(k) is calculated. Bins higher than N_{clip} are truncated.")
    add_body_p(doc, "3. Uniform Redistribution: Truncated excess pixels ΔN are distributed evenly across all 256 bins, flattening the histogram.")
    add_body_p(doc, "4. Bilinear Interpolation: To eliminate artificial boundaries between neighboring 8x8 tiles, pixel intensity S(x,y) is calculated using bilinear interpolation across the 4 nearest tile centers.")

    add_presentation_pointers(doc, "Algorithm 1 (CLAHE Contrast Engine)", [
        ["Why β=2.0?", "If clip limit β is set too high (>4.0), noise in lung tissue gets amplified like standard GHE. If set too low (1.0), no contrast enhancement occurs. β=2.0 is optimal for medical radiography."],
        ["Why Bilinear Interpolation?", "Without bilinear interpolation, tile boundaries create visible blocky grid artifacts. Bilinear interpolation ensures smooth spatial transitions."],
        ["Key Clinical Impact", "Unmasks subtle 'ground-glass' reticulonodular opacities characteristic of early viral pneumonia."]
    ])

    # --------------------------------------------------------------------------
    # ALGORITHM 2: U-NET SEMANTIC SEGMENTATION
    # --------------------------------------------------------------------------
    doc.add_page_break()
    add_styled_heading(doc, "ALGORITHM 2: U-NET LUNG SEGMENTATION & ROI ISOLATION", level=1)
    
    add_styled_heading(doc, "2.1 Clinical Rationale & Objective", level=2)
    add_body_p(doc, 
               "Standard CNN classifiers trained on raw radiographs suffer from severe shortcut learning: they learn to classify based on "
               "radiographer text labels ('L'/'R'), rib cage geometry, diaphragm height, or ECG lead wires. "
               "Algorithm 2 uses a PyTorch Biomedical U-Net semantic segmentation network trained on Google Colab (DiceBCELoss) to generate "
               "a binary lung parenchyma mask M_{lung}(x, y). Multiplying the mask by the enhanced image (Hadamard product) strips away all "
               "extra-pulmonary background noise, forcing downstream classifiers to inspect only genuine lung tissue.")

    add_styled_heading(doc, "2.2 Rendered Pseudocode & Formula Photos", level=2)
    embed_photo(doc, photos_dir / "photo_algo2_unet.png", "Photo 2.1: High-Resolution Pseudocode — Algorithm 2 (Biomedical U-Net Lung Field Extraction)")
    embed_photo(doc, photos_dir / "photo_eq2_unet.png", "Photo 2.2: Rendered Mathematical Formulas — Combined DiceBCELoss & Dice Metric Formulation")

    add_styled_heading(doc, "2.3 Exhaustive Variable & Mathematical Symbol Registry", level=2)
    
    unet_vars = [
        ["X", "Input Image Tensor", "Preprocessed CLAHE image rescaled to 256x256 resolution, normalized to tensor range [0.0, 1.0]."],
        ["DoubleConv(3x3)", "Double Conv Block", "Sequence of Conv2d(3x3) -> BatchNorm2d -> ReLU -> Conv2d(3x3) -> BatchNorm2d -> ReLU."],
        ["F_1, F_2, F_3, F_4", "Skip Connections", "High-resolution spatial feature maps preserved from encoder stages prior to 2x2 max-pooling."],
        ["Z", "Output Logits Tensor", "Raw unscaled output tensor of size (1, 256, 256) produced by final 1x1 convolution layer."],
        ["P_{mask}", "Probability Map", "Continuous pixel-wise lung probability map generated via Sigmoid: P_{mask} = Sigmoid(Z) in [0.0, 1.0]."],
        ["M_{lung}", "Binary Lung Mask", "Binarized spatial mask thresholded at 0.5: M_{lung}(x,y) = 1 if P_{mask}(x,y) > 0.5 else 0."],
        ["I_{ROI}(x,y)", "Isolated Lung ROI", "Anatomically isolated lung parenchyma image produced by Hadamard product: I_{ROI} = I_{CLAHE} \\odot M_{lung}."],
        ["\\odot \\text{ or } \\otimes", "Hadamard Product", "Element-wise matrix multiplication between CLAHE image tensor and binary mask tensor."],
        ["\\mathcal{L}_{DiceBCE}", "Combined Loss Function", "Total segmentation loss: \\mathcal{L}_{DiceBCE} = \\alpha \\mathcal{L}_{BCE} + (1-\\alpha) \\mathcal{L}_{Dice} with \\alpha=0.5."],
        ["\\mathcal{L}_{BCE}", "Binary Cross Entropy", "Pixel-wise classification loss: \\mathcal{L}_{BCE} = - (1/N) \\sum [ y_i \\log(p_i) + (1-y_i) \\log(1-p_i) ]."],
        ["\\mathcal{L}_{Dice}", "Dice Overlap Loss", "Spatial region overlap loss: \\mathcal{L}_{Dice} = 1 - (2 \\sum p_i g_i + \\epsilon) / (\\sum p_i + \\sum g_i + \\epsilon)."],
        ["p_i", "Predicted Probability", "Model predicted lung probability for pixel i (where p_i = Sigmoid(logits_i))."],
        ["g_i", "Ground-Truth Mask", "Binary ground-truth lung label for pixel i (1 = lung parenchyma, 0 = background)."],
        ["\\epsilon", "Smoothing Epsilon", "Small constant (10^{-6}) added to numerator and denominator to prevent division by zero."]
    ]
    
    tbl_u_vars = doc.add_table(rows=len(unet_vars)+1, cols=3)
    style_table(tbl_u_vars, [1.4, 1.6, 3.5], ["Variable / Symbol", "Role Name", "Detailed Mathematical & Technical Description"], unet_vars)

    add_styled_heading(doc, "2.4 Step-by-Step Execution Mechanics", level=2)
    add_body_p(doc, "1. Feature Contracting (Encoder): Input X passes through 4 encoder blocks (64 -> 128 -> 256 -> 512 channels), saving skip connections F_i.")
    add_body_p(doc, "2. Bottleneck Stage: Deep contextual features are processed through bottleneck double convolution (1024 channels).")
    add_body_p(doc, "3. Feature Expanding (Decoder): Transposed 2x2 convolutions upsample feature maps, concatenating skip connections F_i at each stage.")
    add_body_p(doc, "4. Binary Mask Generation & ROI Cropping: Sigmoid produces probability map P_{mask}. Thresholding at 0.5 creates binary mask M_{lung}. Element-wise multiplication produces isolated ROI I_{ROI}.")

    add_presentation_pointers(doc, "Algorithm 2 (U-Net Segmentation)", [
        ["Why Skip Connections?", "Max-pooling reduces spatial resolution. Skip connections copy high-resolution edge features from encoder to decoder, preserving fine apical and costal lung borders."],
        ["Why Combine BCE + Dice Loss?", "BCE measures pixel-wise classification accuracy but struggles with foreground/background imbalance. Dice loss measures region overlap. Combining both achieves high stability and a Dice score of 0.956."],
        ["Key Clinical Impact", "Eliminates shortcut learning artifacts (ribs, text markers, ECG leads) so downstream ResNet50 inspects only lung parenchyma."]
    ])

    # --------------------------------------------------------------------------
    # ALGORITHM 3: RESNET50 CLASSIFIER & MIXUP
    # --------------------------------------------------------------------------
    doc.add_page_break()
    add_styled_heading(doc, "ALGORITHM 3: RESNET50 CLASSIFIER & MIXUP AUGMENTATION", level=1)
    
    add_styled_heading(doc, "3.1 Clinical Rationale & Objective", level=2)
    add_body_p(doc, 
               "Differentiating Bacterial pneumonia (Class 1) from Viral pneumonia (Class 2) is vital for antimicrobial stewardship: "
               "Bacterial infections require immediate antibiotics, while Viral cases require supportive antiviral care and withholding antibiotics. "
               "However, medical datasets exhibit severe class imbalance (Kermany dataset: 47.5% Bacterial, 27.0% Normal, 25.5% Viral). "
               "Algorithm 3 trains a 50-layer deep Residual Network (ResNet50) using inverse class frequency weights w_c and Albumentations "
               "MixUp data augmentation to prevent model bias toward the dominant Bacterial class and enforce smooth decision boundaries.")

    add_styled_heading(doc, "3.2 Rendered Pseudocode & Formula Photos", level=2)
    embed_photo(doc, photos_dir / "photo_algo3_resnet.png", "Photo 3.1: High-Resolution Pseudocode — Algorithm 3 (Class-Weighted ResNet50 Classifier)")
    embed_photo(doc, photos_dir / "photo_eq3_resnet.png", "Photo 3.2: Rendered Mathematical Formulas — Inverse Class Weighting & MixUp Formulation")

    add_styled_heading(doc, "3.3 Exhaustive Variable & Mathematical Symbol Registry", level=2)
    
    resnet_vars = [
        ["K", "Class Count", "Total target diagnostic categories (K = 3: Class 0=Normal, Class 1=Bacterial, Class 2=Viral)."],
        ["N_{total}", "Total Sample Count", "Total radiographs in dataset (N_{total} = 5,856 images)."],
        ["N_c", "Class Frequency", "Number of training samples in class c (N_0 = 1583, N_1 = 2780, N_2 = 1493)."],
        ["w_c", "Inverse Class Weight", "Class balancing multiplier: w_c = N_{total} / (K \\cdot N_c) (w_0 = 1.23, w_1 = 0.70, w_2 = 1.31)."],
        ["\\lambda", "MixUp Interpolation Ratio", "Random mixing coefficient sampled from Beta distribution: \\lambda \\sim \\text{Beta}(\\alpha, \\alpha) with \\alpha=0.2."],
        ["x_i, x_j", "Input Pair Images", "Two distinct lung ROI image tensors sampled from training batch."],
        ["y_i, y_j", "Input Pair Labels", "Corresponding one-hot ground-truth label vectors for images i and j."],
        ["\\tilde{x}", "MixUp Composite Image", "Convex combination input image: \\tilde{x} = \\lambda x_i + (1-\\lambda) x_j."],
        ["\\tilde{y}", "MixUp Composite Label", "Convex combination target label vector: \\tilde{y} = \\lambda y_i + (1-\\lambda) y_j."],
        ["Z", "Unscaled Output Logits", "Raw 3-dimensional classification score vector produced by ResNet50 final linear layer."],
        ["P(c) \\text{ or } \\hat{y}_c", "Softmax Probability", "Predicted probability for class c: P(c) = \\exp(Z_c) / \\sum_{k=0}^2 \\exp(Z_k)."],
        ["\\mathcal{L}_{CE}", "Weighted Cross-Entropy", "Class-weighted loss: \\mathcal{L}_{CE} = - \\sum_{c=0}^{2} w_c \\tilde{y}_c \\log(P(c))."]
    ]
    
    tbl_r_vars = doc.add_table(rows=len(resnet_vars)+1, cols=3)
    style_table(tbl_r_vars, [1.4, 1.6, 3.5], ["Variable / Symbol", "Role Name", "Detailed Mathematical & Technical Description"], resnet_vars)

    add_styled_heading(doc, "3.4 Step-by-Step Execution Mechanics", level=2)
    add_body_p(doc, "1. Weight Calculation: Pre-calculate inverse class weights w_c to penalize majority class misclassifications heavily.")
    add_body_p(doc, "2. MixUp Augmentation: Sample λ ~ Beta(0.2, 0.2). Construct blended image x̃ and target label vector ỹ.")
    add_body_p(doc, "3. Residual Learning: Forward pass x̃ through ResNet50 residual shortcut blocks (F(x) + x), avoiding vanishing gradients.")
    add_body_p(doc, "4. Loss Backpropagation: Calculate weighted loss L_CE. Backpropagate gradients using AdamW optimizer (learning rate 1e-4).")

    add_presentation_pointers(doc, "Algorithm 3 (ResNet50 & MixUp)", [
        ["Why Inverse Class Weighting?", "Bacterial pneumonia comprises 47.5% of dataset samples. Without weights (w_1=0.70 vs w_2=1.31), the model would over-predict Bacterial cases. Class weighting forces equal attention to Viral and Normal cases."],
        ["Why MixUp Augmentation?", "Standard training causes deep networks to become overconfident in linear decision boundaries. MixUp forces the model to behave linearly between training samples, dramatically improving generalizability."],
        ["Key Clinical Impact", "Ensures high-sensitivity 3-way triage, supporting antimicrobial stewardship."]
    ])

    # --------------------------------------------------------------------------
    # ALGORITHM 4: GRAD-CAM EXPLAINABILITY
    # --------------------------------------------------------------------------
    doc.add_page_break()
    add_styled_heading(doc, "ALGORITHM 4: GRAD-CAM EXPLAINABLE AI SALIENCY MAPPING", level=1)
    
    add_styled_heading(doc, "4.1 Clinical Rationale & Objective", level=2)
    add_body_p(doc, 
               "Clinical deployment of medical AI requires explainability: attending radiologists will not trust 'black-box' predictions "
               "without visual confirmation of diagnostic features. Gradient-Weighted Class Activation Mapping (Grad-CAM) computes the "
               "gradients of target score Y^c with respect to Layer 4 bottleneck feature maps A^k. Applying Global Average Pooling (GAP) "
               "and ReLU activation generates a 2D spatial heatmap overlaying genuine pulmonary lesions.")

    add_styled_heading(doc, "4.2 Rendered Pseudocode & Formula Photos", level=2)
    embed_photo(doc, photos_dir / "photo_algo4_gradcam.png", "Photo 4.1: High-Resolution Pseudocode — Algorithm 4 (Grad-CAM Explainable AI Mapping)")
    embed_photo(doc, photos_dir / "photo_eq4_gradcam.png", "Photo 4.2: Rendered Mathematical Formulas — GAP Neuron Weights & Heatmap Activation")

    add_styled_heading(doc, "4.3 Exhaustive Variable & Mathematical Symbol Registry", level=2)
    
    gradcam_vars = [
        ["c", "Target Class Index", "Selected diagnostic class for visual explanation (0=Normal, 1=Bacterial, 2=Viral)."],
        ["Y^c", "Target Class Logit Score", "Unnormalized network classification output logit for class c prior to Softmax."],
        ["A^k", "Feature Activation Map", "k-th feature map tensor extracted from ResNet50 final convolutional layer (Layer 4 bottleneck)."],
        ["A_{i,j}^k", "Spatial Feature Cell", "Activation intensity value at spatial row i and column j within feature map A^k."],
        ["\\frac{\\partial Y^c}{\\partial A_{i,j}^k}", "Backpropagated Gradient", "Partial derivative of class score Y^c with respect to feature map cell A_{i,j}^k."],
        ["Z \\text{ or } (u \\times v)", "Feature Map Area", "Total spatial cell count in feature map (Z = u \\times v, where u, v are feature map dimensions)."],
        ["\\alpha_k^c", "Neuron Importance Weight", "Global Average Pooled weight quantifying total importance of feature channel k for class c."],
        ["L_{\\text{Grad-CAM}}^c", "Raw Activation Map", "Weighted sum of feature maps: L_{\\text{Grad-CAM}}^c = \\text{ReLU}( \\sum_k \\alpha_k^c A^k )."],
        ["\\text{ReLU}", "Rectified Linear Unit", "Activation function \\max(0, \\cdot) that filters out negative gradients and retains positive lesion features."],
        ["H_{\\text{overlay}}", "Final Heatmap Overlay", "Color-blended composite image: H_{\\text{overlay}} = 0.5 \\cdot I_{\\text{ROI}} + 0.5 \\cdot \\text{Jet}(L_{\\text{Grad-CAM}}^c)."]
    ]
    
    tbl_g_vars = doc.add_table(rows=len(gradcam_vars)+1, cols=3)
    style_table(tbl_g_vars, [1.4, 1.6, 3.5], ["Variable / Symbol", "Role Name", "Detailed Mathematical & Technical Description"], gradcam_vars)

    add_styled_heading(doc, "4.4 Step-by-Step Execution Mechanics", level=2)
    add_body_p(doc, "1. Forward Pass & Feature Capture: Pass ROI image I_{ROI} through ResNet50, storing Layer 4 bottleneck feature maps A^k.")
    add_body_p(doc, "2. Gradient Backpropagation: Compute score Y^c for target class c. Backpropagate gradients ∂Y^c / ∂A^k back to Layer 4.")
    add_body_p(doc, "3. GAP Weight Computation: Compute neuron weights α_k^c by taking Global Average Pooling of gradients over spatial dimensions (u x v).")
    add_body_p(doc, "4. ReLU Filtering & Jet Overlay: Compute weighted sum ∑ α_k^c A^k. Apply ReLU to discard negative features. Resize to 256x256, apply Jet colormap (red=high lesion probability, blue=low), and blend with radiograph.")

    add_presentation_pointers(doc, "Algorithm 4 (Grad-CAM Explainer)", [
        ["Why Apply ReLU?", "Without ReLU, features that decrease the class score are included in the map. ReLU ensures only positive evidence supporting class c is visualized in red."],
        ["Why Layer 4 Bottleneck?", "Earlier layers capture low-level edges; Layer 4 captures high-level semantic pathology (lobar consolidations vs reticulonodular opacities)."],
        ["Key Clinical Impact", "Provides transparent visual proof for attending radiologists on Streamlit web dashboard."]
    ])

    # --------------------------------------------------------------------------
    # SUMMARY QUICK REFERENCE CHEAT-SHEET MATRIX
    # --------------------------------------------------------------------------
    doc.add_page_break()
    add_styled_heading(doc, "SECTION 3: ALGORITHM SUMMARY CHEAT-SHEET MATRIX", level=1)
    
    matrix_headers = ["Alg #", "Algorithm Name", "Input -> Output", "Core Mathematical Formula", "Key Parameter & Advantage"]
    matrix_widths = [0.6, 1.4, 1.4, 1.7, 1.4]
    matrix_data = [
        ["Alg 1", "CLAHE Contrast Engine", "Raw CXR -> Enhanced Image", "N_{clip} = \\beta \\cdot (N_x N_y) / L\nS(u,v) = \\text{Bilinear}(CDF)", "Tile Grid: 8x8, β=2.0\nUnmasks ground-glass opacities without over-saturation."],
        ["Alg 2", "U-Net Lung Segmenter", "Enhanced Image -> Lung ROI", "L_{DiceBCE} = 0.5 L_{BCE} + 0.5 L_{Dice}\nL_{Dice} = 1 - \\frac{2 \\sum p_i g_i + \\epsilon}{\\sum p_i + \\sum g_i + \\epsilon}", "Dice Score: 0.956, IoU: 0.898\nEliminates rib/text label shortcut learning."],
        ["Alg 3", "ResNet50 Classifier", "Lung ROI -> 3-Class Probabilities", "w_c = N_{total} / (K \\cdot N_c)\n\\tilde{x} = \\lambda x_i + (1-\\lambda) x_j", "w_{Normal}=1.23, w_{Bacter}=0.70, w_{Viral}=1.31\nResolves severe class imbalance & overconfidence."],
        ["Alg 4", "Grad-CAM Explainer", "ROI + Logits -> Jet Heatmap", "\\alpha_k^c = \\frac{1}{Z} \\sum \\frac{\\partial Y^c}{\\partial A_{i,j}^k}\nL^c = \\text{ReLU}(\\sum \\alpha_k^c A^k)", "Layer 4 Bottleneck\nVisual proof for radiologists on Streamlit Web UI."]
    ]
    
    tbl_matrix = doc.add_table(rows=len(matrix_data)+1, cols=5)
    style_table(tbl_matrix, matrix_widths, matrix_headers, matrix_data)

    # Save output Word Document
    output_docx_path = workspace_dir / "Pneumonia_Detection_Algorithms_and_Formulas_Guide.docx"
    doc.save(str(output_docx_path))
    print(f"\n[SUCCESS] Saved comprehensive algorithm guide document at: {output_docx_path}")

# ==============================================================================
# SCRIPT EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    workspace_root = Path(__file__).parent.resolve()
    print(f"Generating algorithm guide document in: {workspace_root}")
    generate_guide_document(workspace_root)

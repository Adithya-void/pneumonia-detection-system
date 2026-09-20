"""
Self-contained Python script to generate 7 publication-grade, high-resolution (300 DPI)
architectural & system modeling diagrams for 1st Project Review PPT (Agenda Items 4 & 5).

Output Folder: reports/diagrams/
Project: Enhanced CNN-Based Multi-Class Pneumonia Detection System
Institution: The Oxford College of Engineering (TOCE), Dept. of ISE
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create output directory
OUTPUT_DIR = Path("reports/diagrams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Common Styling Constants (Clinical Palette)
BG_COLOR = "#F8FAFC"      # Soft Off-White / Slate 50
TEXT_DARK = "#0F172A"     # Slate 900
TEXT_MUTED = "#475569"    # Slate 600
DARK_SLATE = "#1E293B"    # Slate 800
DEEP_TEAL = "#0F766E"     # Teal 700
CYAN_ACCENT = "#06B6D4"   # Cyan 500
LIGHT_TEAL = "#CCFBF1"    # Teal 100
BORDER_TEAL = "#0D9488"   # Teal 600
LIGHT_CYAN = "#E0F2FE"    # Sky 100
BORDER_CYAN = "#0284C7"   # Sky 600
LIGHT_AMBER = "#FEF3C7"   # Amber 100
BORDER_AMBER = "#D97706"  # Amber 600
LIGHT_GRAY = "#F1F5F9"    # Slate 100
BORDER_GRAY = "#64748B"   # Slate 500
WHITE = "#FFFFFF"

def setup_canvas(title: str, subtitle: str, figsize=(16, 9)):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Title Header Block
    ax.text(0.5, 8.6, title, fontsize=15, fontweight='bold', color=DARK_SLATE, ha='left', va='top')
    ax.text(0.5, 8.25, subtitle, fontsize=10, fontweight='bold', color=DEEP_TEAL, ha='left', va='top')
    ax.plot([0.5, 15.5], [8.05, 8.05], color=BORDER_TEAL, linewidth=1.5)
    
    # Footer Banner
    ax.text(0.5, 0.3, "The Oxford College of Engineering (TOCE) — Department of Information Science & Engineering",
            fontsize=8.5, fontweight='bold', color=TEXT_MUTED, ha='left', va='bottom')
    ax.text(15.5, 0.3, "1st Project Review | Agenda Items 4 & 5",
            fontsize=8.5, fontweight='bold', color=DEEP_TEAL, ha='right', va='bottom')
    return fig, ax

def add_box(ax, x, y, w, h, title, subtitle="", bg_color=DEEP_TEAL, text_color=WHITE, border_color=BORDER_TEAL, fontsize=9.5):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.15",
                                 facecolor=bg_color, edgecolor=border_color, linewidth=1.8)
    ax.add_patch(box)
    
    if subtitle:
        ax.text(x + w/2, y + h/2 + 0.15, title, fontsize=fontsize, fontweight='bold', color=text_color, ha='center', va='center')
        ax.text(x + w/2, y + h/2 - 0.15, subtitle, fontsize=fontsize-1.5, fontweight='normal', color=text_color, ha='center', va='center')
    else:
        ax.text(x + w/2, y + h/2, title, fontsize=fontsize, fontweight='bold', color=text_color, ha='center', va='center')

def add_arrow(ax, start, end, label="", color=DEEP_TEAL, linestyle='-', connectionstyle="arc3,rad=0"):
    ax.annotate(label, xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=2.0, ls=linestyle, connectionstyle=connectionstyle),
                fontsize=8, fontweight='bold', color=TEXT_DARK, ha='center', va='center',
                bbox=dict(boxstyle='square,pad=0.1', fc=BG_COLOR, ec='none', alpha=0.8) if label else None)

# ==============================================================================
# DIAGRAM 1: Proposed System Architecture
# ==============================================================================
def generate_diagram_1():
    fig, ax = setup_canvas("01. Proposed System Architecture Pipeline",
                           "End-to-End Deep Learning Architecture: Multi-Class Pneumonia Triage System")
    
    # Horizontal Flow Boxes (Main Stage Boxes)
    stages = [
        ("Raw CXR Ingestion", "Kermany Dataset\n5,856 Radiographs", 0.5, 5.0, 1.8, 1.4, DARK_SLATE, WHITE),
        ("Stage 1: CLAHE Engine", "OpenCV Contrast\n8x8 Tile (β=2.0)", 2.8, 5.0, 1.9, 1.4, DEEP_TEAL, WHITE),
        ("Stage 2: U-Net Segmenter", "PyTorch Biomedical\nLung Mask Engine", 5.2, 5.0, 2.0, 1.4, DEEP_TEAL, WHITE),
        ("ROI Isolation Unit", "Element-Wise Hadamard\n(I_CLAHE ⊙ M_Mask)", 7.7, 5.0, 2.0, 1.4, CYAN_ACCENT, TEXT_DARK),
        ("Stage 3: ResNet50", "3-Class Classifier\n(MixUp / CutMix)", 10.2, 5.0, 1.9, 1.4, DEEP_TEAL, WHITE),
        ("Stage 4: Grad-CAM", "Layer 4 Feature Map\nGradient Heatmap", 12.6, 5.0, 1.8, 1.4, DARK_SLATE, WHITE),
        ("Streamlit Web UI", "Clinician Triage\nDashboard", 14.8, 5.0, 1.0, 1.4, BORDER_AMBER, WHITE)
    ]
    
    for title, sub, x, y, w, h, bg, txt in stages:
        add_box(ax, x, y, w, h, title, sub, bg_color=bg, text_color=txt, border_color=DARK_SLATE, fontsize=8.5)
        
    # Connect Stage Arrows
    add_arrow(ax, (2.3, 5.7), (2.8, 5.7))
    add_arrow(ax, (4.7, 5.7), (5.2, 5.7))
    add_arrow(ax, (7.2, 5.7), (7.7, 5.7))
    add_arrow(ax, (9.7, 5.7), (10.2, 5.7))
    add_arrow(ax, (12.1, 5.7), (12.6, 5.7))
    add_arrow(ax, (14.4, 5.7), (14.8, 5.7))
    
    # Sub-component Detail Callout Cards (Bottom Row)
    details = [
        ("Taxonomy Ingestion", "Normal (0): 1,583\nBacter (1): 2,780\nViral (2): 1,493", 0.5, 2.2, 2.0, 1.6, LIGHT_GRAY, BORDER_GRAY),
        ("Histogram Redistribution", "Local CDF Clipping\nNoise Limitation\nMicro-texture Boost", 2.8, 2.2, 2.0, 1.6, LIGHT_TEAL, BORDER_TEAL),
        ("U-Net Architecture", "Encoder-Decoder\nSkip Connections\nDiceBCELoss (0.956)", 5.2, 2.2, 2.0, 1.6, LIGHT_CYAN, BORDER_CYAN),
        ("Anatomical Masking", "Eliminates Ribs,\nSoft-Tissue & Letter\nShortcut Artifacts", 7.7, 2.2, 2.0, 1.6, LIGHT_AMBER, BORDER_AMBER),
        ("Classifier Optimization", "ResNet50 Backbone\nWeighted CE Loss\nAdamW Optimizer", 10.2, 2.2, 2.0, 1.6, LIGHT_TEAL, BORDER_TEAL),
        ("Explainability Unit", "Layer 4 Gradient GAP\nReLU Heatmap Mask\nRadiologist Triage", 12.6, 2.2, 2.0, 1.6, LIGHT_GRAY, BORDER_GRAY),
    ]
    
    for title, sub, x, y, w, h, bg, border in details:
        add_box(ax, x, y, w, h, title, sub, bg_color=bg, text_color=TEXT_DARK, border_color=border, fontsize=8.0)
        
    # Vertical Linking Dashed Lines
    for x in [1.5, 3.8, 6.2, 8.7, 11.2, 13.6]:
        add_arrow(ax, (x, 5.0), (x, 3.8), linestyle='--')
        
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_proposed_system_architecture.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("[DIAGRAM 1 OK] Generated 01_proposed_system_architecture.png")

# ==============================================================================
# DIAGRAM 2: Use Case Diagram
# ==============================================================================
def generate_diagram_2():
    fig, ax = setup_canvas("02. System Use Case Diagram",
                           "UML Boundary Specification: Actor-System Interactive Workflows")
    
    # System Boundary Rect
    boundary = patches.FancyBboxPatch((3.5, 1.2), 9.0, 6.5, boxstyle="round,pad=0.2,rounding_size=0.2",
                                      facecolor=WHITE, edgecolor=DARK_SLATE, linewidth=2.0)
    ax.add_patch(boundary)
    ax.text(8.0, 7.4, "PNEUMONIA DIAGNOSTIC SYSTEM BOUNDARY", fontsize=11, fontweight='bold', color=DEEP_TEAL, ha='center')
    
    # Actors (Left & Right)
    # Actor 1: Clinician / Radiologist (Left Box)
    add_box(ax, 0.5, 4.4, 2.3, 1.4, "PRIMARY ACTOR\n\nClinician /\nRadiologist", bg_color=DARK_SLATE, text_color=WHITE, border_color=DARK_SLATE, fontsize=8.5)
    
    # Actor 2: System Administrator / AI Engineer (Right Box)
    add_box(ax, 13.2, 4.4, 2.3, 1.4, "SECONDARY ACTOR\n\nSystem Admin /\nAI Engineer", bg_color=DARK_SLATE, text_color=WHITE, border_color=DARK_SLATE, fontsize=8.5)
    
    # Use Cases (Ovals inside boundary)
    use_cases = [
        ("UC1: Upload Chest Radiograph (CXR)", 4.0, 6.2, LIGHT_TEAL, BORDER_TEAL),
        ("UC2: Apply CLAHE Enhancement (8x8 Tile)", 4.0, 5.0, LIGHT_CYAN, BORDER_CYAN),
        ("UC3: Run U-Net Lung Field Segmentation", 4.0, 3.8, LIGHT_TEAL, BORDER_TEAL),
        ("UC4: Perform 3-Class Differential Diagnosis", 4.0, 2.6, LIGHT_AMBER, BORDER_AMBER),
        ("UC5: Inspect Grad-CAM Heatmap Overlay", 8.2, 5.6, LIGHT_CYAN, BORDER_CYAN),
        ("UC6: Export Clinical PDF Diagnostic Report", 8.2, 4.2, LIGHT_TEAL, BORDER_TEAL),
        ("UC7: Trigger Retraining & Update Checkpoints", 8.2, 2.8, LIGHT_GRAY, BORDER_GRAY),
        ("UC8: Monitor Telemetry & Log Performance", 8.2, 1.6, LIGHT_GRAY, BORDER_GRAY)
    ]
    
    for title, x, y, bg, border in use_cases:
        ellipse = patches.Ellipse((x + 1.8, y + 0.4), 3.4, 0.8, facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(ellipse)
        ax.text(x + 1.8, y + 0.4, title, fontsize=8.0, fontweight='bold', color=TEXT_DARK, ha='center', va='center')
        
    # Actor Links
    # Clinician Links
    for y_pos in [6.6, 5.4, 4.2, 3.0]:
        add_arrow(ax, (2.8, 5.1), (3.9, y_pos), color=DARK_SLATE)
    add_arrow(ax, (2.8, 5.1), (8.1, 6.0), color=DARK_SLATE)
    add_arrow(ax, (2.8, 5.1), (8.1, 4.6), color=DARK_SLATE)
    
    # Admin Links
    for y_pos in [3.2, 2.0]:
        add_arrow(ax, (13.2, 5.1), (11.7, y_pos), color=DARK_SLATE)
        
    # Include & Extend Dependency Relationships
    add_arrow(ax, (5.8, 6.2), (5.8, 5.8), label="<<include>>", color=BORDER_TEAL, linestyle=':')
    add_arrow(ax, (5.8, 5.0), (5.8, 4.6), label="<<include>>", color=BORDER_TEAL, linestyle=':')
    add_arrow(ax, (5.8, 3.8), (5.8, 3.4), label="<<include>>", color=BORDER_TEAL, linestyle=':')
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_use_case_diagram.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("[DIAGRAM 2 OK] Generated 02_use_case_diagram.png")

# ==============================================================================
# DIAGRAM 3: UML Class Diagram
# ==============================================================================
def generate_diagram_3():
    fig, ax = setup_canvas("03. Object-Oriented Class Diagram",
                           "UML Structural Architecture: Core Class Schemas & Dependencies")
    
    def draw_uml_class(x, y, w, h, class_name, attributes, methods, bg_header=DEEP_TEAL):
        # Outer border
        outer = patches.FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", facecolor=WHITE, edgecolor=DARK_SLATE, linewidth=1.8)
        ax.add_patch(outer)
        
        # Header Box
        hh = 0.6
        header = patches.Rectangle((x, y + h - hh), w, hh, facecolor=bg_header, edgecolor=DARK_SLATE, linewidth=1.5)
        ax.add_patch(header)
        ax.text(x + w/2, y + h - hh/2, class_name, fontsize=9.5, fontweight='bold', color=WHITE, ha='center', va='center')
        
        # Separator 1
        ax.plot([x, x + w], [y + h - hh, y + h - hh], color=DARK_SLATE, linewidth=1.5)
        
        # Attributes Box Text
        attr_y = y + h - hh - 0.2
        for attr in attributes:
            ax.text(x + 0.15, attr_y, attr, fontsize=7.5, color=TEXT_DARK, ha='left', va='top', fontfamily='monospace')
            attr_y -= 0.25
            
        # Separator 2
        sep_y = y + h - hh - (len(attributes) * 0.25) - 0.15
        ax.plot([x, x + w], [sep_y, sep_y], color=BORDER_GRAY, linewidth=1.0)
        
        # Methods Box Text
        meth_y = sep_y - 0.2
        for meth in methods:
            ax.text(x + 0.15, meth_y, meth, fontsize=7.5, color=TEXT_DARK, ha='left', va='top', fontfamily='monospace')
            meth_y -= 0.25

    # 5 System Classes
    draw_uml_class(0.5, 4.5, 2.7, 3.2, "DatasetParser",
                   ["- raw_data_dir: Path", "- metadata_csv: Path", "- taxonomy_map: Dict"],
                   ["+ resolve_data_root()", "+ validate_images()", "+ parse_kermany_3class()"])

    draw_uml_class(3.6, 4.5, 2.7, 3.2, "CLAHEEngine",
                   ["- clip_limit: float = 2.0", "- tile_grid: (8,8)", "- num_threads: int"],
                   ["+ apply_clahe(img): Array", "+ process_batch(): CSV", "+ redistribute_hist()"])

    draw_uml_class(6.7, 4.5, 2.8, 3.2, "UNetSegmenter",
                   ["- in_channels: int = 1", "- weights_path: Path", "- dice_loss: Module"],
                   ["+ forward(x): Tensor", "+ generate_mask(): Array", "+ extract_roi(): Array"])

    draw_uml_class(9.9, 4.5, 2.7, 3.2, "ResNet50Classifier",
                   ["- num_classes: int = 3", "- class_weights: Tensor", "- backbone: ResNet"],
                   ["+ forward(roi): Tensor", "+ train_epoch(): Tuple", "+ predict(): Dict"])

    draw_uml_class(13.0, 4.5, 2.6, 3.2, "GradCAMExplainer",
                   ["- target_layer: Layer4", "- heatmap_cam: Array"],
                   ["+ GAP_gradients(): Tensor", "+ generate_heatmap(): Array", "+ overlay_cam(): Array"],
                   bg_header=DARK_SLATE)

    # Database Entity Class (Bottom Center)
    draw_uml_class(6.7, 1.0, 2.8, 2.8, "DiagnosticRecord",
                   ["- record_id: str (PK)", "- patient_id: str", "- label_id: int", "- confidence: float"],
                   ["+ save_to_db()", "+ export_pdf_report()"],
                   bg_header=BORDER_AMBER)

    # Dependency Arrows
    add_arrow(ax, (3.2, 6.1), (3.6, 6.1), label="<<uses>>")
    add_arrow(ax, (6.3, 6.1), (6.7, 6.1), label="<<feeds>>")
    add_arrow(ax, (9.5, 6.1), (9.9, 6.1), label="<<passes ROI>>")
    add_arrow(ax, (12.6, 6.1), (13.0, 6.1), label="<<explains>>")
    
    # Vertical Association
    add_arrow(ax, (8.1, 4.5), (8.1, 3.8), label="<<persists>>", color=BORDER_AMBER)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_class_diagram.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("[DIAGRAM 3 OK] Generated 03_class_diagram.png")

# ==============================================================================
# DIAGRAM 4: Data Flow Diagram (DFD Level 1)
# ==============================================================================
def generate_diagram_4():
    fig, ax = setup_canvas("04. Data Flow Diagram (DFD Level 1)",
                           "Gane-Sarson / Yourdon Notation: Medical Data Vectors & Processing Nodes")

    # External Entity (Square Box)
    add_box(ax, 0.5, 4.2, 1.8, 1.6, "EXTERNAL ENTITY\n\nAttending Clinician\n/ Radiologist",
            bg_color=DARK_SLATE, text_color=WHITE, border_color=DARK_SLATE, fontsize=8.5)

    # Process Circles / Rounded Boxes
    processes = [
        ("1.0 Ingestion &\nValidation", 3.0, 6.2, LIGHT_TEAL, BORDER_TEAL),
        ("2.0 CLAHE Contrast\nEnhancement", 5.8, 6.2, LIGHT_CYAN, BORDER_CYAN),
        ("3.0 U-Net Lung\nSegmentation", 8.6, 6.2, LIGHT_TEAL, BORDER_TEAL),
        ("4.0 ResNet50\nClassification", 8.6, 2.6, LIGHT_AMBER, BORDER_AMBER),
        ("5.0 Grad-CAM\nHeatmap & UI", 4.4, 2.6, LIGHT_CYAN, BORDER_CYAN)
    ]

    for title, x, y, bg, border in processes:
        p_box = patches.FancyBboxPatch((x, y), 2.2, 1.4, boxstyle="circle,pad=0.2", facecolor=bg, edgecolor=border, linewidth=1.8)
        ax.add_patch(p_box)
        ax.text(x + 1.1, y + 0.7, title, fontsize=8.5, fontweight='bold', color=TEXT_DARK, ha='center', va='center')

    # Data Stores (Open Rectangle Notation)
    # Store D1: Dataset CSV
    ax.add_patch(patches.Rectangle((3.0, 0.8), 3.0, 1.0, facecolor=LIGHT_GRAY, edgecolor=BORDER_GRAY, linewidth=1.5))
    ax.text(4.5, 1.3, "D1: Dataset Metadata CSV", fontsize=8.5, fontweight='bold', color=TEXT_DARK, ha='center')
    
    # Store D2: Model Weights Registry
    ax.add_patch(patches.Rectangle((8.6, 0.8), 3.0, 1.0, facecolor=LIGHT_GRAY, edgecolor=BORDER_GRAY, linewidth=1.5))
    ax.text(10.1, 1.3, "D2: Model Weights (.pth)", fontsize=8.5, fontweight='bold', color=TEXT_DARK, ha='center')

    # Data Flow Vectors
    add_arrow(ax, (2.3, 5.0), (3.0, 6.7), label="Raw CXR File")
    add_arrow(ax, (5.2, 6.9), (5.8, 6.9), label="Validated CXR")
    add_arrow(ax, (8.0, 6.9), (8.6, 6.9), label="Enhanced CLAHE")
    add_arrow(ax, (9.7, 6.2), (9.7, 4.0), label="Lung Mask M(x,y)")
    add_arrow(ax, (8.6, 3.3), (6.6, 3.3), label="Probabilities Logits")
    add_arrow(ax, (4.4, 3.3), (1.4, 4.2), label="Diagnostic PDF & Heatmap")

    # Data Store Flows
    add_arrow(ax, (4.1, 6.2), (4.5, 1.8), label="Log Metadata", linestyle='--')
    add_arrow(ax, (10.1, 1.8), (9.7, 2.6), label="Load Weights", linestyle='--')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_data_flow_diagram_level1.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("[DIAGRAM 4 OK] Generated 04_data_flow_diagram_level1.png")

# ==============================================================================
# DIAGRAM 5: Entity-Relationship (ER) Diagram
# ==============================================================================
def generate_diagram_5():
    fig, ax = setup_canvas("05. Entity-Relationship (ER) Diagram",
                           "Database Schema Architecture & Relational Cardinalities")

    # Entity Boxes (Rectangles)
    entities = [
        ("PATIENT_RECORD", 1.0, 5.0, 2.4, 1.2, DARK_SLATE),
        ("RADIOGRAPH_SCAN", 4.8, 5.0, 2.4, 1.2, DEEP_TEAL),
        ("PREPROCESSED_IMAGE", 8.6, 5.0, 2.6, 1.2, DEEP_TEAL),
        ("SEGMENTATION_MASK", 12.4, 5.0, 2.5, 1.2, DEEP_TEAL),
        ("DIAGNOSTIC_PREDICTION", 8.6, 1.5, 2.6, 1.2, BORDER_AMBER)
    ]

    for title, x, y, w, h, bg in entities:
        add_box(ax, x, y, w, h, title, bg_color=bg, text_color=WHITE, border_color=DARK_SLATE, fontsize=9.0)

    # Relationships (Diamonds)
    def draw_diamond(x, y, text):
        diamond = patches.RegularPolygon((x, y), numVertices=4, radius=0.7,
                                         facecolor=LIGHT_CYAN, edgecolor=BORDER_CYAN, linewidth=1.5)
        ax.add_patch(diamond)
        ax.text(x, y, text, fontsize=7.5, fontweight='bold', color=TEXT_DARK, ha='center', va='center')

    draw_diamond(3.9, 5.6, "HAS_SCANS")
    draw_diamond(7.7, 5.6, "CLAHE_ENHANCE")
    draw_diamond(11.7, 5.6, "SEGMENTS")
    draw_diamond(9.9, 3.6, "DERIVES")

    # Connect Entities to Diamonds with Cardinality Labels
    ax.plot([3.4, 3.2], [5.6, 5.6], color=DARK_SLATE, lw=1.5)
    ax.text(3.5, 5.8, "1", fontsize=9, fontweight='bold')
    
    ax.plot([4.6, 4.8], [5.6, 5.6], color=DARK_SLATE, lw=1.5)
    ax.text(4.5, 5.8, "N", fontsize=9, fontweight='bold')

    ax.plot([7.2, 7.0], [5.6, 5.6], color=DARK_SLATE, lw=1.5)
    ax.text(7.3, 5.8, "1", fontsize=9, fontweight='bold')
    
    ax.plot([8.4, 8.6], [5.6, 5.6], color=DARK_SLATE, lw=1.5)
    ax.text(8.3, 5.8, "1", fontsize=9, fontweight='bold')

    ax.plot([11.2, 11.0], [5.6, 5.6], color=DARK_SLATE, lw=1.5)
    ax.text(11.3, 5.8, "1", fontsize=9, fontweight='bold')
    
    ax.plot([12.4, 12.2], [5.6, 5.6], color=DARK_SLATE, lw=1.5)
    ax.text(12.2, 5.8, "1", fontsize=9, fontweight='bold')

    # Vertical Relationship Link
    ax.plot([9.9, 9.9], [5.0, 4.3], color=DARK_SLATE, lw=1.5)
    ax.plot([9.9, 9.9], [2.9, 2.7], color=DARK_SLATE, lw=1.5)
    ax.text(10.1, 4.7, "1", fontsize=9, fontweight='bold')
    ax.text(10.1, 3.0, "1", fontsize=9, fontweight='bold')

    # Attributes (Ellipses attached to Entities)
    attributes = [
        ("Patient_ID (PK)", 1.0, 7.0), ("Age_Gender", 2.5, 7.0),
        ("Scan_ID (PK)", 4.8, 7.0), ("View_Pos (PA/AP)", 6.3, 7.0),
        ("Tile_Grid (8x8)", 8.6, 7.0), ("Clip_Limit (2.0)", 10.1, 7.0),
        ("Mask_Dice (0.956)", 12.4, 7.0), ("IoU (0.898)", 13.9, 7.0),
        ("Prediction_ID (PK)", 7.0, 1.5), ("Etiology_Class (0/1/2)", 7.0, 0.8),
        ("Confidence_Pct", 12.0, 1.5), ("GradCAM_Heatmap", 12.0, 0.8)
    ]

    for title, ax_x, ax_y in attributes:
        ellipse = patches.Ellipse((ax_x, ax_y), 1.6, 0.5, facecolor=LIGHT_GRAY, edgecolor=BORDER_GRAY, linewidth=1.0)
        ax.add_patch(ellipse)
        ax.text(ax_x, ax_y, title, fontsize=7.0, fontweight='bold', color=TEXT_DARK, ha='center', va='center')

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_er_diagram.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("[DIAGRAM 5 OK] Generated 05_er_diagram.png")

# ==============================================================================
# DIAGRAM 6: State Transition Diagram
# ==============================================================================
def generate_diagram_6():
    fig, ax = setup_canvas("06. UML State Transition Diagram",
                           "State Machine Specification: Diagnostic Lifecycle Transitions")

    # Initial State (Solid Circle)
    start = patches.Circle((1.0, 5.0), 0.25, facecolor=DARK_SLATE, edgecolor=DARK_SLATE)
    ax.add_patch(start)
    ax.text(1.0, 4.4, "Initial Node", fontsize=8, fontweight='bold', ha='center')

    # States Chain
    states = [
        ("IDLE", 2.2, 4.4, 1.4, 1.2, LIGHT_GRAY, BORDER_GRAY),
        ("IMAGE_\nUPLOADED", 4.2, 4.4, 1.5, 1.2, LIGHT_TEAL, BORDER_TEAL),
        ("CLAHE_\nENHANCING", 6.3, 4.4, 1.5, 1.2, LIGHT_CYAN, BORDER_CYAN),
        ("SEGMENTING_\nLUNGS", 8.4, 4.4, 1.6, 1.2, LIGHT_TEAL, BORDER_TEAL),
        ("ROI_\nEXTRACTED", 10.6, 4.4, 1.5, 1.2, LIGHT_AMBER, BORDER_AMBER),
        ("CLASSIFYING_\nRESNET50", 12.7, 4.4, 1.6, 1.2, LIGHT_TEAL, BORDER_TEAL),
        ("REPORT_\nREADY", 12.7, 1.8, 1.6, 1.2, CYAN_ACCENT, BORDER_CYAN)
    ]

    for title, x, y, w, h, bg, border in states:
        add_box(ax, x, y, w, h, title, bg_color=bg, text_color=TEXT_DARK, border_color=border, fontsize=8.0)

    # Final State (Double Circle)
    final_outer = patches.Circle((1.0, 1.8), 0.35, facecolor=WHITE, edgecolor=DARK_SLATE, linewidth=2.0)
    final_inner = patches.Circle((1.0, 1.8), 0.22, facecolor=DARK_SLATE, edgecolor=DARK_SLATE)
    ax.add_patch(final_outer)
    ax.add_patch(final_inner)
    ax.text(1.0, 1.2, "Final Node", fontsize=8, fontweight='bold', ha='center')

    # Transition Arrows with Event Triggers
    add_arrow(ax, (1.25, 5.0), (2.2, 5.0))
    add_arrow(ax, (3.6, 5.0), (4.2, 5.0), label="[Upload CXR]")
    add_arrow(ax, (5.7, 5.0), (6.3, 5.0), label="[Exec CLAHE]")
    add_arrow(ax, (7.8, 5.0), (8.4, 5.0), label="[Exec U-Net]")
    add_arrow(ax, (10.0, 5.0), (10.6, 5.0), label="[Hadamard Mask]")
    add_arrow(ax, (12.1, 5.0), (12.7, 5.0), label="[Exec ResNet]")
    add_arrow(ax, (13.5, 4.4), (13.5, 3.0), label="[Grad-CAM Heatmap]")
    add_arrow(ax, (12.7, 2.4), (1.35, 1.8), label="[Export PDF / Display Dashboard]")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_state_transition_diagram.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("[DIAGRAM 6 OK] Generated 06_state_transition_diagram.png")

# ==============================================================================
# DIAGRAM 7: Activity Diagram
# ==============================================================================
def generate_diagram_7():
    fig, ax = setup_canvas("07. UML Activity Diagram with Swimlanes",
                           "Workflow Logic & Control Decisions: Clinician UI vs AI Processing Engine")

    # Swimlane Dividers & Headers
    ax.plot([8.0, 8.0], [0.8, 7.8], color=BORDER_GRAY, linestyle='--', linewidth=2.0)
    
    # Swimlane Titles
    ax.text(4.0, 7.5, "SWIMLANE 1: CLINICIAN INTERFACE", fontsize=11, fontweight='bold', color=DEEP_TEAL, ha='center')
    ax.text(12.0, 7.5, "SWIMLANE 2: MEDICAL AI CORE ENGINE", fontsize=11, fontweight='bold', color=DARK_SLATE, ha='center')

    # Left Swimlane Nodes (Clinician)
    start = patches.Circle((4.0, 6.7), 0.2, facecolor=DARK_SLATE, edgecolor=DARK_SLATE)
    ax.add_patch(start)

    add_box(ax, 2.5, 5.4, 3.0, 0.8, "1. Select & Upload CXR Image", bg_color=LIGHT_TEAL, text_color=TEXT_DARK, border_color=BORDER_TEAL, fontsize=8.5)

    # Right Swimlane Nodes (AI Engine)
    add_box(ax, 10.5, 5.4, 3.0, 0.8, "2. Validate File & Format (8-bit)", bg_color=LIGHT_CYAN, text_color=TEXT_DARK, border_color=BORDER_CYAN, fontsize=8.5)
    
    # Decision Diamond
    dec1 = patches.RegularPolygon((12.0, 4.3), numVertices=4, radius=0.5, facecolor=LIGHT_AMBER, edgecolor=BORDER_AMBER, linewidth=1.5)
    ax.add_patch(dec1)
    ax.text(12.0, 4.3, "Valid?", fontsize=8, fontweight='bold')

    add_box(ax, 10.5, 3.0, 3.0, 0.8, "3. Apply CLAHE Enhancement", bg_color=LIGHT_TEAL, text_color=TEXT_DARK, border_color=BORDER_TEAL, fontsize=8.5)
    add_box(ax, 10.5, 1.8, 3.0, 0.8, "4. Infer U-Net & ResNet50 Logits", bg_color=LIGHT_CYAN, text_color=TEXT_DARK, border_color=BORDER_CYAN, fontsize=8.5)

    # Return to Left Swimlane
    add_box(ax, 2.5, 1.8, 3.0, 0.8, "5. Display Grad-CAM & PDF Report", bg_color=CYAN_ACCENT, text_color=TEXT_DARK, border_color=BORDER_CYAN, fontsize=8.5)

    final_outer = patches.Circle((4.0, 0.8), 0.25, facecolor=WHITE, edgecolor=DARK_SLATE, linewidth=2.0)
    final_inner = patches.Circle((4.0, 0.8), 0.15, facecolor=DARK_SLATE, edgecolor=DARK_SLATE)
    ax.add_patch(final_outer)
    ax.add_patch(final_inner)

    # Connect Activity Workflow Arrows
    add_arrow(ax, (4.0, 6.5), (4.0, 6.2))
    add_arrow(ax, (4.0, 5.4), (10.5, 5.8), label="HTTP POST")
    add_arrow(ax, (12.0, 5.4), (12.0, 4.8))
    add_arrow(ax, (12.0, 3.8), (12.0, 3.8), label="Yes", color=BORDER_TEAL)
    add_arrow(ax, (12.0, 3.0), (12.0, 2.6))
    add_arrow(ax, (10.5, 2.2), (5.5, 2.2), label="JSON Response")
    add_arrow(ax, (4.0, 1.8), (4.0, 1.05))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_activity_diagram.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("[DIAGRAM 7 OK] Generated 07_activity_diagram.png")

# ==============================================================================
# SCRIPT EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    print(f"Executing architecture diagram generator script in workspace...")
    print(f"Target Output Directory: {OUTPUT_DIR.resolve()}\n")
    
    generate_diagram_1()
    generate_diagram_2()
    generate_diagram_3()
    generate_diagram_4()
    generate_diagram_5()
    generate_diagram_6()
    generate_diagram_7()
    
    print("\n[SUCCESS] All 7 publication-grade architecture PNG diagrams generated successfully!")

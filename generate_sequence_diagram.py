"""
Python Script to Generate a Publication-Grade 300-DPI UML Sequence Diagram PNG.

Diagram Title: UML Sequence Diagram: End-to-End Multi-Class Pneumonia Detection & Explainable Inference Flow
Project: An Enhanced CNN-Based Multi-Class Pneumonia Detection System Using Advanced Image Enhancement, Segmentation, And Data Augmentation Techniques
Institution: The Oxford College of Engineering (TOCE), Dept. of ISE

Output Paths:
- reports/diagrams/08_uml_sequence_diagram.png
- generated_photos/architecture_diagrams/08_uml_sequence_diagram.png
"""

import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Output Paths Setup
REPORTS_DIR = Path("reports/diagrams")
GEN_PHOTOS_DIR = Path("generated_photos/architecture_diagrams")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
GEN_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

# Styling Color Tokens (Clinical / Professional Palette)
BG_COLOR = "#F8FAFC"        # Off-white background
DARK_SLATE = "#0F172A"      # Main titles
TEXT_DARK = "#1E293B"       # Body text
TEXT_MUTED = "#64748B"      # Subtitles & secondary labels
BORDER_COLOR = "#334155"    # Solid borders

# Lifeline Actor Colors (7 Lifelines)
LIFELINE_COLORS = [
    {"bg": "#E0F2FE", "border": "#0284C7", "text": "#0369A1", "label": "Radiologist /\nClinician"},                # User Actor
    {"bg": "#FEF3C7", "border": "#D97706", "text": "#B45309", "label": "Streamlit Web\nDashboard"},                # Web UI
    {"bg": "#DCFCE7", "border": "#15803D", "text": "#166534", "label": "OpenCV CLAHE\nEnhancement Engine"},        # CLAHE Engine
    {"bg": "#F3E8FF", "border": "#7E22CE", "text": "#6B21A8", "label": "U-Net Semantic\nSegmentation Engine"},      # U-Net Engine
    {"bg": "#CCFBF1", "border": "#0D9488", "text": "#0F766E", "label": "ResNet50 Multi-Class\nClassifier"},         # ResNet50 Classifier
    {"bg": "#FFE4E6", "border": "#E11D48", "text": "#BE123C", "label": "Grad-CAM Explainable\nAI (XAI) Engine"},    # Grad-CAM Engine
    {"bg": "#F1F5F9", "border": "#475569", "text": "#334155", "label": "Diagnostic Report\nGenerator"}            # Report Generator
]

def draw_uml_sequence_diagram():
    fig, ax = plt.subplots(figsize=(18, 12), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Header Title Block
    ax.text(0.6, 11.6, "UML SEQUENCE DIAGRAM: END-TO-END SYSTEM INFERENCE FLOW",
            fontsize=16, fontweight='bold', color=DARK_SLATE, ha='left', va='top')
    ax.text(0.6, 11.25, "An Enhanced CNN-Based Multi-Class Pneumonia Detection System Using Advanced Image Enhancement, Segmentation & XAI",
            fontsize=10.5, fontweight='bold', color="#0D9488", ha='left', va='top')
    ax.text(0.6, 10.95, "Department of Information Science & Engineering | The Oxford College of Engineering (TOCE)",
            fontsize=9.5, fontweight='bold', color=TEXT_MUTED, ha='left', va='top')
    ax.plot([0.6, 17.4], [10.75, 10.75], color="#0D9488", linewidth=1.8)

    # Lifeline Position Mapping (7 Vertical Lifelines)
    num_lifelines = len(LIFELINE_COLORS)
    x_coords = np.linspace(1.4, 16.6, num_lifelines)
    
    top_y = 10.3
    bottom_y = 0.8
    box_width = 2.0
    box_height = 0.65

    # Draw Lifeline Headers (Top & Bottom) & Dashed Vertical Lines
    for idx, (x, col) in enumerate(zip(x_coords, LIFELINE_COLORS)):
        # Top Header Box
        rect_top = patches.FancyBboxPatch(
            (x - box_width / 2, top_y - box_height), box_width, box_height,
            boxstyle="round,pad=0.03,rounding_size=0.1",
            facecolor=col["bg"], edgecolor=col["border"], linewidth=1.5, zorder=4
        )
        ax.add_patch(rect_top)
        ax.text(x, top_y - box_height / 2, col["label"], fontsize=8.5, fontweight='bold',
                color=col["text"], ha='center', va='center', zorder=5)

        # Bottom Header Box
        rect_bot = patches.FancyBboxPatch(
            (x - box_width / 2, bottom_y), box_width, box_height,
            boxstyle="round,pad=0.03,rounding_size=0.1",
            facecolor=col["bg"], edgecolor=col["border"], linewidth=1.5, zorder=4
        )
        ax.add_patch(rect_bot)
        ax.text(x, bottom_y + box_height / 2, col["label"], fontsize=8.5, fontweight='bold',
                color=col["text"], ha='center', va='center', zorder=5)

        # Vertical Dashed Lifeline
        ax.plot([x, x], [top_y - box_height, bottom_y + box_height],
                color=BORDER_COLOR, linestyle="--", linewidth=1.2, alpha=0.6, zorder=1)

    # Sequence Messages Definition
    # Message format: (from_idx, to_idx, y_pos, label, is_return, is_self)
    messages = [
        (0, 1, 9.35, "1. Uploads Grayscale Chest Radiograph (DICOM / JPEG)", False, False),
        (1, 1, 8.90, "2. Validate Format & Pixel Matrix (256x256)", False, True),
        (1, 2, 8.45, "3. Send Grayscale Image I(x,y) to CLAHE Engine", False, False),
        (2, 2, 8.05, "4. Compute 8x8 Tile Grid CLAHE (Clip Limit β=2.0)", False, True),
        (2, 1, 7.65, "5. Return Enhanced Radiograph I_CLAHE", True, False),
        (1, 3, 7.25, "6. Forward Enhanced Radiograph I_CLAHE", False, False),
        (3, 3, 6.85, "7. U-Net Segmentation & Morphological Mask M_lung", False, True),
        (3, 3, 6.45, "8. Isolate Pulmonary ROI: I_ROI = I_CLAHE ⊙ M_lung", False, True),
        (3, 1, 6.05, "9. Return Isolated Lung ROI & Dice Score (0.956)", True, False),
        (1, 4, 5.65, "10. Pass Segmented ROI to ResNet50 Classifier", False, False),
        (4, 4, 5.25, "11. ResNet50 Softmax Probabilities P(Normal, Bacterial, Viral)", False, True),
        (4, 1, 4.85, "12. Return Multi-Class Diagnosis & Confidence Score", True, False),
        (1, 5, 4.45, "13. Request Feature Map Saliency for Bottleneck Layer 4", False, False),
        (5, 5, 4.05, "14. Compute Gradient Weights α_k^c & ReLU Activation Map L_Grad-CAM", False, True),
        (5, 1, 3.65, "15. Return Visual Heatmap & Overlay H_overlay", True, False),
        (1, 6, 3.25, "16. Package Classification, Segmentation Mask & XAI Heatmap", False, False),
        (6, 6, 2.85, "17. Compile Clinical Diagnostic Summary & PDF Report", False, True),
        (6, 1, 2.45, "18. Deliver Generated Diagnostic Report to Web Dashboard", True, False),
        (1, 0, 1.95, "19. Render Interactive Diagnostic Dashboard & Visual Overlay", True, False)
    ]

    # Draw Messages and Activation Boxes
    for from_idx, to_idx, y_pos, label, is_return, is_self in messages:
        x_from = x_coords[from_idx]
        x_to = x_coords[to_idx]

        # Draw Activation Boxes on Lifelines
        act_width = 0.16
        act_height = 0.35
        for x in [x_from, x_to]:
            act_rect = patches.Rectangle(
                (x - act_width / 2, y_pos - act_height / 2), act_width, act_height,
                facecolor="#FFFFFF", edgecolor="#0D9488", linewidth=1.2, zorder=3
            )
            ax.add_patch(act_rect)

        if is_self:
            # Self-Call Loop Arrow
            loop_x_end = x_from + 0.65
            loop_y_top = y_pos + 0.12
            loop_y_bot = y_pos - 0.12

            ax.plot([x_from + 0.08, loop_x_end, loop_x_end, x_from + 0.08],
                    [loop_y_top, loop_y_top, loop_y_bot, loop_y_bot],
                    color="#0284C7", linewidth=1.4, linestyle="-", zorder=4)

            # Arrowhead pointing back to activation box
            ax.annotate("", xy=(x_from + 0.08, loop_y_bot), xytext=(x_from + 0.25, loop_y_bot),
                        arrowprops=dict(arrowstyle="->", color="#0284C7", lw=1.4), zorder=5)

            # Text label beside self loop
            ax.text(loop_x_end + 0.08, y_pos, label, fontsize=8.0, fontweight='bold',
                    color="#0F172A", ha='left', va='center', zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=0.8))

        else:
            # Horizontal Message Arrow between Lifelines
            line_style = "--" if is_return else "-"
            line_color = "#0D9488" if is_return else "#1E293B"
            arrow_style = "<-" if (is_return and x_from > x_to) else "->"

            ax.plot([x_from, x_to], [y_pos, y_pos], color=line_color, linestyle=line_style, linewidth=1.4, zorder=4)

            # Arrowhead
            if x_to > x_from:
                arrow_x_target = x_to - 0.08
                arrow_x_start = x_to - 0.3
            else:
                arrow_x_target = x_to + 0.08
                arrow_x_start = x_to + 0.3

            ax.annotate("", xy=(arrow_x_target, y_pos), xytext=(arrow_x_start, y_pos),
                        arrowprops=dict(arrowstyle="->", color=line_color, lw=1.4), zorder=5)

            # Text Label on Arrow Line
            mid_x = (x_from + x_to) / 2
            text_color = "#0F766E" if is_return else "#0F172A"
            ax.text(mid_x, y_pos + 0.12, label, fontsize=8.0, fontweight='bold',
                    color=text_color, ha='center', va='bottom', zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=0.8, alpha=0.95))

    # Add Sequence Diagram Legend / Notes Footer Box
    legend_rect = patches.FancyBboxPatch(
        (0.6, 0.15), 16.8, 0.45,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor="#F1F5F9", edgecolor="#94A3B8", linewidth=1.0, zorder=4
    )
    ax.add_patch(legend_rect)
    
    legend_text = (
        "DIAGRAM LEGEND & NOTATION:  "
        "───► Synchronous Method Call   │   "
        "─ ─ ─► Asynchronous Return Response   │   "
        "↺ Self-Execution Loop   │   "
        "▮ Active Execution Context (Activation Bar)"
    )
    ax.text(9.0, 0.375, legend_text, fontsize=8.2, fontweight='bold', color=DARK_SLATE, ha='center', va='center', zorder=5)

    # Save 300-DPI Output Files
    out_file1 = REPORTS_DIR / "08_uml_sequence_diagram.png"
    out_file2 = GEN_PHOTOS_DIR / "08_uml_sequence_diagram.png"

    plt.tight_layout()
    plt.savefig(out_file1, dpi=300, bbox_inches='tight')
    plt.savefig(out_file2, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[SUCCESS] Saved 300-DPI UML Sequence Diagram to:")
    print(f"  • {out_file1}")
    print(f"  • {out_file2}")

if __name__ == "__main__":
    draw_uml_sequence_diagram()

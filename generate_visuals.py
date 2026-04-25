"""
Generate visuals for RenAIssance-GSoC-2025 README.
Tailored to this specific project:
1. Hero: Show actual Renaissance manuscript pages → layout masks → detected text overlay
2. Pipeline: Full document analysis pipeline (3 tasks: layout, OCR, synthetic text)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
import os, glob

BG = '#0d1117'
CARD = '#161b22'
BORDER = '#30363d'
ACCENT = '#58a6ff'
TEXT = '#e6edf3'
MUTED = '#8b949e'
GREEN = '#3fb950'
PURPLE = '#bc8cff'
ORANGE = '#f0883e'
PINK = '#f778ba'
YELLOW = '#d29922'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': CARD,
    'text.color': TEXT,
    'axes.labelcolor': TEXT,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'font.family': 'DejaVu Sans',
})

BASE = '/home/user/workspace/RenAIssance-GSoC-2025'
out = f'{BASE}/assets'
os.makedirs(out, exist_ok=True)


def make_hero():
    """Show real manuscript pages side-by-side with their text masks from the dataset."""
    fig = plt.figure(figsize=(14, 7), facecolor=BG)
    
    fig.text(0.5, 0.97, 'RenAIssance \u2014 GSoC 2025', fontsize=26, fontweight='bold',
             color=ACCENT, ha='center', va='top')
    fig.text(0.5, 0.91, 'Layout Recognition & OCR for Historical Renaissance Documents',
             fontsize=11, color=MUTED, ha='center', va='top')
    
    # Find 3 document-mask pairs
    train_dir = f'{BASE}/data/renaissance/train'
    mask_dir = f'{BASE}/data/renaissance/train_masks'
    
    # Pick 3 diverse documents
    doc_names = [
        'Buendia - Instruccion_page_1.png',
        'Ezcaray - Vozes_page_1.png',
        'Mendo - Principe perfecto_page_1.png',
    ]
    
    gs = fig.add_gridspec(2, 3, left=0.03, right=0.97, top=0.85, bottom=0.08,
                          hspace=0.15, wspace=0.08)
    
    for i, name in enumerate(doc_names):
        doc_path = os.path.join(train_dir, name)
        mask_path = os.path.join(mask_dir, name)
        
        # Top row: Original document
        ax_doc = fig.add_subplot(gs[0, i])
        try:
            img = Image.open(doc_path).convert('RGB')
            ax_doc.imshow(img)
        except:
            ax_doc.text(0.5, 0.5, 'Document', ha='center', va='center', color=MUTED, fontsize=12)
        ax_doc.set_xticks([])
        ax_doc.set_yticks([])
        for spine in ax_doc.spines.values():
            spine.set_color(ACCENT)
            spine.set_linewidth(1.5)
        
        short = name.split(' - ')[0] if ' - ' in name else name[:15]
        ax_doc.set_title(f'Document: {short}', fontsize=9, color=TEXT, pad=5)
        
        # Bottom row: Mask with colored overlay
        ax_mask = fig.add_subplot(gs[1, i])
        try:
            mask = Image.open(mask_path).convert('L')
            mask_arr = np.array(mask)
            # Create colored overlay: text regions in accent color
            overlay = np.zeros((*mask_arr.shape, 3), dtype=np.uint8)
            overlay[..., 0] = 13  # BG dark
            overlay[..., 1] = 17
            overlay[..., 2] = 23
            # Text regions highlighted
            text_mask = mask_arr > 128
            overlay[text_mask, 0] = 88   # accent blue
            overlay[text_mask, 1] = 166
            overlay[text_mask, 2] = 255
            ax_mask.imshow(overlay)
        except:
            ax_mask.text(0.5, 0.5, 'Mask', ha='center', va='center', color=MUTED, fontsize=12)
        ax_mask.set_xticks([])
        ax_mask.set_yticks([])
        for spine in ax_mask.spines.values():
            spine.set_color(GREEN)
            spine.set_linewidth(1.5)
        
        ax_mask.set_title('Text Segmentation Mask', fontsize=9, color=GREEN, pad=5)
        
        # Arrow between rows
        if i == 1:
            fig.text(0.5, 0.50, '\u2193  Model Inference  \u2193', fontsize=10, color=ORANGE,
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor=ORANGE, alpha=0.95))
    
    fig.savefig(f'{out}/hero_banner.png', dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print("\u2713 hero_banner.png")


def make_pipeline():
    """Show the 3-task pipeline: Layout → OCR → Synthetic, with model architectures."""
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.set_facecolor(BG)
    ax.axis('off')
    
    fig.text(0.5, 0.97, 'Three-Task Document Analysis Pipeline',
             fontsize=16, fontweight='bold', color=ACCENT, ha='center', va='top')
    
    # Task 1: Layout Recognition
    task_y = 4.5
    ax.add_patch(FancyBboxPatch((0.3, task_y - 0.8), 4.0, 1.6,
                boxstyle="round,pad=0.15", facecolor='#1a3a5c', edgecolor=ACCENT, linewidth=1.5))
    ax.text(2.3, task_y + 0.35, 'Task 1: Layout Recognition', fontsize=11,
            color=ACCENT, ha='center', va='center', fontweight='bold')
    ax.text(2.3, task_y - 0.05, 'Models: ResNet34 | U-Net | LayoutLMv3', fontsize=8,
            color=TEXT, ha='center', va='center')
    ax.text(2.3, task_y - 0.35, 'Metrics: IoU, Dice, Precision, Recall, F1', fontsize=7,
            color=MUTED, ha='center', va='center')
    
    # Task 2: OCR
    task2_y = 2.5
    ax.add_patch(FancyBboxPatch((0.3, task2_y - 0.8), 4.0, 1.6,
                boxstyle="round,pad=0.15", facecolor='#2d1a3e', edgecolor=PURPLE, linewidth=1.5))
    ax.text(2.3, task2_y + 0.35, 'Task 2: Optical Character Recognition', fontsize=11,
            color=PURPLE, ha='center', va='center', fontweight='bold')
    ax.text(2.3, task2_y - 0.05, 'Model: CRNN (ResNet + BiGRU + CTC)', fontsize=8,
            color=TEXT, ha='center', va='center')
    ax.text(2.3, task2_y - 0.35, 'Metrics: CER, WER', fontsize=7,
            color=MUTED, ha='center', va='center')
    
    # Task 3: Synthetic Generation
    task3_y = 0.5
    ax.add_patch(FancyBboxPatch((0.3, task3_y - 0.8), 4.0, 1.6,
                boxstyle="round,pad=0.15", facecolor='#1a3e2d', edgecolor=GREEN, linewidth=1.5))
    ax.text(2.3, task3_y + 0.35, 'Task 3: Synthetic Text Generation', fontsize=11,
            color=GREEN, ha='center', va='center', fontweight='bold')
    ax.text(2.3, task3_y - 0.05, 'Model: Diffusion + Period Degradation', fontsize=8,
            color=TEXT, ha='center', va='center')
    ax.text(2.3, task3_y - 0.35, 'Metrics: Diversity, OCR Readability', fontsize=7,
            color=MUTED, ha='center', va='center')
    
    # Connecting arrows (flow between tasks)
    ax.annotate('', xy=(2.3, task2_y + 0.8), xytext=(2.3, task_y - 0.8),
               arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.5))
    ax.annotate('', xy=(2.3, task3_y + 0.8), xytext=(2.3, task2_y - 0.8),
               arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.5))
    
    # Right side: Model comparison for layout recognition
    comp_x = 5.5
    ax.add_patch(FancyBboxPatch((comp_x, 0.3), 8.0, 5.0,
                boxstyle="round,pad=0.15", facecolor=CARD, edgecolor=BORDER, linewidth=1))
    ax.text(comp_x + 4.0, 5.0, 'Layout Model Comparison', fontsize=12,
            color=TEXT, ha='center', va='center', fontweight='bold')
    
    models = [
        ('ResNet34 + Decoder', 'Binary segmentation\nSimple encoder-decoder', ACCENT, 4.0),
        ('U-Net', 'Skip connections\nEncoder-decoder with\nbilinear upsampling', PURPLE, 2.7),
        ('LayoutLMv3', 'Transformer-based\nDocument-specific\npre-training', GREEN, 1.4),
    ]
    
    for name, desc, color, y in models:
        ax.add_patch(FancyBboxPatch((comp_x + 0.3, y - 0.4), 3.2, 0.9,
                    boxstyle="round,pad=0.1", facecolor=BG, edgecolor=color, linewidth=1.5))
        ax.text(comp_x + 1.9, y, name, fontsize=9, color=color,
                ha='center', va='center', fontweight='bold')
        
        ax.text(comp_x + 5.5, y, desc, fontsize=7, color=MUTED,
                ha='center', va='center', linespacing=1.3)
    
    # Features list
    features = [
        ('Adaptive Thresholding', ORANGE),
        ('Morphological Refinement', ORANGE),
        ('Dark Area Focus', ORANGE),
        ('Region Filtering', ORANGE),
    ]
    fx = comp_x + 6.8
    ax.text(fx + 0.5, 4.5, 'Post-Processing', fontsize=9, color=ORANGE,
            ha='center', va='center', fontweight='bold')
    for i, (feat, color) in enumerate(features):
        fy = 4.0 - i * 0.35
        ax.text(fx - 0.1, fy, '\u2022', fontsize=10, color=color, ha='center', va='center')
        ax.text(fx + 0.1, fy, feat, fontsize=7, color=TEXT, ha='left', va='center')
    
    fig.savefig(f'{out}/pipeline.png', dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close()
    print("\u2713 pipeline.png")


if __name__ == '__main__':
    make_hero()
    make_pipeline()
    print("All RenAIssance visuals generated.")

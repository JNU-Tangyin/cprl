"""
Conformal Prediction Results Analysis Script
Generates publication-quality figures and statistical summaries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Nature-style color palette
NATURE_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']

print("=" * 80)
print("CONFORMAL PREDICTION RESULTS ANALYSIS")
print("=" * 80)

# Load main results file
results_path = Path('/Users/yin/Desktop/doing/cprl/all_results/baseline_results/conformal_results.csv')
if results_path.exists():
    df = pd.read_csv(results_path)
    print(f"\n✓ Loaded main results: {len(df)} experiments")
else:
    # Fallback to basic results
    results_path = Path('/Users/yin/Desktop/doing/cprl/results/conformal_results.csv')
    df = pd.read_csv(results_path)
    print(f"\n✓ Loaded basic results: {len(df)} experiments")

# Parse setting column to extract model and configuration info
def parse_setting(setting_str):
    """Parse experiment setting string"""
    parts = setting_str.split('_')
    info = {}
    
    # Extract model name
    for i, part in enumerate(parts):
        if part.startswith('model'):
            info['model'] = part.replace('model', '')
        elif part.startswith('cp'):
            info['cp_mode'] = part.replace('cp', '')
        elif part == 'lags':
            info['lags'] = int(parts[i+1]) if i+1 < len(parts) else None
    
    return info

# Apply parsing
parsed_info = df['setting'].apply(parse_setting)
df['model'] = parsed_info.apply(lambda x: x.get('model', 'unknown'))
df['lags'] = parsed_info.apply(lambda x: x.get('lags', 96))

print(f"\n{'─' * 80}")
print("DATA OVERVIEW")
print(f"{'─' * 80}")
print(f"Total experiments: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nCP Modes: {df['cp_mode'].unique()}")
print(f"Models: {df['model'].unique()}")

# Statistical summary
print(f"\n{'─' * 80}")
print("SUMMARY STATISTICS BY CP MODE")
print(f"{'─' * 80}")

summary_cols = ['coverage', 'coverage_gap', 'avg_width', 'ces', 'rcs', 'point_mse', 'point_mae']
summary_stats = df.groupby('cp_mode')[summary_cols].agg(['mean', 'std', 'median']).round(4)
print(summary_stats)

# Coverage vs Target analysis
print(f"\n{'─' * 80}")
print("COVERAGE ANALYSIS (Target = 0.9)")
print(f"{'─' * 80}")

coverage_analysis = df.groupby('cp_mode').agg({
    'coverage': ['mean', 'std', 'min', 'max'],
    'coverage_gap': ['mean', 'std'],
    'avg_width': ['mean', 'std']
}).round(4)
print(coverage_analysis)

# Model performance comparison
print(f"\n{'─' * 80}")
print("MODEL PERFORMANCE (Top 10 by Coverage)")
print(f"{'─' * 80}")

top_performers = df.nlargest(10, 'coverage')[['setting', 'model', 'cp_mode', 'coverage', 'avg_width', 'point_mse']]
print(top_performers.to_string(index=False))

# Generate visualizations
print(f"\n{'─' * 80}")
print("GENERATING VISUALIZATIONS")
print(f"{'─' * 80}")

fig_dir = Path('/Users/yin/Desktop/doing/cprl/analysis_figures')
fig_dir.mkdir(exist_ok=True)

# Figure 1: Coverage by CP Mode
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Subplot 1: Coverage distribution by CP mode
ax1 = axes[0, 0]
cp_modes = df['cp_mode'].unique()
coverage_data = [df[df['cp_mode'] == mode]['coverage'].values for mode in cp_modes]
bp1 = ax1.boxplot(coverage_data, labels=cp_modes, patch_artist=True)
for patch, color in zip(bp1['boxes'], NATURE_COLORS[:len(cp_modes)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.axhline(y=0.9, color='red', linestyle='--', linewidth=1, label='Target (0.9)')
ax1.set_ylabel('Coverage', fontweight='bold')
ax1.set_xlabel('CP Mode', fontweight='bold')
ax1.set_title('(A) Coverage Distribution by CP Mode', fontweight='bold', fontsize=11)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Subplot 2: Width vs Coverage scatter
ax2 = axes[0, 1]
for i, mode in enumerate(cp_modes[:5]):  # Limit to first 5 for clarity
    mode_data = df[df['cp_mode'] == mode]
    ax2.scatter(mode_data['avg_width'], mode_data['coverage'], 
               c=NATURE_COLORS[i], label=mode, alpha=0.6, s=50)
ax2.axhline(y=0.9, color='red', linestyle='--', linewidth=1)
ax2.set_xlabel('Average Width', fontweight='bold')
ax2.set_ylabel('Coverage', fontweight='bold')
ax2.set_title('(B) Coverage vs Interval Width', fontweight='bold', fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Subplot 3: Coverage Gap analysis
ax3 = axes[1, 0]
gap_data = [df[df['cp_mode'] == mode]['coverage_gap'].values for mode in cp_modes]
bp3 = ax3.boxplot(gap_data, labels=cp_modes, patch_artist=True)
for patch, color in zip(bp3['boxes'], NATURE_COLORS[:len(cp_modes)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.axhline(y=0, color='green', linestyle='--', linewidth=1, label='Perfect (0)')
ax3.set_ylabel('Coverage Gap', fontweight='bold')
ax3.set_xlabel('CP Mode', fontweight='bold')
ax3.set_title('(C) Coverage Gap by CP Mode', fontweight='bold', fontsize=11)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Subplot 4: Model comparison (heatmap)
ax4 = axes[1, 1]
# Create pivot table for heatmap
pivot_data = df.pivot_table(values='coverage', index='model', columns='cp_mode', aggfunc='mean')
im = ax4.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=1.0)
ax4.set_xticks(range(len(pivot_data.columns)))
ax4.set_xticklabels(pivot_data.columns, rotation=45, ha='right')
ax4.set_yticks(range(len(pivot_data.index)))
ax4.set_yticklabels(pivot_data.index)
ax4.set_title('(D) Coverage Heatmap (Model × CP Mode)', fontweight='bold', fontsize=11)
# Add text annotations
for i in range(len(pivot_data.index)):
    for j in range(len(pivot_data.columns)):
        text = ax4.text(j, i, f'{pivot_data.values[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=8)
plt.colorbar(im, ax=ax4, label='Coverage')

plt.suptitle('Conformal Prediction Results Analysis', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
fig_path = fig_dir / 'cp_results_overview.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_dir / 'cp_results_overview.pdf', bbox_inches='tight')
print(f"✓ Saved: {fig_path}")

# Figure 2: Performance metrics comparison
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

# Coverage by model
ax = axes2[0]
model_coverage = df.groupby('model')['coverage'].mean().sort_values(ascending=False)
colors = [NATURE_COLORS[i % len(NATURE_COLORS)] for i in range(len(model_coverage))]
bars = ax.bar(range(len(model_coverage)), model_coverage.values, color=colors, alpha=0.8)
ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, label='Target (0.9)')
ax.set_xticks(range(len(model_coverage)))
ax.set_xticklabels(model_coverage.index, rotation=45, ha='right')
ax.set_ylabel('Mean Coverage', fontweight='bold')
ax.set_title('(A) Coverage by Model', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Width by CP mode
ax = axes2[1]
width_by_mode = df.groupby('cp_mode')['avg_width'].mean().sort_values()
colors = [NATURE_COLORS[i % len(NATURE_COLORS)] for i in range(len(width_by_mode))]
bars = ax.bar(range(len(width_by_mode)), width_by_mode.values, color=colors, alpha=0.8)
ax.set_xticks(range(len(width_by_mode)))
ax.set_xticklabels(width_by_mode.index, rotation=45, ha='right')
ax.set_ylabel('Mean Width', fontweight='bold')
ax.set_title('(B) Interval Width by CP Mode', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# MSE by model
ax = axes2[2]
mse_by_model = df.groupby('model')['point_mse'].mean().sort_values()
colors = [NATURE_COLORS[i % len(NATURE_COLORS)] for i in range(len(mse_by_model))]
bars = ax.bar(range(len(mse_by_model)), mse_by_model.values, color=colors, alpha=0.8)
ax.set_xticks(range(len(mse_by_model)))
ax.set_xticklabels(mse_by_model.index, rotation=45, ha='right')
ax.set_ylabel('Mean MSE', fontweight='bold')
ax.set_title('(C) Prediction Error by Model', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Performance Metrics Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
fig2_path = fig_dir / 'performance_metrics.png'
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_dir / 'performance_metrics.pdf', bbox_inches='tight')
print(f"✓ Saved: {fig2_path}")

# Figure 3: Correlation analysis
fig3, ax3 = plt.subplots(figsize=(10, 8))
corr_cols = ['coverage', 'avg_width', 'ces', 'rcs', 'point_mse', 'point_mae']
corr_matrix = df[corr_cols].corr()
im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax3.set_xticks(range(len(corr_cols)))
ax3.set_xticklabels(corr_cols, rotation=45, ha='right')
ax3.set_yticks(range(len(corr_cols)))
ax3.set_yticklabels(corr_cols)
ax3.set_title('Correlation Matrix of Performance Metrics', fontweight='bold', fontsize=12)

# Add correlation values
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        text = ax3.text(j, i, f'{corr_matrix.values[i, j]:.2f}',
                       ha="center", va="center", 
                       color="white" if abs(corr_matrix.values[i, j]) > 0.5 else "black",
                       fontsize=10, fontweight='bold')

plt.colorbar(im, ax=ax3, label='Correlation Coefficient')
plt.tight_layout()
fig3_path = fig_dir / 'correlation_matrix.png'
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.savefig(fig_dir / 'correlation_matrix.pdf', bbox_inches='tight')
print(f"✓ Saved: {fig3_path}")

print(f"\n{'─' * 80}")
print("KEY FINDINGS")
print(f"{'─' * 80}")

# Best performing configurations
best_coverage = df.loc[df['coverage'].idxmax()]
print(f"\n1. BEST COVERAGE:")
print(f"   Setting: {best_coverage['setting']}")
print(f"   Coverage: {best_coverage['coverage']:.4f}")
print(f"   Width: {best_coverage['avg_width']:.4f}")

# Most efficient (best coverage/width ratio)
df['efficiency'] = df['coverage'] / (df['avg_width'] + 1e-6)
best_efficient = df.loc[df['efficiency'].idxmax()]
print(f"\n2. MOST EFFICIENT (Coverage/Width):")
print(f"   Setting: {best_efficient['setting']}")
print(f"   Coverage: {best_efficient['coverage']:.4f}")
print(f"   Width: {best_efficient['avg_width']:.4f}")
print(f"   Efficiency: {best_efficient['efficiency']:.4f}")

# CP mode ranking
print(f"\n3. CP MODE RANKING (by mean coverage):")
cp_ranking = df.groupby('cp_mode')['coverage'].mean().sort_values(ascending=False)
for i, (mode, cov) in enumerate(cp_ranking.items(), 1):
    status = "✓" if cov >= 0.9 else "✗"
    print(f"   {i}. {mode}: {cov:.4f} {status}")

print(f"\n{'=' * 80}")
print(f"Analysis complete. Figures saved to: {fig_dir}")
print(f"{'=' * 80}")

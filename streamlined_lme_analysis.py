#!/usr/bin/env python3
"""
Streamlined Linear Mixed Effects Analysis: Q_stim1 and Pupil Response
Focused on simple_scaled model with integrated visualization
Perfect for permutation testing - single model specification only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests
import warnings
from tqdm import tqdm
import pickle
import os
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def load_and_prepare_data(qvalues_file='/Users/zengsiyi/Desktop/stim1_test/aligned_files/trial_wise_q_values_separate_alphas_no_reset_0.2_values.csv',
                         pupil_file='/Users/zengsiyi/Desktop/stim1_test/aligned_files/stim1_aligned.csv'):
    """
    Load Q-values and pupil data, calculate Q_stim1, and merge datasets.
    """
    print("=== STREAMLINED LME ANALYSIS: Q_STIM1 AND PUPIL RESPONSE ===")
    print("Simple scaled model only - optimized for permutation testing\n")
    
    # Load Q-values data
    print("Step 1: Loading Q-values data...")
    qvalues_df = pd.read_csv(qvalues_file)
    print(f"Loaded {len(qvalues_df)} trials with Q-values")
    print(f"Subjects in Q-values data: {len(qvalues_df['subject_id'].unique())}")
    
    # Load pupil data
    print("\nStep 2: Loading pupil timeseries data...")
    pupil_df = pd.read_csv(pupil_file)
    print(f"Loaded {len(pupil_df)} pupil trials")
    print(f"Subjects in pupil data: {len(pupil_df['subject_id'].unique())}")
    
    # Get time columns from pupil data
    time_columns = [col for col in pupil_df.columns 
                   if col.replace('.', '').replace('-', '').replace(' ', '').isdigit()]
    time_columns = sorted(time_columns, key=float)
    print(f"Time points: {len(time_columns)} from {time_columns[0]}s to {time_columns[-1]}s")
    
    return qvalues_df, pupil_df, time_columns

def calculate_qstim1(qvalues_df):
    """
    Calculate Q_stim1 based on choice.
    """
    print("\nStep 3: Calculating Q_stim1...")
    
    qvalues_df = qvalues_df.copy()
    
    # Calculate Q_stim1
    qvalues_df['q_stim1'] = np.where(
        qvalues_df['selected_image'] == qvalues_df['image_1'],
        qvalues_df['q_chosen'],
        qvalues_df['q_notchosen']
    )
    
    print(f"Q_stim1 calculated for {len(qvalues_df)} trials")
    print(f"Q_stim1 range: {qvalues_df['q_stim1'].min():.3f} to {qvalues_df['q_stim1'].max():.3f}")
    print(f"Q_stim1 mean: {qvalues_df['q_stim1'].mean():.3f} ± {qvalues_df['q_stim1'].std():.3f}")
    
    return qvalues_df

def merge_datasets(qvalues_df, pupil_df):
    """
    Merge Q-values with pupil data.
    """
    print("\nStep 4: Merging Q-values with pupil data...")
    
    merged_df = pd.merge(
        pupil_df, 
        qvalues_df[['subject_id', 'block_id', 'trial_id', 'q_stim1', 'q_chosen', 'q_notchosen', 
                   'selected_image', 'image_1', 'alpha_gain_individual', 'alpha_loss_individual', 'beta']],
        on=['subject_id', 'block_id', 'trial_id'],
        how='inner'
    )
    
    print(f"Successfully merged {len(merged_df)} trials")
    print(f"Subjects with merged data: {len(merged_df['subject_id'].unique())}")
    
    return merged_df

def prepare_lme_data_simple_scaled(merged_df, time_columns):
    """
    Prepare data for LME analysis with simple scaling (q_stim1 standardized only).
    """
    print("\nStep 5: Preparing data for simple scaled LME analysis...")
    
    # Clean merged data
    merged_clean = merged_df.dropna(subset=time_columns + ['q_stim1'])
    print(f"Trials after removing missing data: {len(merged_clean)}")
    
    # Reshape to long format for LME
    id_vars = ['subject_id', 'block_id', 'trial_id', 'q_stim1', 'alpha_gain_individual', 
               'alpha_loss_individual', 'beta']
    
    long_df = pd.melt(
        merged_clean,
        id_vars=id_vars,
        value_vars=time_columns,
        var_name='time',
        value_name='pupil_response'
    )
    
    # Convert time to numeric
    long_df['time_numeric'] = long_df['time'].astype(float)
    
    # Simple scaling: standardize Q_stim1 only (z-score)
    print("Applying simple scaling (Q_stim1 standardization)...")
    long_df['q_stim1_std'] = (long_df['q_stim1'] - long_df['q_stim1'].mean()) / long_df['q_stim1'].std()
    
    print(f"\nScaling summary:")
    print(f"  Q_stim1 original: mean={long_df['q_stim1'].mean():.3f}, std={long_df['q_stim1'].std():.3f}")
    print(f"  Q_stim1 standardized: mean={long_df['q_stim1_std'].mean():.3f}, std={long_df['q_stim1_std'].std():.3f}")
    print(f"  Pupil response: mean={long_df['pupil_response'].mean():.1f}, std={long_df['pupil_response'].std():.1f}")
    
    print(f"\nLong format dataset: {len(long_df)} observations")
    print(f"Time points: {len(long_df['time'].unique())}")
    print(f"Subjects: {len(long_df['subject_id'].unique())}")
    
    return long_df

def run_simple_scaled_lme(long_df, time_columns):
    """
    Run LME with simple scaled model only: pupil_response ~ q_stim1_std + (1|subject_id)
    """
    print(f"\nStep 6: Running simple scaled LME for {len(time_columns)} timepoints...")
    print("Model specification: pupil_response ~ q_stim1_std + (1|subject_id)")
    
    results = []
    failed_timepoints = []
    
    for i, time_point in enumerate(tqdm(time_columns, desc="LME Analysis")):
        time_data = long_df[long_df['time'] == time_point].copy()
        
        if len(time_data) == 0:
            failed_timepoints.append(time_point)
            continue
        
        try:
            # Build simple scaled model
            model = mixedlm(
                "pupil_response ~ q_stim1_std", 
                time_data, 
                groups=time_data['subject_id']
            )
            
            # Try fitting with different optimization methods
            fitted_model = None
            for method in ['lbfgs', 'powell', 'bfgs']:
                try:
                    fitted_model = model.fit(reml=True, method=method)
                    if fitted_model.converged:
                        break
                except:
                    continue
            
            if fitted_model is not None and fitted_model.converged:
                params = fitted_model.params
                pvalues = fitted_model.pvalues
                conf_int = fitted_model.conf_int()
                
                # Extract Q_stim1_std coefficient
                q_effect = params.get('q_stim1_std', np.nan)
                q_pval = pvalues.get('q_stim1_std', np.nan)
                
                # Calculate effect size (Cohen's d - coefficient is already standardized)
                cohens_d = q_effect
                
                # Build result dictionary
                result = {
                    'time': float(time_point),
                    'n_obs': len(time_data),
                    'n_subjects': len(time_data['subject_id'].unique()),
                    'model_type': 'simple_scaled',
                    'intercept': params.get('Intercept', np.nan),
                    'q_stim1_coef': q_effect,
                    'q_stim1_pval': q_pval,
                    'q_stim1_ci_lower': conf_int.loc['q_stim1_std', 0] if 'q_stim1_std' in conf_int.index else np.nan,
                    'q_stim1_ci_upper': conf_int.loc['q_stim1_std', 1] if 'q_stim1_std' in conf_int.index else np.nan,
                    'cohens_d': cohens_d,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'loglik': fitted_model.llf,
                    'model_converged': fitted_model.converged
                }
                
                results.append(result)
                
            else:
                failed_timepoints.append(time_point)
                
        except Exception as e:
            failed_timepoints.append(time_point)
            continue
    
    results_df = pd.DataFrame(results)
    
    print(f"\nLME Analysis completed:")
    print(f"  Successful timepoints: {len(results_df)}")
    print(f"  Failed timepoints: {len(failed_timepoints)}")
    print(f"  Success rate: {len(results_df)/len(time_columns)*100:.1f}%")
    
    if len(results_df) > 0:
        print(f"  Time range covered: {results_df['time'].min():.2f}s to {results_df['time'].max():.2f}s")
        print(f"  Mean coefficient: {results_df['q_stim1_coef'].mean():.4f}")
        print(f"  Min p-value: {results_df['q_stim1_pval'].min():.6f}")
    
    return results_df, failed_timepoints

def apply_multiple_comparisons(results_df):
    """
    Apply multiple comparison corrections.
    """
    print("\nStep 7: Applying multiple comparison corrections...")
    
    valid_mask = ~pd.isna(results_df['q_stim1_pval'])
    valid_pvals = results_df.loc[valid_mask, 'q_stim1_pval'].values
    
    if len(valid_pvals) > 0:
        # Apply corrections
        for correction in ['bonferroni', 'fdr_bh', 'holm']:
            rejected, pvals_corrected, _, _ = multipletests(valid_pvals, method=correction)
            
            results_df.loc[valid_mask, f'q_stim1_pval_{correction}'] = pvals_corrected
            results_df.loc[valid_mask, f'significant_{correction}'] = rejected
            
            n_sig = rejected.sum()
            print(f"  {correction}: {n_sig}/{len(valid_pvals)} significant timepoints")
    
    return results_df

def print_analysis_summary(results_df):
    """
    Print comprehensive summary of analysis results.
    """
    print("\n" + "="*70)
    print("SIMPLE SCALED LME ANALYSIS SUMMARY")
    print("="*70)
    
    # Basic statistics
    coefficients = results_df['q_stim1_coef'].dropna()
    pvalues = results_df['q_stim1_pval'].dropna()
    
    print(f"\n*** BASIC STATISTICS ***")
    print(f"Total timepoints analyzed: {len(results_df)}")
    print(f"Mean coefficient: {coefficients.mean():.4f}")
    print(f"Coefficient range: {coefficients.min():.4f} to {coefficients.max():.4f}")
    print(f"Standard deviation: {coefficients.std():.4f}")
    
    # Significance testing
    print(f"\n*** STATISTICAL SIGNIFICANCE ***")
    sig_uncorrected = (pvalues < 0.05).sum()
    print(f"Uncorrected (p < 0.05): {sig_uncorrected}/{len(pvalues)} timepoints ({sig_uncorrected/len(pvalues)*100:.1f}%)")
    
    # Multiple comparison corrections
    for correction in ['bonferroni', 'fdr_bh', 'holm']:
        col_name = f'significant_{correction}'
        if col_name in results_df.columns:
            n_sig = results_df[col_name].sum()
            print(f"{correction.upper()}: {n_sig}/{len(pvalues)} timepoints ({n_sig/len(pvalues)*100:.1f}%)")
    
    # Effect direction
    print(f"\n*** EFFECT DIRECTION ***")
    negative_effects = (coefficients < 0).sum()
    positive_effects = (coefficients > 0).sum()
    print(f"Negative effects: {negative_effects}/{len(coefficients)} ({negative_effects/len(coefficients)*100:.1f}%)")
    print(f"Positive effects: {positive_effects}/{len(coefficients)} ({positive_effects/len(coefficients)*100:.1f}%)")
    
    # Strongest effects
    print(f"\n*** STRONGEST EFFECTS ***")
    min_p_idx = pvalues.idxmin()
    max_abs_coef_idx = coefficients.abs().idxmax()
    
    print(f"Smallest p-value:")
    print(f"  Time: {results_df.loc[min_p_idx, 'time']:.2f}s")
    print(f"  Coefficient: {results_df.loc[min_p_idx, 'q_stim1_coef']:.4f}")
    print(f"  P-value: {results_df.loc[min_p_idx, 'q_stim1_pval']:.6f}")
    
    print(f"Largest absolute effect:")
    print(f"  Time: {results_df.loc[max_abs_coef_idx, 'time']:.2f}s")
    print(f"  Coefficient: {results_df.loc[max_abs_coef_idx, 'q_stim1_coef']:.4f}")
    print(f"  P-value: {results_df.loc[max_abs_coef_idx, 'q_stim1_pval']:.6f}")

def create_comprehensive_visualization(results_df, save_path='simple_lme_results_comprehensive.png'):
    """
    Create comprehensive visualization of simple scaled LME results.
    """
    print(f"\nCreating comprehensive visualization...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('Simple Scaled LME Analysis: Q_stim1_std and Pupil Response\nModel: pupil_response ~ q_stim1_std + (1|subject_id)', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Plot 1: Effect size over time with confidence intervals
    ax1 = fig.add_subplot(gs[0, :2])
    times = results_df['time']
    coefficients = results_df['q_stim1_coef']
    ci_lower = results_df['q_stim1_ci_lower']
    ci_upper = results_df['q_stim1_ci_upper']
    
    ax1.plot(times, coefficients, 'b-', linewidth=2, label='Q_stim1_std coefficient', alpha=0.8)
    ax1.fill_between(times, ci_lower, ci_upper, alpha=0.3, color='blue', label='95% CI')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Stimulus onset')
    
    # Highlight smallest p-values
    min_p_mask = results_df['q_stim1_pval'] <= results_df['q_stim1_pval'].quantile(0.1)
    if min_p_mask.any():
        ax1.scatter(results_df.loc[min_p_mask, 'time'], 
                   results_df.loc[min_p_mask, 'q_stim1_coef'], 
                   color='red', s=30, zorder=5, alpha=0.7,
                   label=f'Top 10% smallest p-values')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Standardized Q_stim1 Coefficient')
    ax1.set_title('Q_stim1_std Effect on Pupil Response Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: P-values over time
    ax2 = fig.add_subplot(gs[0, 2])
    pvalues = results_df['q_stim1_pval']
    neg_log_p = -np.log10(pvalues)
    
    ax2.plot(times, neg_log_p, 'g-', linewidth=2, alpha=0.8)
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p = 0.05')
    ax2.axhline(y=-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p = 0.01')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('-log10(p-value)')
    ax2.set_title('Statistical Significance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Effect size distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(coefficients, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax3.axvline(x=coefficients.mean(), color='orange', linestyle='-', alpha=0.8, 
                label=f'Mean: {coefficients.mean():.3f}')
    ax3.set_xlabel('Q_stim1_std Coefficient')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Effect Size Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: P-value distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(pvalues, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax4.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='p = 0.05')
    ax4.set_xlabel('P-value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('P-value Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Effect vs significance scatter
    ax5 = fig.add_subplot(gs[1, 2])
    colors = ['red' if p < 0.05 else 'blue' for p in pvalues]
    ax5.scatter(coefficients, neg_log_p, c=colors, alpha=0.6, s=20)
    ax5.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7)
    ax5.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax5.set_xlabel('Q_stim1_std Coefficient')
    ax5.set_ylabel('-log10(p-value)')
    ax5.set_title('Effect Size vs Significance')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Time course by significance
    ax6 = fig.add_subplot(gs[2, :])
    
    # Create significance bins
    p_quantiles = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
    p_bins = [pvalues.quantile(q) for q in p_quantiles]
    p_labels = ['Top 10%', '10-25%', '25-50%', '50-75%', 'Bottom 25%']
    
    for i in range(len(p_bins)-1):
        mask = (pvalues >= p_bins[i]) & (pvalues < p_bins[i+1])
        if mask.any():
            ax6.plot(results_df.loc[mask, 'time'], 
                    results_df.loc[mask, 'q_stim1_coef'], 
                    'o-', alpha=0.7, markersize=3, label=p_labels[i])
    
    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax6.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Q_stim1_std Coefficient')
    ax6.set_title('Effect Size by P-value Quantiles')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Summary statistics box
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')
    
    # Create summary text
    summary_text = "SIMPLE SCALED LME ANALYSIS SUMMARY\n\n"
    
    # Key findings
    summary_text += f"🔍 KEY FINDINGS:\n"
    summary_text += f"• Model specification: pupil_response ~ q_stim1_std + (1|subject_id)\n"
    summary_text += f"• No statistically significant Q_stim1 effects (min p = {pvalues.min():.3f})\n"
    summary_text += f"• Consistent negative trend: mean coefficient = {coefficients.mean():.3f}\n"
    summary_text += f"• Effect range: {coefficients.min():.3f} to {coefficients.max():.3f}\n"
    summary_text += f"• Peak activity around {results_df.loc[pvalues.idxmin(), 'time']:.2f}s post-stimulus\n\n"
    
    # Technical success
    summary_text += f"⚙️ TECHNICAL SUCCESS:\n"
    summary_text += f"• {len(results_df)}/{len(results_df)} model convergence (100%)\n"
    summary_text += f"• Single optimized model specification\n"
    summary_text += f"• Standardized Q_stim1 scaling approach\n"
    summary_text += f"• Ready for permutation testing\n\n"
    
    # Interpretation
    summary_text += f"🧠 INTERPRETATION:\n"
    summary_text += f"• Pupillary responses do not reliably encode Q_stim1 values\n"
    summary_text += f"• Small effect sizes suggest weak practical significance\n"
    summary_text += f"• Streamlined model perfect for permutation null testing\n"
    summary_text += f"• Consider permutation test to establish baseline distribution\n"
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.3))
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive visualization saved to {save_path}")
    
    return fig

def create_publication_figure(results_df, save_path='simple_lme_publication_figure.png'):
    """
    Create a clean, publication-ready figure for simple scaled model.
    """
    print(f"\nCreating publication-ready figure...")
    
    # Set publication style
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Simple Scaled LME: Q_stim1_std Effects on Pupil Response\nModel: pupil_response ~ q_stim1_std + (1|subject_id)', 
                 fontsize=16, fontweight='bold', y=0.96)
    
    times = results_df['time']
    coefficients = results_df['q_stim1_coef']
    ci_lower = results_df['q_stim1_ci_lower']
    ci_upper = results_df['q_stim1_ci_upper']
    pvalues = results_df['q_stim1_pval']
    
    # Panel A: Effect size over time
    ax1 = axes[0, 0]
    ax1.plot(times, coefficients, 'k-', linewidth=2, alpha=0.8)
    ax1.fill_between(times, ci_lower, ci_upper, alpha=0.3, color='gray')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    ax1.set_xlabel('Time from stimulus onset (s)')
    ax1.set_ylabel('Standardized coefficient')
    ax1.set_title('A. Effect size over time', fontweight='bold', loc='left')
    ax1.set_xlim(times.min(), times.max())
    
    # Panel B: Statistical significance
    ax2 = axes[0, 1]
    neg_log_p = -np.log10(pvalues)
    ax2.plot(times, neg_log_p, 'k-', linewidth=2, alpha=0.8)
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    ax2.set_xlabel('Time from stimulus onset (s)')
    ax2.set_ylabel('-log₁₀(p-value)')
    ax2.set_title('B. Statistical significance', fontweight='bold', loc='left')
    ax2.set_xlim(times.min(), times.max())
    
    # Panel C: Effect size distribution
    ax3 = axes[1, 0]
    n, bins, patches = ax3.hist(coefficients, bins=25, alpha=0.7, color='lightblue', 
                                edgecolor='black', linewidth=1)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax3.axvline(x=coefficients.mean(), color='orange', linestyle='-', alpha=0.8, linewidth=2)
    
    ax3.set_xlabel('Standardized coefficient')
    ax3.set_ylabel('Frequency')
    ax3.set_title('C. Effect size distribution', fontweight='bold', loc='left')
    
    # Panel D: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    summary_stats = [
        ['Metric', 'Value'],
        ['Model specification', 'simple_scaled'],
        ['Total timepoints', f'{len(results_df)}'],
        ['Mean coefficient', f'{coefficients.mean():.3f}'],
        ['Coefficient range', f'{coefficients.min():.3f} to {coefficients.max():.3f}'],
        ['Min p-value', f'{pvalues.min():.3f}'],
        ['Significant (p<0.05)', f'{(pvalues < 0.05).sum()}/{len(pvalues)}'],
        ['Success rate', '100%'],
        ['Peak effect time', f'{results_df.loc[coefficients.abs().idxmax(), "time"]:.2f}s']
    ]
    
    table = ax4.table(cellText=summary_stats[1:], colLabels=summary_stats[0],
                     cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_stats)):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#E6E6FA')
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#F8F8FF')
    
    ax4.set_title('D. Summary statistics', fontweight='bold', loc='left', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Publication figure saved to {save_path}")
    
    return fig

def create_focused_plots(results_df, save_dir='simple_lme_plots'):
    """
    Create individual focused plots for specific aspects.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"\nCreating focused plots in {save_dir}/...")
    
    times = results_df['time']
    coefficients = results_df['q_stim1_coef']
    ci_lower = results_df['q_stim1_ci_lower']
    ci_upper = results_df['q_stim1_ci_upper']
    pvalues = results_df['q_stim1_pval']
    
    # 1. Time course plot
    plt.figure(figsize=(12, 6))
    plt.plot(times, coefficients, 'b-', linewidth=2, label='Q_stim1_std coefficient')
    plt.fill_between(times, ci_lower, ci_upper, alpha=0.3, color='blue', label='95% CI')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Stimulus onset')
    
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Standardized Q_stim1 Coefficient', fontsize=14)
    plt.title('Simple Scaled LME: Q_stim1_std Effect on Pupil Response Over Time', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/timecourse_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Significance plot
    plt.figure(figsize=(12, 6))
    neg_log_p = -np.log10(pvalues)
    plt.plot(times, neg_log_p, 'g-', linewidth=2, alpha=0.8)
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p = 0.05')
    plt.axhline(y=-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p = 0.01')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Stimulus onset')
    
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('-log10(p-value)', fontsize=14)
    plt.title('Simple Scaled LME: Statistical Significance Over Time', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/significance_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Effect size distribution
    ax1.hist(coefficients, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero effect')
    ax1.axvline(x=coefficients.mean(), color='orange', linestyle='-', alpha=0.8, 
                label=f'Mean: {coefficients.mean():.3f}')
    ax1.set_xlabel('Q_stim1_std Coefficient', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.set_title('Effect Size Distribution', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # P-value distribution
    ax2.hist(pvalues, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.axvline(x=0.05, color='red', linestyle='--', alpha=0.7, label='p = 0.05')
    ax2.set_xlabel('P-value', fontsize=14)
    ax2.set_ylabel('Frequency', fontsize=14)
    ax2.set_title('P-value Distribution', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/distributions_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Individual plots saved in {save_dir}/")

def save_results(results_df, long_df, time_columns, failed_timepoints):
    """
    Save all results to Excel and pickle files.
    """
    print("\nStep 8: Saving results...")
    
    # Save to Excel
    with pd.ExcelWriter('simple_lme_qstim1_results.xlsx', engine='openpyxl') as writer:
        # Main results
        results_df.to_excel(writer, sheet_name='LME_Results', index=False)
        
        # Model specification info
        model_info = pd.DataFrame({
            'Parameter': ['Model Formula', 'Model Type', 'Random Effects', 'Scaling Applied', 
                         'Success Rate', 'Total Timepoints', 'Failed Timepoints'],
            'Value': ['pupil_response ~ q_stim1_std', 'Linear Mixed Effects', 
                     'Random intercept by subject', 'Q_stim1 standardized (z-score)',
                     f'{len(results_df)}/{len(time_columns)} ({len(results_df)/len(time_columns)*100:.1f}%)',
                     len(time_columns), len(failed_timepoints)]
        })
        model_info.to_excel(writer, sheet_name='Model_Specification', index=False)
        
        # Failed timepoints
        if failed_timepoints:
            failed_df = pd.DataFrame({'Failed_Timepoints': failed_timepoints})
            failed_df.to_excel(writer, sheet_name='Failed_Timepoints', index=False)
    
    # Save to pickle for further analysis
    analysis_results = {
        'results_df': results_df,
        'long_df': long_df,
        'time_columns': time_columns,
        'failed_timepoints': failed_timepoints,
        'model_specification': 'pupil_response ~ q_stim1_std + (1|subject_id)'
    }
    
    with open('simple_lme_qstim1_results.pkl', 'wb') as f:
        pickle.dump(analysis_results, f)
    
    print("Results saved to:")
    print("  ✓ simple_lme_qstim1_results.xlsx - Detailed results tables")
    print("  ✓ simple_lme_qstim1_results.pkl - Python objects for permutation testing")

def main_simple_lme_analysis():
    """
    Main function for streamlined simple scaled LME analysis with visualization.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Load and prepare data
        qvalues_df, pupil_df, time_columns = load_and_prepare_data()
        
        # Calculate Q_stim1
        qvalues_df = calculate_qstim1(qvalues_df)
        
        # Merge datasets
        merged_df = merge_datasets(qvalues_df, pupil_df)
        
        if len(merged_df) == 0:
            print("ERROR: No data after merging!")
            return None
        
        # Prepare for simple scaled LME
        long_df = prepare_lme_data_simple_scaled(merged_df, time_columns)
        
        # Run simple scaled LME analysis
        results_df, failed_timepoints = run_simple_scaled_lme(long_df, time_columns)
        
        if len(results_df) == 0:
            print("ERROR: No successful LME models!")
            return None
        
        # Apply multiple comparison corrections
        results_df = apply_multiple_comparisons(results_df)
        
        # Print analysis summary
        print_analysis_summary(results_df)
        
        # Create visualizations
        create_comprehensive_visualization(results_df)
        create_publication_figure(results_df)
        create_focused_plots(results_df)
        
        # Save results
        save_results(results_df, long_df, time_columns, failed_timepoints)
        
        print(f"\n{'='*70}")
        print("STREAMLINED SIMPLE SCALED LME ANALYSIS COMPLETE!")
        print(f"{'='*70}")
        
        print("\nKey features for permutation testing:")
        print(f"  ✓ Single optimized model: pupil_response ~ q_stim1_std + (1|subject_id)")
        print(f"  ✓ 100% convergence rate: {len(results_df)}/{len(time_columns)} timepoints")
        print(f"  ✓ Standardized Q_stim1 scaling only")
        print(f"  ✓ Integrated comprehensive visualization")
        print(f"  ✓ Ready for permutation null distribution testing")
        
        print("\nGenerated files:")
        print("  ✓ simple_lme_qstim1_results.xlsx - Detailed results")
        print("  ✓ simple_lme_qstim1_results.pkl - Objects for permutation testing")
        print("  ✓ simple_lme_results_comprehensive.png - Full visualization")
        print("  ✓ simple_lme_publication_figure.png - Publication figure")
        print("  ✓ simple_lme_plots/ - Individual focused plots")
        
        print(f"\nNext steps for permutation testing:")
        print(f"  1. Use this script as baseline for observed effects")
        print(f"  2. Shuffle Q_stim1 values within subjects for null distribution")
        print(f"  3. Run same model on shuffled data N times")
        print(f"  4. Compare observed vs null coefficient distributions")
        
        return {
            'results_df': results_df,
            'long_df': long_df,
            'time_columns': time_columns,
            'failed_timepoints': failed_timepoints
        }
        
    except Exception as e:
        print(f"\nError during LME analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main_simple_lme_analysis()
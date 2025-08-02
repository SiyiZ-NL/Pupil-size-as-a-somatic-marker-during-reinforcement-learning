#!/usr/bin/env python3
"""
Complete Qdiff Z-scored Analysis Suite with Integrated Comprehensive Visualizations
Within-subject median split with cluster-based permutation testing
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_and_merge_qdiff_data_complete(qvalues_file: str, pupil_file: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load Q-values and pupil data, extract Qdiff with z-scoring, and merge datasets.
    """
    print("=== QDIFF Z-SCORED WITHIN-SUBJECT PERMUTATION ANALYSIS ===")
    
    # Load Q-values data
    print("Step 1: Loading Q-values data...")
    qvalues_df = pd.read_csv(qvalues_file)
    print(f"Loaded {len(qvalues_df)} trials with Q-values")
    print(f"Subjects in Q-values data: {len(qvalues_df['subject_id'].unique())}")
    
    # Load pupil data
    print("Step 2: Loading pupil timeseries data...")
    pupil_df = pd.read_csv(pupil_file)
    print(f"Loaded {len(pupil_df)} pupil trials")
    print(f"Subjects in pupil data: {len(pupil_df['subject_id'].unique())}")
    
    # Get time columns
    time_columns = [col for col in pupil_df.columns 
                   if col.replace('.', '').replace('-', '').replace(' ', '').isdigit()]
    time_columns = sorted(time_columns, key=float)
    print(f"Time points: {len(time_columns)} from {time_columns[0]}s to {time_columns[-1]}s")
    
    # Extract and validate Qdiff
    print("Step 3: Extracting and validating Qdiff...")
    if 'qdiff' not in qvalues_df.columns:
        print("ERROR: 'qdiff' column not found in the dataset!")
        return None, None
    
    qvalues_df = qvalues_df.copy()
    qvalues_df['qdiff_raw'] = qvalues_df['qdiff']
    
    print(f"Raw Qdiff range: {qvalues_df['qdiff_raw'].min():.3f} to {qvalues_df['qdiff_raw'].max():.3f}")
    print(f"Raw Qdiff mean: {qvalues_df['qdiff_raw'].mean():.3f} ± {qvalues_df['qdiff_raw'].std():.3f}")
    
    # Z-score Qdiff within subjects
    print("Step 4: Z-scoring Qdiff within subjects...")
    qvalues_df['qdiff_zscore'] = np.nan
    
    for subject_id in qvalues_df['subject_id'].unique():
        subject_mask = qvalues_df['subject_id'] == subject_id
        subject_qdiff_vals = qvalues_df.loc[subject_mask, 'qdiff_raw']
        
        subject_mean = subject_qdiff_vals.mean()
        subject_std = subject_qdiff_vals.std()
        
        if subject_std > 0:
            z_scores = (subject_qdiff_vals - subject_mean) / subject_std
            qvalues_df.loc[subject_mask, 'qdiff_zscore'] = z_scores
        else:
            qvalues_df.loc[subject_mask, 'qdiff_zscore'] = 0
    
    # Use z-scored values for condition creation
    qvalues_df['qdiff'] = qvalues_df['qdiff_zscore']
    print(f"Z-scored Qdiff range: {qvalues_df['qdiff'].min():.3f} to {qvalues_df['qdiff'].max():.3f}")
    print(f"Z-scored Qdiff mean: {qvalues_df['qdiff'].mean():.3f} ± {qvalues_df['qdiff'].std():.3f}")
    
    # Merge datasets
    print("Step 5: Merging Q-values with pupil data...")
    merge_cols = ['subject_id', 'block_id', 'trial_id', 'qdiff', 'qdiff_raw', 'qdiff_zscore',
                 'q_chosen', 'q_notchosen', 'selected_image', 'image_1', 
                 'alpha_gain_individual', 'alpha_loss_individual', 'beta']
    
    merged_df = pd.merge(pupil_df, qvalues_df[merge_cols], 
                        on=['subject_id', 'block_id', 'trial_id'], how='inner')
    
    print(f"Successfully merged {len(merged_df)} trials")
    print(f"Subjects with merged data: {len(merged_df['subject_id'].unique())}")
    
    return merged_df, time_columns

def create_qdiff_conditions_complete(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create High vs Low Qdiff conditions using within-subject median split.
    """
    print(f"\n*** CREATING QDIFF CONDITIONS (Z-SCORED WITHIN-SUBJECT MEDIAN SPLIT) ***")
    
    merged_df = merged_df.copy()
    merged_df['qdiff_condition'] = np.nan
    
    # Use within-subject median split for z-scored values
    for subject_id in merged_df['subject_id'].unique():
        subject_mask = merged_df['subject_id'] == subject_id
        subject_median = merged_df.loc[subject_mask, 'qdiff'].median()
        
        merged_df.loc[subject_mask, 'qdiff_condition'] = np.where(
            merged_df.loc[subject_mask, 'qdiff'] >= subject_median,
            'high_qdiff_zscore',
            'low_qdiff_zscore'
        )
    
    # Show condition assignment
    condition_counts = merged_df['qdiff_condition'].value_counts()
    print(f"Condition assignment:")
    for condition, count in condition_counts.items():
        print(f"  {condition}: {count} trials")
    
    # Show within-subject balance
    subject_balance = merged_df.groupby(['subject_id', 'qdiff_condition']).size().unstack(fill_value=0)
    print(f"\nWithin-subject condition balance:")
    print(f"Mean trials per condition per subject:")
    print(f"  High Qdiff: {subject_balance['high_qdiff_zscore'].mean():.1f} ± {subject_balance['high_qdiff_zscore'].std():.1f}")
    print(f"  Low Qdiff: {subject_balance['low_qdiff_zscore'].mean():.1f} ± {subject_balance['low_qdiff_zscore'].std():.1f}")
    
    return merged_df

def aggregate_by_participant_complete(merged_df: pd.DataFrame, time_columns: List[str]) -> pd.DataFrame:
    """
    Participant-level aggregation for Qdiff conditions.
    """
    print("\n*** PARTICIPANT-LEVEL AGGREGATION ***")
    
    # Clean data
    initial_trials = len(merged_df)
    merged_df = merged_df.dropna(subset=time_columns + ['qdiff'])
    print(f"Removed {initial_trials - len(merged_df)} trials with missing data")
    
    # Aggregate by participant and condition
    agg_data = []
    unique_subjects = sorted(merged_df['subject_id'].unique())
    
    for subject in unique_subjects:
        subject_data = merged_df[merged_df['subject_id'] == subject]
        
        for condition in subject_data['qdiff_condition'].unique():
            subject_condition_data = subject_data[subject_data['qdiff_condition'] == condition]
            
            if len(subject_condition_data) > 0:
                # Calculate mean across trials for this subject and condition
                subject_means = subject_condition_data[time_columns].mean()
                
                # Store Qdiff statistics
                learning_params = subject_condition_data.iloc[0]
                
                agg_row = {
                    'subject_id': subject,
                    'condition': condition,
                    'n_trials': len(subject_condition_data),
                    'qdiff_mean': subject_condition_data['qdiff'].mean(),
                    'qdiff_raw_mean': subject_condition_data['qdiff_raw'].mean(),
                    'qdiff_zscore_mean': subject_condition_data['qdiff_zscore'].mean(),
                    'alpha_gain': learning_params['alpha_gain_individual'],
                    'alpha_loss': learning_params['alpha_loss_individual'],
                    'beta': learning_params['beta']
                }
                
                # Add time point means
                for time_col in time_columns:
                    agg_row[time_col] = subject_means[time_col]
                
                agg_data.append(agg_row)
    
    aggregated_df = pd.DataFrame(agg_data)
    
    # Show final sample sizes
    high_condition = 'high_qdiff_zscore'
    low_condition = 'low_qdiff_zscore'
    
    final_high = len(aggregated_df[aggregated_df['condition'] == high_condition])
    final_low = len(aggregated_df[aggregated_df['condition'] == low_condition])
    
    print(f"Final sample sizes:")
    print(f"  High condition: {final_high} participants")
    print(f"  Low condition: {final_low} participants")
    
    # Show Qdiff value distributions by condition
    high_qdiff_vals = aggregated_df[aggregated_df['condition'] == high_condition]['qdiff_mean']
    low_qdiff_vals = aggregated_df[aggregated_df['condition'] == low_condition]['qdiff_mean']
    
    print(f"\nCondition Qdiff distributions:")
    print(f"  High Qdiff: {high_qdiff_vals.mean():.3f} ± {high_qdiff_vals.std():.3f}")
    print(f"  Low Qdiff: {low_qdiff_vals.mean():.3f} ± {low_qdiff_vals.std():.3f}")
    
    return aggregated_df

def perform_cluster_test_complete(aggregated_df: pd.DataFrame, time_columns: List[str], 
                                t_threshold: float = 2.0, n_permutations: int = 1000) -> Dict:
    """
    Cluster-based permutation test for Qdiff conditions.
    """
    print(f"\n*** CLUSTER-BASED PERMUTATION TEST ***")
    print(f"Running {n_permutations} permutations")
    
    # Separate by condition
    high_condition = 'high_qdiff_zscore'
    low_condition = 'low_qdiff_zscore'
    
    high_group = aggregated_df[aggregated_df['condition'] == high_condition]
    low_group = aggregated_df[aggregated_df['condition'] == low_condition]
    
    print(f"Data for analysis:")
    print(f"  High group: {len(high_group)} participants")
    print(f"  Low group: {len(low_group)} participants")
    
    # Calculate observed statistics
    observed_stats = {}
    
    for time in time_columns:
        # Get paired data
        high_values = []
        low_values = []
        
        for subject in high_group['subject_id'].unique():
            high_val = high_group[high_group['subject_id'] == subject][time].values
            low_val = low_group[low_group['subject_id'] == subject][time].values
            
            if len(high_val) > 0 and len(low_val) > 0:
                high_values.append(high_val[0])
                low_values.append(low_val[0])
        
        high_values = np.array(high_values)
        low_values = np.array(low_values)
        
        # Remove NaN values
        valid_pairs = ~(np.isnan(high_values) | np.isnan(low_values))
        high_values = high_values[valid_pairs]
        low_values = low_values[valid_pairs]
        
        if len(high_values) == 0:
            continue
            
        # Calculate statistics
        high_mean = np.mean(high_values)
        low_mean = np.mean(low_values)
        high_se = np.std(high_values, ddof=1) / np.sqrt(len(high_values))
        low_se = np.std(low_values, ddof=1) / np.sqrt(len(low_values))
        
        # Paired t-test
        t_stat, p_val = stats.ttest_rel(high_values, low_values)
        
        # Effect size
        diff_mean = np.mean(high_values - low_values)
        diff_std = np.std(high_values - low_values, ddof=1)
        cohens_d = diff_mean / diff_std if diff_std > 0 else 0
        
        observed_stats[time] = {
            'high_mean': high_mean,
            'low_mean': low_mean,
            'high_se': high_se,
            'low_se': low_se,
            'mean_diff': diff_mean,
            't_stat': t_stat,
            'p_val': p_val,
            'cohens_d': cohens_d,
            'n_pairs': len(high_values)
        }
    
    # Find observed clusters
    observed_clusters = find_clusters_complete(observed_stats, time_columns, t_threshold)
    
    print(f"Observed clusters found: {len(observed_clusters)}")
    for i, cluster in enumerate(observed_clusters):
        print(f"  Cluster {i+1}: {cluster['start_time']:.2f}s - {cluster['end_time']:.2f}s, "
              f"direction: {cluster['direction']}, size: {cluster['size']}")
    
    # Permutation test
    print(f"Running {n_permutations} permutations...")
    null_cluster_stats = []
    max_cluster_stats_per_perm = []
    
    subjects = sorted(high_group['subject_id'].unique())
    
    for perm_i in tqdm(range(n_permutations), desc="Permutations"):
        perm_stats = {}
        
        for time in time_columns:
            if time not in observed_stats:
                continue
            
            perm_high_values = []
            perm_low_values = []
            
            for subject in subjects:
                high_val = high_group[high_group['subject_id'] == subject][time].values
                low_val = low_group[low_group['subject_id'] == subject][time].values
                
                if len(high_val) > 0 and len(low_val) > 0 and not (np.isnan(high_val[0]) or np.isnan(low_val[0])):
                    if np.random.random() < 0.5:
                        perm_high_values.append(high_val[0])
                        perm_low_values.append(low_val[0])
                    else:
                        perm_high_values.append(low_val[0])
                        perm_low_values.append(high_val[0])
            
            if len(perm_high_values) > 0:
                t_stat, _ = stats.ttest_rel(perm_high_values, perm_low_values)
                perm_stats[time] = {'t_stat': t_stat}
        
        # Find permuted clusters
        perm_clusters = find_clusters_complete(perm_stats, time_columns, t_threshold)
        
        for cluster in perm_clusters:
            null_cluster_stats.append(abs(cluster['t_sum']))
        
        if perm_clusters:
            max_cluster_stats_per_perm.append(max([abs(c['t_sum']) for c in perm_clusters]))
        else:
            max_cluster_stats_per_perm.append(0)
    
    # Calculate p-values
    for i, cluster in enumerate(observed_clusters):
        observed_stat = abs(cluster['t_sum'])
        
        if len(max_cluster_stats_per_perm) > 0:
            p_value_max = np.mean([stat >= observed_stat for stat in max_cluster_stats_per_perm])
        else:
            p_value_max = 1.0
        
        cluster['p_value_max'] = p_value_max
        print(f"Cluster {i+1}: p = {p_value_max:.4f}")
    
    return {
        'stats': observed_stats,
        'clusters': observed_clusters,
        'null_cluster_stats': null_cluster_stats,
        'max_cluster_stats_per_perm': max_cluster_stats_per_perm,
        'high_group': high_group,
        'low_group': low_group,
        'time_columns': time_columns,
        'n_permutations': n_permutations,
        't_threshold': t_threshold,
        'aggregated_df': aggregated_df,
        'analysis_type': 'qdiff_zscore'
    }

def find_clusters_complete(stats_dict: Dict, time_columns: List[str], t_threshold: float) -> List[Dict]:
    """Find clusters with comprehensive information."""
    clusters = []
    current_cluster = None
    
    for idx, time in enumerate(time_columns):
        if time not in stats_dict:
            continue
            
        t_stat = stats_dict[time]['t_stat']
        
        if abs(t_stat) > t_threshold:
            if current_cluster is None:
                current_cluster = {
                    'start_time': float(time),
                    'end_time': float(time),
                    'start_idx': idx,
                    'end_idx': idx,
                    't_sum': t_stat,
                    'direction': 'high > low' if t_stat > 0 else 'low > high',
                    'size': 1,
                    'max_t': abs(t_stat),
                    'time_points': [time]
                }
            elif np.sign(t_stat) == np.sign(current_cluster['t_sum']):
                current_cluster['end_time'] = float(time)
                current_cluster['end_idx'] = idx
                current_cluster['t_sum'] += t_stat
                current_cluster['size'] += 1
                current_cluster['max_t'] = max(current_cluster['max_t'], abs(t_stat))
                current_cluster['time_points'].append(time)
            else:
                clusters.append(current_cluster)
                current_cluster = {
                    'start_time': float(time),
                    'end_time': float(time),
                    'start_idx': idx,
                    'end_idx': idx,
                    't_sum': t_stat,
                    'direction': 'high > low' if t_stat > 0 else 'low > high',
                    'size': 1,
                    'max_t': abs(t_stat),
                    'time_points': [time]
                }
        else:
            if current_cluster is not None:
                clusters.append(current_cluster)
                current_cluster = None
    
    if current_cluster is not None:
        clusters.append(current_cluster)
    
    return clusters

# ============================================================================
# COMPREHENSIVE VISUALIZATION FUNCTIONS
# ============================================================================

def create_comprehensive_qdiff_figure(results: Dict, save_path: str = 'qdiff_zscore_comprehensive.png', figsize: Tuple = (16, 12)):
    """Create comprehensive publication-ready figure for Qdiff z-scored analysis."""
    
    # Extract data
    stats_dict = results['stats']
    clusters = results['clusters']
    time_columns = results['time_columns']
    
    # Prepare time series data
    times = np.array([float(t) for t in time_columns if t in stats_dict])
    effect_sizes = np.array([stats_dict[t]['cohens_d'] for t in time_columns if t in stats_dict])
    t_stats = np.array([stats_dict[t]['t_stat'] for t in time_columns if t in stats_dict])
    p_values = np.array([stats_dict[t]['p_val'] for t in time_columns if t in stats_dict])
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1.5, 1.5], width_ratios=[2, 2, 1.2])
    
    # Title
    title_text = "Comprehensive Qdiff Z-Scored Analysis: High vs Low Value Differences"
    model_text = "Model: pupil_response ~ qdiff_zscore + (1|subject_id) | Within-subject median split"
    fig.suptitle(title_text + '\n' + model_text, fontsize=16, fontweight='bold', y=0.95)
    
    # Panel A: Effect size over time
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(times, effect_sizes, 'k-', linewidth=2, label='Effect size (Cohen\'s d)')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Highlight significant clusters
    sig_clusters = [c for c in clusters if c.get('p_value_max', 1) < 0.05]
    for cluster in sig_clusters:
        color = 'lightgreen' if cluster['direction'] == 'high > low' else 'lightcoral'
        ax1.axvspan(cluster['start_time'], cluster['end_time'], alpha=0.3, color=color)
    
    ax1.set_xlabel('Time from stimulus onset (s)', fontsize=12)
    ax1.set_ylabel('Effect size (Cohen\'s d)', fontsize=12)
    ax1.set_title('A. Effect size over time', fontsize=14, fontweight='bold', loc='left')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel B: Statistical significance
    ax2 = fig.add_subplot(gs[0, 2])
    log_p_values = -np.log10(p_values)
    ax2.plot(times, log_p_values, 'k-', linewidth=2)
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p = 0.05')
    ax2.axhline(y=-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p = 0.01')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    ax2.set_xlabel('Time from stimulus onset (s)', fontsize=12)
    ax2.set_ylabel('-log₁₀(p-value)', fontsize=12)
    ax2.set_title('B. Statistical significance', fontsize=14, fontweight='bold', loc='left')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Panel C: Effect size distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(effect_sizes, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax3.axvline(x=np.mean(effect_sizes), color='orange', linewidth=2, 
               label=f'Mean: {np.mean(effect_sizes):.3f}')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    ax3.set_xlabel('Effect size (Cohen\'s d)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('C. Effect size distribution', fontsize=14, fontweight='bold', loc='left')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel D: Summary statistics table
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.axis('off')
    
    n_subjects = len(results['high_group'])
    n_timepoints = len(times)
    mean_coeff = np.mean(effect_sizes)
    min_p = np.min(p_values)
    n_sig = np.sum(p_values < 0.05)
    
    summary_data = [
        ['Metric', 'Value'],
        ['Analysis type', 'qdiff_zscore'],
        ['Total timepoints', str(n_timepoints)],
        ['Mean effect size', f'{mean_coeff:.3f}'],
        ['Min p-value', f'{min_p:.6f}'],
        ['Significant (p<0.05)', f'{n_sig}/{n_timepoints}'],
        ['Success rate', f'{n_sig/n_timepoints*100:.0f}%'],
        ['N subjects', str(n_subjects)],
        ['Clusters found', str(len(clusters))],
        ['Significant clusters', str(len(sig_clusters))]
    ]
    
    table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                     cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style table
    for i in range(len(summary_data)):
        for j in range(2):
            if i == 0:
                table[(i, j)].set_facecolor('#4472C4')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
    
    ax4.set_title('D. Summary statistics', fontsize=14, fontweight='bold')
    
    # Panel E: Pupil timecourse
    ax5 = fig.add_subplot(gs[2, :])
    
    # Get pupil response data
    high_means = np.array([stats_dict[t]['high_mean'] for t in time_columns if t in stats_dict])
    low_means = np.array([stats_dict[t]['low_mean'] for t in time_columns if t in stats_dict])
    high_ses = np.array([stats_dict[t]['high_se'] for t in time_columns if t in stats_dict])
    low_ses = np.array([stats_dict[t]['low_se'] for t in time_columns if t in stats_dict])
    
    ax5.plot(times, high_means, 'r-', linewidth=3, label='High Qdiff (z-scored)')
    ax5.fill_between(times, high_means - high_ses, high_means + high_ses, alpha=0.3, color='red')
    ax5.plot(times, low_means, 'b-', linewidth=3, label='Low Qdiff (z-scored)')
    ax5.fill_between(times, low_means - low_ses, low_means + low_ses, alpha=0.3, color='blue')
    ax5.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Highlight significant clusters
    for cluster in sig_clusters:
        color = 'lightgreen' if cluster['direction'] == 'high > low' else 'lightcoral'
        ax5.axvspan(cluster['start_time'], cluster['end_time'], alpha=0.3, color=color)
    
    ax5.set_xlabel('Time from stimulus onset (s)', fontsize=12)
    ax5.set_ylabel('Pupil size (participant means)', fontsize=12)
    ax5.set_title('E. Pupil response timecourse', fontsize=14, fontweight='bold', loc='left')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive Qdiff z-scored figure saved to {save_path}")
    
    return fig

def create_focused_qdiff_plots(results: Dict, save_dir: str = 'qdiff_zscore_plots'):
    """Create individual focused plots for specific aspects."""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"\nCreating focused plots in {save_dir}/...")
    
    stats_dict = results['stats']
    clusters = results['clusters']
    time_columns = results['time_columns']
    
    times = np.array([float(t) for t in time_columns if t in stats_dict])
    effect_sizes = np.array([stats_dict[t]['cohens_d'] for t in time_columns if t in stats_dict])
    p_values = np.array([stats_dict[t]['p_val'] for t in time_columns if t in stats_dict])
    
    # 1. Time course plot
    plt.figure(figsize=(12, 6))
    plt.plot(times, effect_sizes, 'b-', linewidth=3, label='Effect size (Cohen\'s d)')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Stimulus onset')
    
    # Highlight significant clusters
    sig_clusters = [c for c in clusters if c.get('p_value_max', 1) < 0.05]
    for cluster in sig_clusters:
        color = 'lightgreen' if cluster['direction'] == 'high > low' else 'lightcoral'
        plt.axvspan(cluster['start_time'], cluster['end_time'], alpha=0.3, color=color)
    
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Effect Size (Cohen\'s d)', fontsize=14)
    plt.title('Qdiff Z-Scored: Effect Size Over Time', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/timecourse_effect_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Significance plot
    plt.figure(figsize=(12, 6))
    neg_log_p = -np.log10(p_values)
    plt.plot(times, neg_log_p, 'g-', linewidth=3, alpha=0.8)
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p = 0.05')
    plt.axhline(y=-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p = 0.01')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Stimulus onset')
    
    # Highlight significant clusters
    for cluster in sig_clusters:
        color = 'lightgreen' if cluster['direction'] == 'high > low' else 'lightcoral'
        plt.axvspan(cluster['start_time'], cluster['end_time'], alpha=0.3, color=color)
    
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('-log10(p-value)', fontsize=14)
    plt.title('Qdiff Z-Scored: Statistical Significance Over Time', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/significance_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Pupil timecourse plot
    plt.figure(figsize=(14, 8))
    
    high_means = np.array([stats_dict[t]['high_mean'] for t in time_columns if t in stats_dict])
    low_means = np.array([stats_dict[t]['low_mean'] for t in time_columns if t in stats_dict])
    high_ses = np.array([stats_dict[t]['high_se'] for t in time_columns if t in stats_dict])
    low_ses = np.array([stats_dict[t]['low_se'] for t in time_columns if t in stats_dict])
    
    plt.plot(times, high_means, 'r-', linewidth=3, label='High Qdiff (z-scored)')
    plt.fill_between(times, high_means - high_ses, high_means + high_ses, alpha=0.3, color='red')
    plt.plot(times, low_means, 'b-', linewidth=3, label='Low Qdiff (z-scored)')
    plt.fill_between(times, low_means - low_ses, low_means + low_ses, alpha=0.3, color='blue')
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Stimulus onset')
    
    # Highlight significant clusters
    for cluster in sig_clusters:
        color = 'lightgreen' if cluster['direction'] == 'high > low' else 'lightcoral'
        plt.axvspan(cluster['start_time'], cluster['end_time'], alpha=0.3, color=color)
    
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Pupil Size (participant means)', fontsize=14)
    plt.title('Qdiff Z-Scored: Pupil Response Timecourse', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pupil_timecourse.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Distribution plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Effect size distribution
    ax1.hist(effect_sizes, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero effect')
    ax1.axvline(x=effect_sizes.mean(), color='orange', linestyle='-', alpha=0.8, 
                label=f'Mean: {effect_sizes.mean():.3f}')
    ax1.set_xlabel('Effect Size (Cohen\'s d)', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.set_title('Effect Size Distribution', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # P-value distribution
    ax2.hist(p_values, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
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

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def print_analysis_summary(results: Dict):
    """Print detailed analysis summary."""
    
    clusters = results['clusters']
    stats_dict = results['stats']
    
    print(f"\n{'='*70}")
    print(f"QDIFF Z-SCORED ANALYSIS RESULTS")
    print(f"{'='*70}")
    
    print(f"\n*** SAMPLE INFORMATION ***")
    print(f"Participants: {len(results['high_group'])}")
    print(f"Time points: {len([t for t in results['time_columns'] if t in stats_dict])}")
    print(f"Permutations: {results['n_permutations']}")
    print(f"T-threshold: {results['t_threshold']}")
    
    print(f"\n*** CLUSTER ANALYSIS ***")
    print(f"Total clusters: {len(clusters)}")
    sig_clusters = [c for c in clusters if c.get('p_value_max', 1) < 0.05]
    print(f"Significant clusters (p < 0.05): {len(sig_clusters)}")
    
    if sig_clusters:
        for i, cluster in enumerate(sig_clusters):
            duration = cluster['end_time'] - cluster['start_time']
            print(f"\n  Cluster {i+1}:")
            print(f"    Time: {cluster['start_time']:.2f}s - {cluster['end_time']:.2f}s ({duration:.2f}s)")
            print(f"    Direction: {cluster['direction']}")
            print(f"    Size: {cluster['size']} timepoints")
            print(f"    Cluster statistic: {abs(cluster['t_sum']):.2f}")
            print(f"    p-value: {cluster['p_value_max']:.4f}")
    else:
        print("  No significant clusters found")
    
    print(f"\n*** EFFECT SIZE SUMMARY ***")
    effect_sizes = [stats_dict[t]['cohens_d'] for t in results['time_columns'] if t in stats_dict]
    p_values = [stats_dict[t]['p_val'] for t in results['time_columns'] if t in stats_dict]
    
    print(f"Mean Cohen's d: {np.mean(effect_sizes):.3f}")
    print(f"Std Cohen's d: {np.std(effect_sizes):.3f}")
    print(f"Max |Cohen's d|: {np.max(np.abs(effect_sizes)):.3f}")
    print(f"Min p-value: {np.min(p_values):.6f}")
    print(f"Significant timepoints: {np.sum(np.array(p_values) < 0.05)}/{len(p_values)} ({np.sum(np.array(p_values) < 0.05)/len(p_values)*100:.1f}%)")
    
    # Direction analysis
    positive_effects = np.sum(np.array(effect_sizes) > 0)
    negative_effects = np.sum(np.array(effect_sizes) < 0)
    print(f"Positive effects (High > Low): {positive_effects}/{len(effect_sizes)} ({positive_effects/len(effect_sizes)*100:.1f}%)")
    print(f"Negative effects (Low > High): {negative_effects}/{len(effect_sizes)} ({negative_effects/len(effect_sizes)*100:.1f}%)")
    
    print(f"\n*** INTERPRETATION ***")
    if len(sig_clusters) > 0:
        print("✓ Significant differences found between high and low Qdiff conditions")
        dominant_direction = max(sig_clusters, key=lambda x: abs(x['t_sum']))['direction']
        print(f"✓ Dominant pattern: {dominant_direction}")
        if 'high > low' in dominant_direction:
            print("  → Higher value differences associated with larger pupil responses")
        else:
            print("  → Lower value differences associated with larger pupil responses")
    else:
        print("✗ No significant cluster-corrected differences found")
        uncorrected_sig = np.sum(np.array(p_values) < 0.05)
        if uncorrected_sig > 0:
            print(f"  Note: {uncorrected_sig} uncorrected significant timepoints exist")

def save_analysis_results(results: Dict):
    """Save analysis results to files."""
    
    print(f"\nSaving qdiff z-scored results...")
    
    # Save to Excel
    with pd.ExcelWriter('qdiff_zscore_results.xlsx', engine='openpyxl') as writer:
        # Time series data
        time_series_data = []
        for time in results['time_columns']:
            if time in results['stats']:
                stat = results['stats'][time]
                time_series_data.append({
                    'time': float(time),
                    'high_mean': stat['high_mean'],
                    'low_mean': stat['low_mean'],
                    'mean_diff': stat['mean_diff'],
                    't_stat': stat['t_stat'],
                    'p_val': stat['p_val'],
                    'cohens_d': stat['cohens_d'],
                    'n_pairs': stat['n_pairs']
                })
        
        pd.DataFrame(time_series_data).to_excel(writer, sheet_name='Time_Series', index=False)
        
        # Cluster data
        if results['clusters']:
            cluster_data = []
            for i, cluster in enumerate(results['clusters']):
                cluster_data.append({
                    'cluster_id': i + 1,
                    'start_time': cluster['start_time'],
                    'end_time': cluster['end_time'],
                    'duration': cluster['end_time'] - cluster['start_time'],
                    'direction': cluster['direction'],
                    'cluster_statistic': abs(cluster['t_sum']),
                    'p_value': cluster.get('p_value_max', 1),
                    'significant': cluster.get('p_value_max', 1) < 0.05,
                    'size': cluster['size']
                })
            pd.DataFrame(cluster_data).to_excel(writer, sheet_name='Clusters', index=False)
        
        # Participant data
        results['aggregated_df'].to_excel(writer, sheet_name='Participant_Data', index=False)
    
    # Save pickle
    with open('qdiff_zscore_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to qdiff_zscore_results.xlsx and .pkl")

def run_qdiff_zscore_analysis(qvalues_file: str = '/Users/zengsiyi/Desktop/stim1_test/aligned_files/trial_wise_q_values_separate_alphas_no_reset_0.2_values.csv',
                             pupil_file: str = '/Users/zengsiyi/Desktop/stim1_test/aligned_files/stim1_aligned.csv',
                             t_threshold: float = 2.0, 
                             n_permutations: int = 1000):
    """
    Run complete Qdiff z-scored analysis with cluster-based permutation testing.
    """
    
    print("="*80)
    print("COMPLETE QDIFF Z-SCORED ANALYSIS SUITE")
    print("="*80)
    print("Within-subject median split with cluster-based permutation testing")
    
    # Check for data files
    if not os.path.exists(qvalues_file):
        print(f"Error: {qvalues_file} not found!")
        return None
    
    if not os.path.exists(pupil_file):
        print(f"Error: {pupil_file} not found!")
        return None
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Step 1: Load and merge data
        merged_df, time_columns = load_and_merge_qdiff_data_complete(qvalues_file, pupil_file)
        if merged_df is None:
            return None
        
        # Step 2: Create conditions with within-subject median split
        merged_df = create_qdiff_conditions_complete(merged_df)
        
        # Step 3: Aggregate by participant
        aggregated_df = aggregate_by_participant_complete(merged_df, time_columns)
        
        if len(aggregated_df) == 0:
            print("Error: No data after aggregation!")
            return None
        
        # Step 4: Cluster-based permutation test
        results = perform_cluster_test_complete(
            aggregated_df, time_columns, t_threshold, n_permutations)
        
        # Step 5: Print results summary
        print_analysis_summary(results)
        
        # Step 6: Create comprehensive visualization
        create_comprehensive_qdiff_figure(results)
        
        # Step 7: Create focused plots
        create_focused_qdiff_plots(results)
        
        # Step 8: Save results
        save_analysis_results(results)
        
        print("\n" + "="*80)
        print("QDIFF Z-SCORED ANALYSIS COMPLETED!")
        print("="*80)
        print("\nGenerated files:")
        print("• qdiff_zscore_comprehensive.png - Comprehensive analysis figure")
        print("• qdiff_zscore_plots/ - Individual focused plots")
        print("  - timecourse_effect_size.png")
        print("  - significance_plot.png") 
        print("  - pupil_timecourse.png")
        print("  - distributions_plot.png")
        print("• qdiff_zscore_results.xlsx - Detailed results tables")
        print("• qdiff_zscore_results.pkl - Complete results for further analysis")
        
        # Final summary
        sig_clusters = [c for c in results['clusters'] if c.get('p_value_max', 1) < 0.05]
        effect_sizes = [results['stats'][t]['cohens_d'] for t in results['time_columns'] if t in results['stats']]
        p_values = [results['stats'][t]['p_val'] for t in results['time_columns'] if t in results['stats']]
        
        print(f"\n*** FINAL SUMMARY ***")
        print(f"Analysis type: Within-subject z-scored Qdiff median split")
        print(f"Participants: {len(results['high_group'])}")
        print(f"Significant clusters: {len(sig_clusters)}")
        print(f"Mean effect size: {np.mean(effect_sizes):.3f}")
        print(f"Uncorrected significant timepoints: {np.sum(np.array(p_values) < 0.05)}/{len(p_values)}")
        
        if len(sig_clusters) > 0:
            print("\n✓ SIGNIFICANT RESULTS: Qdiff reliably affects pupil responses")
            for i, cluster in enumerate(sig_clusters):
                print(f"  Cluster {i+1}: {cluster['start_time']:.2f}-{cluster['end_time']:.2f}s, {cluster['direction']}")
        else:
            print("\n✗ NO SIGNIFICANT CLUSTERS: Qdiff effects do not survive cluster correction")
        
        return results
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the complete analysis
    results = run_qdiff_zscore_analysis()
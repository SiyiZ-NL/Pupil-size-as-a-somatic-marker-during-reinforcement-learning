#!/usr/bin/env python3
"""
Complete Q_stim1 Analysis Suite with Integrated Comprehensive Visualizations
Runs both raw and z-scored analyses and generates all publication-ready figures
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
# ANALYSIS FUNCTIONS (Copy from previous scripts)
# ============================================================================

def load_and_merge_qstim1_data_complete(qvalues_file: str, pupil_file: str, 
                                       zscore: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load Q-values and pupil data, calculate Q_stim1 (with optional z-scoring), and merge datasets.
    """
    analysis_type = "Z-SCORED" if zscore else "RAW"
    print(f"=== Q_STIM1 {analysis_type} WITHIN-SUBJECT PERMUTATION ANALYSIS ===")
    
    # Load Q-values data
    print("Step 1: Loading Q-values data...")
    qvalues_df = pd.read_csv(qvalues_file)
    print(f"Loaded {len(qvalues_df)} trials with Q-values")
    
    # Load pupil data
    print("Step 2: Loading pupil timeseries data...")
    pupil_df = pd.read_csv(pupil_file)
    print(f"Loaded {len(pupil_df)} pupil trials")
    
    # Get time columns
    time_columns = [col for col in pupil_df.columns 
                   if col.replace('.', '').replace('-', '').replace(' ', '').isdigit()]
    time_columns = sorted(time_columns, key=float)
    
    # Calculate Q_stim1
    print("Step 3: Calculating Q_stim1...")
    qvalues_df = qvalues_df.copy()
    qvalues_df['q_stim1_raw'] = np.where(
        qvalues_df['selected_image'] == qvalues_df['image_1'],
        qvalues_df['q_chosen'],
        qvalues_df['q_notchosen']
    )
    
    if zscore:
        print("Step 4: Z-scoring Q_stim1 within subjects...")
        qvalues_df['q_stim1_zscore'] = np.nan
        
        for subject_id in qvalues_df['subject_id'].unique():
            subject_mask = qvalues_df['subject_id'] == subject_id
            subject_q_vals = qvalues_df.loc[subject_mask, 'q_stim1_raw']
            
            subject_mean = subject_q_vals.mean()
            subject_std = subject_q_vals.std()
            
            if subject_std > 0:
                z_scores = (subject_q_vals - subject_mean) / subject_std
                qvalues_df.loc[subject_mask, 'q_stim1_zscore'] = z_scores
            else:
                qvalues_df.loc[subject_mask, 'q_stim1_zscore'] = 0
        
        # Use z-scored values for condition creation
        qvalues_df['q_stim1'] = qvalues_df['q_stim1_zscore']
        print(f"Z-scored Q_stim1 range: {qvalues_df['q_stim1'].min():.3f} to {qvalues_df['q_stim1'].max():.3f}")
    else:
        qvalues_df['q_stim1'] = qvalues_df['q_stim1_raw']
        print(f"Raw Q_stim1 range: {qvalues_df['q_stim1'].min():.3f} to {qvalues_df['q_stim1'].max():.3f}")
    
    # Merge datasets
    print("Step 5: Merging Q-values with pupil data...")
    merge_cols = ['subject_id', 'block_id', 'trial_id', 'q_stim1', 'q_stim1_raw']
    if zscore:
        merge_cols.append('q_stim1_zscore')
    merge_cols.extend(['q_chosen', 'q_notchosen', 'selected_image', 'image_1', 
                      'alpha_gain_individual', 'alpha_loss_individual', 'beta'])
    
    merged_df = pd.merge(pupil_df, qvalues_df[merge_cols], 
                        on=['subject_id', 'block_id', 'trial_id'], how='inner')
    
    print(f"Successfully merged {len(merged_df)} trials")
    return merged_df, time_columns

def create_qstim1_conditions_complete(merged_df: pd.DataFrame, zscore: bool = False) -> pd.DataFrame:
    """
    Create High vs Low Q_stim1 conditions using within-subject median split.
    """
    print(f"\n*** CREATING Q_STIM1 CONDITIONS ({'Z-SCORED' if zscore else 'RAW'}) ***")
    
    merged_df = merged_df.copy()
    
    if zscore:
        # Use global median for z-scored values (already standardized)
        global_median = merged_df['q_stim1'].median()
        merged_df['q_stim1_condition'] = np.where(
            merged_df['q_stim1'] >= global_median,
            'high_qstim1_zscore',
            'low_qstim1_zscore'
        )
    else:
        # Use within-subject median split for raw values
        merged_df['q_stim1_condition'] = np.nan
        
        for subject_id in merged_df['subject_id'].unique():
            subject_mask = merged_df['subject_id'] == subject_id
            subject_median = merged_df.loc[subject_mask, 'q_stim1'].median()
            
            merged_df.loc[subject_mask, 'q_stim1_condition'] = np.where(
                merged_df.loc[subject_mask, 'q_stim1'] >= subject_median,
                'high_qstim1',
                'low_qstim1'
            )
    
    # Show condition assignment
    condition_counts = merged_df['q_stim1_condition'].value_counts()
    print(f"Condition assignment:")
    for condition, count in condition_counts.items():
        print(f"  {condition}: {count} trials")
    
    return merged_df

def aggregate_by_participant_complete(merged_df: pd.DataFrame, time_columns: List[str]) -> pd.DataFrame:
    """
    Participant-level aggregation for Q_stim1 conditions.
    """
    print("\n*** PARTICIPANT-LEVEL AGGREGATION ***")
    
    # Clean data
    initial_trials = len(merged_df)
    merged_df = merged_df.dropna(subset=time_columns + ['q_stim1'])
    print(f"Removed {initial_trials - len(merged_df)} trials with missing data")
    
    # Aggregate by participant and condition
    agg_data = []
    unique_subjects = sorted(merged_df['subject_id'].unique())
    
    for subject in unique_subjects:
        subject_data = merged_df[merged_df['subject_id'] == subject]
        
        for condition in subject_data['q_stim1_condition'].unique():
            subject_condition_data = subject_data[subject_data['q_stim1_condition'] == condition]
            
            if len(subject_condition_data) > 0:
                # Calculate mean across trials for this subject and condition
                subject_means = subject_condition_data[time_columns].mean()
                
                # Store Q_stim1 statistics
                learning_params = subject_condition_data.iloc[0]
                
                agg_row = {
                    'subject_id': subject,
                    'condition': condition,
                    'n_trials': len(subject_condition_data),
                    'q_stim1_mean': subject_condition_data['q_stim1'].mean(),
                    'q_stim1_raw_mean': subject_condition_data['q_stim1_raw'].mean(),
                    'alpha_gain': learning_params['alpha_gain_individual'],
                    'alpha_loss': learning_params['alpha_loss_individual'],
                    'beta': learning_params['beta']
                }
                
                # Add z-scored values if available
                if 'q_stim1_zscore' in subject_condition_data.columns:
                    agg_row['q_stim1_zscore_mean'] = subject_condition_data['q_stim1_zscore'].mean()
                
                # Add time point means
                for time_col in time_columns:
                    agg_row[time_col] = subject_means[time_col]
                
                agg_data.append(agg_row)
    
    aggregated_df = pd.DataFrame(agg_data)
    
    # Determine high/low groups based on condition names
    if 'high_qstim1_zscore' in aggregated_df['condition'].values:
        high_condition = 'high_qstim1_zscore'
        low_condition = 'low_qstim1_zscore'
    else:
        high_condition = 'high_qstim1'
        low_condition = 'low_qstim1'
    
    final_high = len(aggregated_df[aggregated_df['condition'] == high_condition])
    final_low = len(aggregated_df[aggregated_df['condition'] == low_condition])
    
    print(f"Final sample sizes:")
    print(f"  High condition: {final_high} participants")
    print(f"  Low condition: {final_low} participants")
    
    return aggregated_df

def perform_cluster_test_complete(aggregated_df: pd.DataFrame, time_columns: List[str], 
                                t_threshold: float = 2.0, n_permutations: int = 1000) -> Dict:
    """
    Cluster-based permutation test for Q_stim1 conditions.
    """
    print(f"\n*** CLUSTER-BASED PERMUTATION TEST ***")
    print(f"Running {n_permutations} permutations")
    
    # Determine condition names
    conditions = aggregated_df['condition'].unique()
    if 'high_qstim1_zscore' in conditions:
        high_condition = 'high_qstim1_zscore'
        low_condition = 'low_qstim1_zscore'
    else:
        high_condition = 'high_qstim1'
        low_condition = 'low_qstim1'
    
    # Separate by condition
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
        'analysis_type': 'zscore' if 'zscore' in high_condition else 'raw'
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

def create_comprehensive_qstim1_figure(results: Dict, save_path: str = None, figsize: Tuple = (16, 12)):
    """Create comprehensive publication-ready figure."""
    
    analysis_type = results['analysis_type']
    if save_path is None:
        save_path = f'qstim1_{analysis_type}_comprehensive.png'
    
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
    title_text = f"Comprehensive Q_stim1 Analysis: {'Z-Scored' if analysis_type == 'zscore' else 'Raw'} Values"
    model_text = f"Model: pupil_response ~ q_stim1_{'zscore' if analysis_type == 'zscore' else 'raw'} + (1|subject_id)"
    fig.suptitle(title_text + '\n' + model_text, fontsize=16, fontweight='bold', y=0.95)
    
    # Panel A: Effect size over time
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(times, effect_sizes, 'k-', linewidth=2, label='Effect size (Cohen\'s d)')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Highlight significant clusters
    sig_clusters = [c for c in clusters if c.get('p_value_max', 1) < 0.05]
    for cluster in sig_clusters:
        ax1.axvspan(cluster['start_time'], cluster['end_time'], alpha=0.2, color='yellow')
    
    ax1.set_xlabel('Time from stimulus onset (s)', fontsize=12)
    ax1.set_ylabel('Standardized coefficient', fontsize=12)
    ax1.set_title('A. Effect size over time', fontsize=14, fontweight='bold', loc='left')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel B: Statistical significance
    ax2 = fig.add_subplot(gs[0, 2])
    log_p_values = -np.log10(p_values)
    ax2.plot(times, log_p_values, 'k-', linewidth=2)
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p = 0.05')
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
    
    ax3.set_xlabel('Standardized coefficient', fontsize=12)
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
        ['Analysis type', analysis_type + '_scaled'],
        ['Total timepoints', str(n_timepoints)],
        ['Mean coefficient', f'{mean_coeff:.3f}'],
        ['Min p-value', f'{min_p:.3f}'],
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
    
    ax5.plot(times, high_means, 'r-', linewidth=3, label=f'High {analysis_type} Q_stim1')
    ax5.fill_between(times, high_means - high_ses, high_means + high_ses, alpha=0.3, color='red')
    ax5.plot(times, low_means, 'b-', linewidth=3, label=f'Low {analysis_type} Q_stim1')
    ax5.fill_between(times, low_means - low_ses, low_means + low_ses, alpha=0.3, color='blue')
    ax5.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Highlight significant clusters
    for cluster in sig_clusters:
        ax5.axvspan(cluster['start_time'], cluster['end_time'], alpha=0.2, color='yellow')
    
    ax5.set_xlabel('Time from stimulus onset (s)', fontsize=12)
    ax5.set_ylabel('Pupil size (participant means)', fontsize=12)
    ax5.set_title('E. Pupil response timecourse', fontsize=14, fontweight='bold', loc='left')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive {analysis_type} figure saved to {save_path}")
    
    return fig

def create_comparison_figure(results_raw: Dict, results_zscore: Dict, 
                           save_path: str = 'qstim1_comparison_comprehensive.png'):
    """Create comprehensive comparison between raw and z-scored analyses."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Q_stim1 Analysis: Raw vs Z-Scored Comparison', fontsize=16, fontweight='bold')
    
    # Extract data
    times = np.array([float(t) for t in results_raw['time_columns'] if t in results_raw['stats']])
    
    effect_sizes_raw = np.array([results_raw['stats'][t]['cohens_d'] for t in results_raw['time_columns'] if t in results_raw['stats']])
    effect_sizes_zscore = np.array([results_zscore['stats'][t]['cohens_d'] for t in results_zscore['time_columns'] if t in results_zscore['stats']])
    
    p_values_raw = np.array([results_raw['stats'][t]['p_val'] for t in results_raw['time_columns'] if t in results_raw['stats']])
    p_values_zscore = np.array([results_zscore['stats'][t]['p_val'] for t in results_zscore['time_columns'] if t in results_zscore['stats']])
    
    # Panel 1: Effect sizes comparison
    ax1 = axes[0, 0]
    ax1.plot(times, effect_sizes_raw, 'b-', linewidth=3, label='Raw Q_stim1', alpha=0.8)
    ax1.plot(times, effect_sizes_zscore, 'r-', linewidth=3, label='Z-scored Q_stim1', alpha=0.8)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Effect Sizes Over Time', fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Cohen\'s d')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Statistical significance
    ax2 = axes[0, 1]
    ax2.plot(times, -np.log10(p_values_raw), 'b-', linewidth=3, label='Raw Q_stim1', alpha=0.8)
    ax2.plot(times, -np.log10(p_values_zscore), 'r-', linewidth=3, label='Z-scored Q_stim1', alpha=0.8)
    ax2.axhline(y=-np.log10(0.05), color='orange', linestyle='--', alpha=0.7, label='p = 0.05')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Statistical Significance', fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('-log₁₀(p-value)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Direct comparison
    ax3 = axes[0, 2]
    correlation = np.corrcoef(effect_sizes_raw, effect_sizes_zscore)[0, 1]
    ax3.scatter(effect_sizes_raw, effect_sizes_zscore, c=times, cmap='viridis', alpha=0.7)
    
    # Add diagonal line
    min_val = min(np.min(effect_sizes_raw), np.min(effect_sizes_zscore))
    max_val = max(np.max(effect_sizes_raw), np.max(effect_sizes_zscore))
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    ax3.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax3.transAxes, fontsize=12)
    ax3.set_title('Direct Comparison', fontweight='bold')
    ax3.set_xlabel('Raw Effect Size')
    ax3.set_ylabel('Z-scored Effect Size')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Pupil timecourse - Raw
    ax4 = axes[1, 0]
    stats_raw = results_raw['stats']
    
    # Get the time columns that actually exist in the stats
    valid_times_raw = [t for t in results_raw['time_columns'] if t in stats_raw]
    times_for_raw = np.array([float(t) for t in valid_times_raw])
    
    high_means_raw = np.array([stats_raw[t]['high_mean'] for t in valid_times_raw])
    low_means_raw = np.array([stats_raw[t]['low_mean'] for t in valid_times_raw])
    high_ses_raw = np.array([stats_raw[t]['high_se'] for t in valid_times_raw])
    low_ses_raw = np.array([stats_raw[t]['low_se'] for t in valid_times_raw])
    
    ax4.plot(times_for_raw, high_means_raw, 'r-', linewidth=2, label='High raw Q_stim1')
    ax4.fill_between(times_for_raw, high_means_raw - high_ses_raw, high_means_raw + high_ses_raw, alpha=0.3, color='red')
    ax4.plot(times_for_raw, low_means_raw, 'b-', linewidth=2, label='Low raw Q_stim1')
    ax4.fill_between(times_for_raw, low_means_raw - low_ses_raw, low_means_raw + low_ses_raw, alpha=0.3, color='blue')
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Highlight significant clusters for raw
    for cluster in results_raw['clusters']:
        if cluster.get('p_value_max', 1) < 0.05:
            ax4.axvspan(cluster['start_time'], cluster['end_time'], alpha=0.3, color='yellow', zorder=0)
    
    ax4.set_title('Raw Q_stim1 Pupil Response', fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Pupil Size')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Pupil timecourse - Z-scored
    ax5 = axes[1, 1]
    stats_zscore = results_zscore['stats']
    
    # Get the time columns that actually exist in the stats
    valid_times_zscore = [t for t in results_zscore['time_columns'] if t in stats_zscore]
    times_for_zscore = np.array([float(t) for t in valid_times_zscore])
    
    high_means_zscore = np.array([stats_zscore[t]['high_mean'] for t in valid_times_zscore])
    low_means_zscore = np.array([stats_zscore[t]['low_mean'] for t in valid_times_zscore])
    high_ses_zscore = np.array([stats_zscore[t]['high_se'] for t in valid_times_zscore])
    low_ses_zscore = np.array([stats_zscore[t]['low_se'] for t in valid_times_zscore])
    
    ax5.plot(times_for_zscore, high_means_zscore, 'r-', linewidth=2, label='High z-score Q_stim1')
    ax5.fill_between(times_for_zscore, high_means_zscore - high_ses_zscore, high_means_zscore + high_ses_zscore, alpha=0.3, color='red')
    ax5.plot(times_for_zscore, low_means_zscore, 'b-', linewidth=2, label='Low z-score Q_stim1')
    ax5.fill_between(times_for_zscore, low_means_zscore - low_ses_zscore, low_means_zscore + low_ses_zscore, alpha=0.3, color='blue')
    ax5.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Highlight significant clusters for z-scored (if any)
    for cluster in results_zscore['clusters']:
        if cluster.get('p_value_max', 1) < 0.05:
            ax5.axvspan(cluster['start_time'], cluster['end_time'], alpha=0.3, color='yellow', zorder=0)
    
    ax5.set_title('Z-scored Q_stim1 Pupil Response', fontweight='bold')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Pupil Size')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Summary comparison
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create comparison summary
    sig_clusters_raw = [c for c in results_raw['clusters'] if c.get('p_value_max', 1) < 0.05]
    sig_clusters_zscore = [c for c in results_zscore['clusters'] if c.get('p_value_max', 1) < 0.05]
    
    summary_text = "COMPARISON SUMMARY\n\n"
    summary_text += "Raw Analysis:\n"
    summary_text += f"• Mean effect size: {np.mean(effect_sizes_raw):.3f}\n"
    summary_text += f"• Significant timepoints: {np.sum(p_values_raw < 0.05)}/{len(p_values_raw)}\n"
    summary_text += f"• Significant clusters: {len(sig_clusters_raw)}\n\n"
    
    summary_text += "Z-scored Analysis:\n"
    summary_text += f"• Mean effect size: {np.mean(effect_sizes_zscore):.3f}\n"
    summary_text += f"• Significant timepoints: {np.sum(p_values_zscore < 0.05)}/{len(p_values_zscore)}\n"
    summary_text += f"• Significant clusters: {len(sig_clusters_zscore)}\n\n"
    
    summary_text += f"Effect correlation: {correlation:.3f}\n\n"
    
    summary_text += "Advantages:\n"
    summary_text += "• Raw: Interpretable scale\n"
    summary_text += "• Z-scored: Controls individual\n  differences better"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison figure saved to {save_path}")
    
    return fig

# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def run_qstim1_analysis(qvalues_file: str, pupil_file: str, zscore: bool = False, 
                       t_threshold: float = 2.0, n_permutations: int = 1000) -> Dict:
    """
    Run complete Q_stim1 analysis (raw or z-scored) with comprehensive visualizations.
    """
    
    print(f"Starting {'Z-SCORED' if zscore else 'RAW'} Q_stim1 analysis...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Step 1: Load and merge data
        merged_df, time_columns = load_and_merge_qstim1_data_complete(
            qvalues_file, pupil_file, zscore=zscore)
        
        # Step 2: Create conditions
        merged_df = create_qstim1_conditions_complete(merged_df, zscore=zscore)
        
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
        create_comprehensive_qstim1_figure(results)
        
        # Step 7: Save results
        save_analysis_results(results, zscore)
        
        return results
        
    except Exception as e:
        print(f"Error during {'z-scored' if zscore else 'raw'} analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def print_analysis_summary(results: Dict):
    """Print detailed analysis summary."""
    
    analysis_type = results['analysis_type']
    clusters = results['clusters']
    stats_dict = results['stats']
    
    print(f"\n{'='*70}")
    print(f"Q_STIM1 {analysis_type.upper()} ANALYSIS RESULTS")
    print(f"{'='*70}")
    
    print(f"\n*** SAMPLE INFORMATION ***")
    print(f"Participants: {len(results['high_group'])}")
    print(f"Time points: {len([t for t in results['time_columns'] if t in stats_dict])}")
    print(f"Permutations: {results['n_permutations']}")
    
    print(f"\n*** CLUSTER ANALYSIS ***")
    print(f"Total clusters: {len(clusters)}")
    sig_clusters = [c for c in clusters if c.get('p_value_max', 1) < 0.05]
    print(f"Significant clusters (p < 0.05): {len(sig_clusters)}")
    
    if sig_clusters:
        for i, cluster in enumerate(sig_clusters):
            print(f"\n  Cluster {i+1}:")
            print(f"    Time: {cluster['start_time']:.2f}s - {cluster['end_time']:.2f}s")
            print(f"    Direction: {cluster['direction']}")
            print(f"    p-value: {cluster['p_value_max']:.4f}")
    
    print(f"\n*** EFFECT SIZE SUMMARY ***")
    effect_sizes = [stats_dict[t]['cohens_d'] for t in results['time_columns'] if t in stats_dict]
    p_values = [stats_dict[t]['p_val'] for t in results['time_columns'] if t in stats_dict]
    
    print(f"Mean Cohen's d: {np.mean(effect_sizes):.3f}")
    print(f"Max |Cohen's d|: {np.max(np.abs(effect_sizes)):.3f}")
    print(f"Significant timepoints: {np.sum(np.array(p_values) < 0.05)}/{len(p_values)}")

def save_analysis_results(results: Dict, zscore: bool):
    """Save analysis results to files."""
    
    suffix = 'zscore' if zscore else 'raw'
    
    print(f"\nSaving {suffix} results...")
    
    # Save to Excel
    with pd.ExcelWriter(f'qstim1_{suffix}_results.xlsx', engine='openpyxl') as writer:
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
                    'significant': cluster.get('p_value_max', 1) < 0.05
                })
            pd.DataFrame(cluster_data).to_excel(writer, sheet_name='Clusters', index=False)
    
    # Save pickle
    with open(f'qstim1_{suffix}_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to qstim1_{suffix}_results.xlsx and .pkl")

def run_complete_qstim1_analysis(qvalues_file: str = '/Users/zengsiyi/Desktop/stim1_test/aligned_files/trial_wise_q_values_separate_alphas_no_reset_0.2_values.csv',
                                pupil_file: str = '/Users/zengsiyi/Desktop/stim1_test/aligned_files/stim1_aligned.csv',
                                t_threshold: float = 2.0, 
                                n_permutations: int = 1000):
    """
    Run complete Q_stim1 analysis suite: both raw and z-scored with comprehensive comparisons.
    """
    
    print("="*80)
    print("COMPLETE Q_STIM1 ANALYSIS SUITE")
    print("="*80)
    print("This will run both raw and z-scored analyses with comprehensive visualizations")
    
    # Check for data files
    if not os.path.exists(qvalues_file):
        print(f"Error: {qvalues_file} not found!")
        return None, None
    
    if not os.path.exists(pupil_file):
        print(f"Error: {pupil_file} not found!")
        return None, None
    
    # Run raw analysis
    print("\n" + "="*50)
    print("PHASE 1: RAW Q_STIM1 ANALYSIS")
    print("="*50)
    results_raw = run_qstim1_analysis(
        qvalues_file, pupil_file, zscore=False, 
        t_threshold=t_threshold, n_permutations=n_permutations
    )
    
    # Run z-scored analysis
    print("\n" + "="*50)
    print("PHASE 2: Z-SCORED Q_STIM1 ANALYSIS")
    print("="*50)
    results_zscore = run_qstim1_analysis(
        qvalues_file, pupil_file, zscore=True, 
        t_threshold=t_threshold, n_permutations=n_permutations
    )
    
    # Create comparison figures
    if results_raw is not None and results_zscore is not None:
        print("\n" + "="*50)
        print("PHASE 3: COMPREHENSIVE COMPARISON")
        print("="*50)
        create_comparison_figure(results_raw, results_zscore)
        
        # Print final comparison
        print("\n*** FINAL COMPARISON SUMMARY ***")
        
        raw_effect_sizes = [results_raw['stats'][t]['cohens_d'] for t in results_raw['time_columns'] if t in results_raw['stats']]
        zscore_effect_sizes = [results_zscore['stats'][t]['cohens_d'] for t in results_zscore['time_columns'] if t in results_zscore['stats']]
        
        raw_p_values = [results_raw['stats'][t]['p_val'] for t in results_raw['time_columns'] if t in results_raw['stats']]
        zscore_p_values = [results_zscore['stats'][t]['p_val'] for t in results_zscore['time_columns'] if t in results_zscore['stats']]
        
        raw_sig_clusters = len([c for c in results_raw['clusters'] if c.get('p_value_max', 1) < 0.05])
        zscore_sig_clusters = len([c for c in results_zscore['clusters'] if c.get('p_value_max', 1) < 0.05])
        
        print(f"Raw analysis:")
        print(f"  • Mean effect size: {np.mean(raw_effect_sizes):.3f}")
        print(f"  • Significant timepoints: {np.sum(np.array(raw_p_values) < 0.05)}/{len(raw_p_values)}")
        print(f"  • Significant clusters: {raw_sig_clusters}")
        
        print(f"\nZ-scored analysis:")
        print(f"  • Mean effect size: {np.mean(zscore_effect_sizes):.3f}")
        print(f"  • Significant timepoints: {np.sum(np.array(zscore_p_values) < 0.05)}/{len(zscore_p_values)}")
        print(f"  • Significant clusters: {zscore_sig_clusters}")
        
        correlation = np.corrcoef(raw_effect_sizes, zscore_effect_sizes)[0, 1]
        print(f"\nEffect size correlation: {correlation:.3f}")
        
        if zscore_sig_clusters > raw_sig_clusters:
            print("\n✓ Z-scoring approach found more significant clusters")
        elif raw_sig_clusters > zscore_sig_clusters:
            print("\n✓ Raw approach found more significant clusters")
        else:
            print("\n→ Both approaches found similar numbers of significant clusters")
    
    print("\n" + "="*80)
    print("COMPLETE ANALYSIS FINISHED!")
    print("="*80)
    print("\nGenerated files:")
    print("• qstim1_raw_comprehensive.png - Raw analysis comprehensive figure")
    print("• qstim1_zscore_comprehensive.png - Z-scored analysis comprehensive figure") 
    print("• qstim1_comparison_comprehensive.png - Side-by-side comparison")
    print("• qstim1_raw_results.xlsx/.pkl - Raw analysis results")
    print("• qstim1_zscore_results.xlsx/.pkl - Z-scored analysis results")
    
    return results_raw, results_zscore

if __name__ == "__main__":
    # Run the complete analysis suite
    results_raw, results_zscore = run_complete_qstim1_analysis()

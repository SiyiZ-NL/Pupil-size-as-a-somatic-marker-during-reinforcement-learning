#!/usr/bin/env python3
"""
Comprehensive CORRECTED 44-Subject Pupil Response Analysis Script
Performs proper participant-level aggregation with detailed visualizations
Creates publication-ready figures similar to the original script but methodologically sound
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_and_clean_data_corrected(timeseries_file: str = 'stim1_aligned.csv') -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and clean the 44-subject timeseries data with proper aggregation.
    """
    print("=== COMPREHENSIVE CORRECTED 44-SUBJECT PUPIL ANALYSIS ===")
    print("Proper participant-level aggregation to avoid pseudoreplication\n")
    
    # Load data
    print("Step 1: Loading 44-subject data...")
    timeseries_df = pd.read_csv(timeseries_file)
    
    print(f"Loaded {len(timeseries_df)} total trials from dataset")
    
    # Get time columns
    time_columns = [col for col in timeseries_df.columns 
                   if col.replace('.', '').replace('-', '').replace(' ', '').isdigit()]
    time_columns = sorted(time_columns, key=float)
    
    print(f"Found {len(time_columns)} time points from {time_columns[0]}s to {time_columns[-1]}s")
    
    # Check data structure
    if 'rp1' not in timeseries_df.columns:
        print("Error: 'rp1' column not found!")
        raise ValueError("Required 'rp1' column missing")
    
    if 'subject_id' not in timeseries_df.columns:
        print("Error: 'subject_id' column not found!")
        raise ValueError("Required 'subject_id' column missing")
    
    print(f"Unique rp1 values: {sorted(timeseries_df['rp1'].unique())}")
    
    # Clean data
    print("\nStep 2: Cleaning data...")
    initial_trials = len(timeseries_df)
    timeseries_df = timeseries_df.dropna(subset=time_columns)
    print(f"Removed {initial_trials - len(timeseries_df)} trials with missing data")
    
    # Check subject distribution
    subject_counts = timeseries_df['subject_id'].value_counts()
    print(f"Data from {len(subject_counts)} unique subjects")
    print(f"Trials per subject: min={subject_counts.min()}, max={subject_counts.max()}, mean={subject_counts.mean():.1f}")
    
    return timeseries_df, time_columns

def aggregate_by_participant_comprehensive(timeseries_df: pd.DataFrame, time_columns: List[str]) -> pd.DataFrame:
    """
    Comprehensive participant-level aggregation with detailed reporting.
    """
    print("\n*** STEP 3: PARTICIPANT-LEVEL AGGREGATION ***")
    print("This is the critical step that prevents pseudoreplication")
    
    # Create condition labels
    timeseries_df = timeseries_df.copy()
    timeseries_df['condition'] = timeseries_df['rp1'].apply(
        lambda x: 'high_reward' if x in [0.7, 0.8] else 'low_reward' if x in [0.2, 0.3] else 'other'
    )
    
    # Remove trials that don't fit our conditions
    before_filter = len(timeseries_df)
    timeseries_df = timeseries_df[timeseries_df['condition'] != 'other']
    print(f"Filtered to high/low reward trials: {len(timeseries_df)} of {before_filter} trials")
    
    print(f"Trial distribution:")
    condition_counts = timeseries_df['condition'].value_counts()
    for condition, count in condition_counts.items():
        print(f"  {condition}: {count} trials")
    
    # Aggregate by participant and condition
    print(f"\nAggregating across participants...")
    agg_data = []
    
    unique_subjects = sorted(timeseries_df['subject_id'].unique())
    subjects_with_both = 0
    subjects_missing_condition = []
    
    for subject in unique_subjects:
        subject_data = timeseries_df[timeseries_df['subject_id'] == subject]
        
        has_high = len(subject_data[subject_data['condition'] == 'high_reward']) > 0
        has_low = len(subject_data[subject_data['condition'] == 'low_reward']) > 0
        
        if has_high and has_low:
            subjects_with_both += 1
        else:
            subjects_missing_condition.append(subject)
        
        for condition in ['high_reward', 'low_reward']:
            subject_condition_data = subject_data[subject_data['condition'] == condition]
            
            if len(subject_condition_data) > 0:
                # Calculate mean across trials for this subject and condition
                subject_means = subject_condition_data[time_columns].mean()
                
                agg_row = {
                    'subject_id': subject,
                    'condition': condition,
                    'n_trials': len(subject_condition_data)
                }
                
                # Add time point means
                for time_col in time_columns:
                    agg_row[time_col] = subject_means[time_col]
                
                agg_data.append(agg_row)
    
    aggregated_df = pd.DataFrame(agg_data)
    
    print(f"\nAggregation Results:")
    print(f"  Total subjects: {len(unique_subjects)}")
    print(f"  Subjects with both conditions: {subjects_with_both}")
    print(f"  Subjects missing a condition: {len(subjects_missing_condition)}")
    if subjects_missing_condition:
        print(f"    Missing: {subjects_missing_condition}")
    
    # Final sample sizes
    final_high = len(aggregated_df[aggregated_df['condition'] == 'high_reward'])
    final_low = len(aggregated_df[aggregated_df['condition'] == 'low_reward'])
    
    print(f"\n*** FINAL SAMPLE SIZES FOR ANALYSIS ***")
    print(f"High reward condition: {final_high} participants")
    print(f"Low reward condition: {final_low} participants")
    print(f"This is the CORRECT statistical approach!")
    
    return aggregated_df

def perform_comprehensive_cluster_test(aggregated_df: pd.DataFrame, time_columns: List[str], 
                                     t_threshold: float = 2.0, n_permutations: int = 1000) -> Dict:
    """
    Comprehensive cluster-based permutation test with detailed statistics.
    """
    print(f"\n*** STEP 4: CLUSTER-BASED PERMUTATION TEST ***")
    print(f"Running {n_permutations} permutations on participant-level data")
    
    # Separate by condition
    high_reward = aggregated_df[aggregated_df['condition'] == 'high_reward']
    low_reward = aggregated_df[aggregated_df['condition'] == 'low_reward']
    
    print(f"\nData for analysis:")
    print(f"  High reward: {len(high_reward)} participants")
    print(f"  Low reward: {len(low_reward)} participants")
    print(f"  Time points: {len(time_columns)}")
    
    # Calculate observed statistics
    print(f"\nCalculating observed statistics...")
    observed_stats = {}
    
    for time in time_columns:
        high_values = high_reward[time].values
        low_values = low_reward[time].values
        
        # Remove NaN values
        high_values = high_values[~np.isnan(high_values)]
        low_values = low_values[~np.isnan(low_values)]
        
        if len(high_values) == 0 or len(low_values) == 0:
            continue
            
        # Calculate comprehensive statistics
        high_mean = np.mean(high_values)
        low_mean = np.mean(low_values)
        high_std = np.std(high_values, ddof=1)
        low_std = np.std(low_values, ddof=1)
        high_se = high_std / np.sqrt(len(high_values))
        low_se = low_std / np.sqrt(len(low_values))
        
        # Statistical tests
        t_stat, p_val = stats.ttest_ind(high_values, low_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(high_values) - 1) * high_std**2 + 
                             (len(low_values) - 1) * low_std**2) / 
                            (len(high_values) + len(low_values) - 2))
        cohens_d = (high_mean - low_mean) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for difference
        diff = high_mean - low_mean
        se_diff = np.sqrt(high_se**2 + low_se**2)
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        
        observed_stats[time] = {
            'high_mean': high_mean,
            'low_mean': low_mean,
            'high_std': high_std,
            'low_std': low_std,
            'high_se': high_se,
            'low_se': low_se,
            'mean_diff': diff,
            't_stat': t_stat,
            'p_val': p_val,
            'cohens_d': cohens_d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'high_n': len(high_values),
            'low_n': len(low_values)
        }
    
    # Find observed clusters
    observed_clusters = find_clusters_comprehensive(observed_stats, time_columns, t_threshold)
    
    print(f"\nObserved clusters found: {len(observed_clusters)}")
    for i, cluster in enumerate(observed_clusters):
        print(f"  Cluster {i+1}: {cluster['start_time']:.2f}s - {cluster['end_time']:.2f}s")
        print(f"    Direction: {cluster['direction']}")
        print(f"    Cluster statistic: {abs(cluster['t_sum']):.2f}")
        print(f"    Size: {cluster['size']} time points")
    
    # Permutation test
    print(f"\nRunning {n_permutations} permutations...")
    
    all_participant_data = aggregated_df.copy()
    null_cluster_stats = []
    max_cluster_stats_per_perm = []
    
    for perm_i in tqdm(range(n_permutations), desc="Permutations"):
        # Shuffle condition labels
        perm_data = all_participant_data.copy()
        shuffled_conditions = np.random.permutation(perm_data['condition'].values)
        perm_data['condition'] = shuffled_conditions
        
        # Calculate permuted statistics
        perm_high = perm_data[perm_data['condition'] == 'high_reward']
        perm_low = perm_data[perm_data['condition'] == 'low_reward']
        
        perm_stats = {}
        for time in time_columns:
            if time not in observed_stats:
                continue
                
            high_vals = perm_high[time].values
            low_vals = perm_low[time].values
            
            high_vals = high_vals[~np.isnan(high_vals)]
            low_vals = low_vals[~np.isnan(low_vals)]
            
            if len(high_vals) == 0 or len(low_vals) == 0:
                continue
            
            t_stat, _ = stats.ttest_ind(high_vals, low_vals)
            perm_stats[time] = {'t_stat': t_stat}
        
        # Find permuted clusters
        perm_clusters = find_clusters_comprehensive(perm_stats, time_columns, t_threshold)
        
        # Store cluster statistics
        for cluster in perm_clusters:
            null_cluster_stats.append(abs(cluster['t_sum']))
        
        if perm_clusters:
            max_cluster_stats_per_perm.append(max([abs(c['t_sum']) for c in perm_clusters]))
        else:
            max_cluster_stats_per_perm.append(0)
    
    # Calculate p-values
    print(f"\nCalculating cluster p-values...")
    for i, cluster in enumerate(observed_clusters):
        observed_stat = abs(cluster['t_sum'])
        
        if len(null_cluster_stats) > 0:
            p_value_all = np.mean([stat >= observed_stat for stat in null_cluster_stats])
        else:
            p_value_all = 1.0
        
        if len(max_cluster_stats_per_perm) > 0:
            p_value_max = np.mean([stat >= observed_stat for stat in max_cluster_stats_per_perm])
        else:
            p_value_max = 1.0
        
        cluster['p_value_all'] = p_value_all
        cluster['p_value_max'] = p_value_max
        
        print(f"Cluster {i+1}: p = {p_value_max:.4f} (corrected)")
    
    return {
        'stats': observed_stats,
        'clusters': observed_clusters,
        'null_cluster_stats': null_cluster_stats,
        'max_cluster_stats_per_perm': max_cluster_stats_per_perm,
        'high_reward': high_reward,
        'low_reward': low_reward,
        'time_columns': time_columns,
        'n_permutations': n_permutations,
        't_threshold': t_threshold,
        'aggregated_df': aggregated_df
    }

def find_clusters_comprehensive(stats_dict: Dict, time_columns: List[str], 
                              t_threshold: float) -> List[Dict]:
    """
    Find clusters with comprehensive information.
    """
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

def create_comprehensive_publication_figure(results: Dict, save_path: str = 'corrected_44subject_publication.png'):
    """
    Create a comprehensive publication-ready figure.
    """
    stats_dict = results['stats']
    clusters = results['clusters']
    time_columns = results['time_columns']
    
    # Prepare data
    times = np.array([float(t) for t in time_columns if t in stats_dict])
    high_means = np.array([stats_dict[t]['high_mean'] for t in time_columns if t in stats_dict])
    low_means = np.array([stats_dict[t]['low_mean'] for t in time_columns if t in stats_dict])
    high_ses = np.array([stats_dict[t]['high_se'] for t in time_columns if t in stats_dict])
    low_ses = np.array([stats_dict[t]['low_se'] for t in time_columns if t in stats_dict])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot means with error bands
    ax.plot(times, high_means, color='#e74c3c', linewidth=3, 
            label=f'High reward (N={len(results["high_reward"])})', zorder=3)
    ax.fill_between(times, high_means - high_ses, high_means + high_ses, 
                   alpha=0.25, color='#e74c3c', zorder=2)
    
    ax.plot(times, low_means, color='#3498db', linewidth=3, 
            label=f'Low reward (N={len(results["low_reward"])})', zorder=3)
    ax.fill_between(times, low_means - low_ses, low_means + low_ses, 
                   alpha=0.25, color='#3498db', zorder=2)
    
    # Mark stimulus onset
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, 
              alpha=0.6, zorder=1, label='Stimulus onset')
    
    # Highlight significant clusters (if any)
    sig_clusters = [c for c in clusters if c.get('p_value_max', 1) < 0.05]
    for i, cluster in enumerate(sig_clusters):
        ax.axvspan(cluster['start_time'], cluster['end_time'], 
                  alpha=0.2, color='yellow', zorder=0)
        
        y_pos = ax.get_ylim()[1] * 0.95
        x_pos = (cluster['start_time'] + cluster['end_time']) / 2
        ax.text(x_pos, y_pos, f'***', ha='center', va='top', 
               fontsize=16, fontweight='bold')
    
    # Add non-significant clusters with light shading
    nonsig_clusters = [c for c in clusters if c.get('p_value_max', 1) >= 0.05]
    for cluster in nonsig_clusters:
        ax.axvspan(cluster['start_time'], cluster['end_time'], 
                  alpha=0.05, color='gray', zorder=0)
    
    # Formatting
    ax.set_xlabel('Time from stimulus onset (s)', fontsize=14)
    ax.set_ylabel('Pupil size (participant means)', fontsize=14)
    ax.set_title('44-Subject Pupil Response: Corrected Participant-Level Analysis', 
                fontsize=16, fontweight='bold')
    ax.set_xlim(-0.5, 3)
    
    # Legend
    legend_elements = ax.get_legend_handles_labels()[0][:3]  # First 3 elements
    if sig_clusters:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='yellow', alpha=0.2, label='Significant cluster'))
    if nonsig_clusters:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.05, label='Non-significant cluster'))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, 
             frameon=True, fancybox=True, shadow=True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Publication figure saved to {save_path}")
    
    return fig

def create_comprehensive_analysis_figure(results: Dict, save_path: str = 'corrected_44subject_comprehensive.png'):
    """
    Create comprehensive multi-panel analysis figure.
    """
    stats_dict = results['stats']
    clusters = results['clusters']
    time_columns = results['time_columns']
    
    # Prepare data
    times = np.array([float(t) for t in time_columns if t in stats_dict])
    high_means = np.array([stats_dict[t]['high_mean'] for t in time_columns if t in stats_dict])
    low_means = np.array([stats_dict[t]['low_mean'] for t in time_columns if t in stats_dict])
    high_ses = np.array([stats_dict[t]['high_se'] for t in time_columns if t in stats_dict])
    low_ses = np.array([stats_dict[t]['low_se'] for t in time_columns if t in stats_dict])
    t_stats = np.array([stats_dict[t]['t_stat'] for t in time_columns if t in stats_dict])
    differences = np.array([stats_dict[t]['mean_diff'] for t in time_columns if t in stats_dict])
    cohens_d = np.array([stats_dict[t]['cohens_d'] for t in time_columns if t in stats_dict])
    ci_lower = np.array([stats_dict[t]['ci_lower'] for t in time_columns if t in stats_dict])
    ci_upper = np.array([stats_dict[t]['ci_upper'] for t in time_columns if t in stats_dict])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1.5], width_ratios=[3, 1], 
                         hspace=0.3, wspace=0.3)
    
    # Main plot: Pupil size with error bands
    ax1 = fig.add_subplot(gs[0, :])
    
    ax1.plot(times, high_means, 'r-', linewidth=3, label=f'High Reward (N={len(results["high_reward"])})')
    ax1.fill_between(times, high_means - high_ses, high_means + high_ses, 
                     alpha=0.3, color='red')
    
    ax1.plot(times, low_means, 'b-', linewidth=3, label=f'Low Reward (N={len(results["low_reward"])})')
    ax1.fill_between(times, low_means - low_ses, low_means + low_ses, 
                     alpha=0.3, color='blue')
    
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    # Highlight clusters
    for i, cluster in enumerate(clusters):
        p_val = cluster.get('p_value_max', 1)
        alpha = 0.3 if p_val < 0.05 else 0.1
        color = 'yellow' if p_val < 0.05 else 'gray'
        ax1.axvspan(cluster['start_time'], cluster['end_time'], 
                   alpha=alpha, color=color, zorder=0)
        
        if p_val < 0.05:
            mid_time = (cluster['start_time'] + cluster['end_time']) / 2
            y_max = max(np.max(high_means + high_ses), np.max(low_means + low_ses))
            ax1.text(mid_time, y_max * 1.02, f'Cluster {i+1}*', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Pupil Size (participant means)', fontsize=12)
    ax1.set_title('Corrected 44-Subject Pupil Response Analysis', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # T-statistics plot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(times, t_stats, 'k-', linewidth=2)
    ax2.axhline(y=2, color='red', linestyle='--', linewidth=1, alpha=0.7, label='p < 0.05')
    ax2.axhline(y=-2, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    for cluster in clusters:
        p_val = cluster.get('p_value_max', 1)
        alpha = 0.4 if p_val < 0.05 else 0.2
        color = 'yellow' if p_val < 0.05 else 'gray'
        ax2.axvspan(cluster['start_time'], cluster['end_time'], 
                   alpha=alpha, color=color)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('T-statistic', fontsize=12)
    ax2.set_title('Statistical Significance Over Time', fontsize=13)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Effect size plot (Cohen's d)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(times, cohens_d, 'g-', linewidth=2, label="Cohen's d")
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax3.axhline(y=0.2, color='orange', linestyle=':', alpha=0.7, label='Small effect')
    ax3.axhline(y=0.5, color='red', linestyle=':', alpha=0.7, label='Medium effect')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    for cluster in clusters:
        p_val = cluster.get('p_value_max', 1)
        alpha = 0.3 if p_val < 0.05 else 0.1
        color = 'yellow' if p_val < 0.05 else 'gray'
        ax3.axvspan(cluster['start_time'], cluster['end_time'], 
                   alpha=alpha, color=color)
    
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel("Effect Size (Cohen's d)", fontsize=12)
    ax3.set_title('Effect Size Over Time', fontsize=13)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Summary statistics panel
    ax4 = fig.add_subplot(gs[1:3, 1])
    ax4.axis('off')
    
    summary_text = "CORRECTED ANALYSIS SUMMARY\n"
    summary_text += f"({results['n_permutations']} permutations)\n\n"
    
    summary_text += "Sample Sizes:\n"
    summary_text += f"• High reward: {len(results['high_reward'])} participants\n"
    summary_text += f"• Low reward: {len(results['low_reward'])} participants\n"
    summary_text += f"• Total timepoints: {len(times)}\n\n"
    
    summary_text += f"Clusters Found: {len(clusters)}\n"
    sig_clusters = [c for c in clusters if c.get('p_value_max', 1) < 0.05]
    summary_text += f"Significant clusters: {len(sig_clusters)}\n\n"
    
    if clusters:
        for i, cluster in enumerate(clusters):
            p_val = cluster.get('p_value_max', 1)
            summary_text += f"Cluster {i + 1}:\n"
            summary_text += f"  Time: {cluster['start_time']:.2f}s - {cluster['end_time']:.2f}s\n"
            summary_text += f"  Duration: {cluster['end_time'] - cluster['start_time']:.2f}s\n"
            summary_text += f"  Direction: {cluster['direction']}\n"
            summary_text += f"  Cluster stat: {abs(cluster['t_sum']):.1f}\n"
            summary_text += f"  Max |t|: {cluster['max_t']:.2f}\n"
            summary_text += f"  Size: {cluster['size']} points\n"
            summary_text += f"  p-value: {p_val:.3f} {'***' if p_val < 0.05 else ''}\n\n"
    else:
        summary_text += "No clusters found\n\n"
    
    # Overall effect statistics
    overall_effect = np.mean(cohens_d)
    max_effect = np.max(np.abs(cohens_d))
    
    summary_text += "Overall Effects:\n"
    summary_text += f"• Mean Cohen's d: {overall_effect:.3f}\n"
    summary_text += f"• Max |Cohen's d|: {max_effect:.3f}\n"
    summary_text += f"• Mean difference: {np.mean(differences):.2f}\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('Comprehensive Corrected 44-Subject Pupil Analysis', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.08)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive figure saved to {save_path}")
    
    return fig

def create_permutation_visualization_corrected(results: Dict, save_path: str = 'corrected_44subject_permutation.png'):
    """
    Create detailed permutation test visualization.
    """
    stats_dict = results['stats']
    clusters = results['clusters']
    time_columns = results['time_columns']
    null_stats = results['null_cluster_stats']
    max_stats = results['max_cluster_stats_per_perm']
    
    # Prepare data
    times = np.array([float(t) for t in time_columns if t in stats_dict])
    high_means = np.array([stats_dict[t]['high_mean'] for t in time_columns if t in stats_dict])
    low_means = np.array([stats_dict[t]['low_mean'] for t in time_columns if t in stats_dict])
    t_stats = np.array([stats_dict[t]['t_stat'] for t in time_columns if t in stats_dict])
    p_vals = np.array([stats_dict[t]['p_val'] for t in time_columns if t in stats_dict])
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean pupil size with confidence intervals
    high_ses = np.array([stats_dict[t]['high_se'] for t in time_columns if t in stats_dict])
    low_ses = np.array([stats_dict[t]['low_se'] for t in time_columns if t in stats_dict])
    
    ax1.plot(times, high_means, 'r-', linewidth=2.5, label=f'High Reward (N={len(results["high_reward"])})')
    ax1.fill_between(times, high_means - 1.96*high_ses, high_means + 1.96*high_ses, 
                     alpha=0.3, color='red', label='95% CI')
    
    ax1.plot(times, low_means, 'b-', linewidth=2.5, label=f'Low Reward (N={len(results["low_reward"])})')
    ax1.fill_between(times, low_means - 1.96*low_ses, low_means + 1.96*low_ses, 
                     alpha=0.3, color='blue')
    
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Stimulus onset')
    
    # Highlight clusters
    for i, cluster in enumerate(clusters):
        p_val = cluster.get('p_value_max', 1)
        color = 'red' if cluster['direction'] == 'high > low' else 'blue'
        alpha = 0.3 if p_val < 0.05 else 0.1
        ax1.axvspan(cluster['start_time'], cluster['end_time'], 
                   alpha=alpha, color=color)
        
        if p_val < 0.05:
            mid_time = (cluster['start_time'] + cluster['end_time']) / 2
            y_pos = ax1.get_ylim()[1] * 0.95
            ax1.text(mid_time, y_pos, f'C{i+1}*', ha='center', va='top', 
                    fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Pupil Size (participant means)')
    ax1.set_title('Corrected 44-Subject Analysis: Participant Means ± 95% CI')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: T-statistics with p-value significance
    ax2.plot(times, t_stats, 'k-', linewidth=2)
    ax2.axhline(y=2, color='r', linestyle='--', alpha=0.7, label='t = ±2 (p ≈ 0.05)')
    ax2.axhline(y=-2, color='r', linestyle='--', alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Color points by significance
    significant_mask = np.abs(t_stats) > 2
    ax2.scatter(times[significant_mask], t_stats[significant_mask], 
               color='red', s=20, zorder=5, alpha=0.7, label='|t| > 2')
    
    # Highlight clusters
    for cluster in clusters:
        p_val = cluster.get('p_value_max', 1)
        alpha = 0.4 if p_val < 0.05 else 0.2
        ax2.axvspan(cluster['start_time'], cluster['end_time'], 
                   alpha=alpha, color='yellow')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('T-statistic')
    ax2.set_title('T-statistics Over Time (Corrected Analysis)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Null distribution of cluster statistics
    if null_stats:
        ax3.hist(null_stats, bins=50, alpha=0.7, color='lightgray', 
                 density=True, label=f'Null distribution ({len(null_stats)} clusters)')
        
        # Mark observed cluster statistics
        for i, cluster in enumerate(clusters):
            p_val = cluster.get('p_value_max', 1)
            color = 'red' if p_val < 0.05 else 'orange'
            linewidth = 3 if p_val < 0.05 else 2
            ax3.axvline(abs(cluster['t_sum']), color=color, linestyle='-', 
                       linewidth=linewidth, label=f'Observed C{i+1}' if i < 3 else "")
        
        # Add statistical information
        if null_stats:
            null_mean = np.mean(null_stats)
            null_95th = np.percentile(null_stats, 95)
            ax3.axvline(null_mean, color='gray', linestyle=':', alpha=0.7, 
                       label=f'Null mean: {null_mean:.1f}')
            ax3.axvline(null_95th, color='gray', linestyle='--', alpha=0.7, 
                       label=f'95th percentile: {null_95th:.1f}')
    
    ax3.set_xlabel('Cluster Statistic (sum |t|)')
    ax3.set_ylabel('Density')
    ax3.set_title('Null Distribution vs Observed Clusters')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: P-value distribution and multiple comparison correction
    if max_stats:
        ax4.hist(max_stats, bins=50, alpha=0.7, color='lightblue',
                 density=True, label=f'Max cluster per permutation')
        
        # Mark observed cluster statistics for multiple comparison correction
        for i, cluster in enumerate(clusters):
            p_val = cluster.get('p_value_max', 1)
            color = 'red' if p_val < 0.05 else 'orange'
            linewidth = 3 if p_val < 0.05 else 2
            ax4.axvline(abs(cluster['t_sum']), color=color, linestyle='-', 
                       linewidth=linewidth, 
                       label=f'C{i+1} (p={p_val:.3f})' if i < 3 else "")
        
        # Add percentile lines
        if max_stats:
            percentiles = [90, 95, 99]
            colors = ['green', 'orange', 'red']
            for pct, color in zip(percentiles, colors):
                val = np.percentile(max_stats, pct)
                ax4.axvline(val, color=color, linestyle='--', alpha=0.7,
                           label=f'{pct}th percentile: {val:.1f}')
    
    ax4.set_xlabel('Maximum Cluster Statistic per Permutation')
    ax4.set_ylabel('Density')
    ax4.set_title('Multiple Comparison Correction')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Permutation visualization saved to {save_path}")
    
    return fig

def save_comprehensive_results(results: Dict, filename: str = 'corrected_44subject_comprehensive_results.xlsx'):
    """
    Save comprehensive corrected analysis results.
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Time series statistics (comprehensive)
        time_series_data = []
        for time in results['time_columns']:
            if time in results['stats']:
                stat = results['stats'][time]
                time_series_data.append({
                    'time': float(time),
                    'high_mean': stat['high_mean'],
                    'low_mean': stat['low_mean'],
                    'high_se': stat['high_se'],
                    'low_se': stat['low_se'],
                    'high_std': stat['high_std'],
                    'low_std': stat['low_std'],
                    'mean_diff': stat['mean_diff'],
                    't_stat': stat['t_stat'],
                    'p_val': stat['p_val'],
                    'cohens_d': stat['cohens_d'],
                    'ci_lower': stat['ci_lower'],
                    'ci_upper': stat['ci_upper'],
                    'high_n': stat['high_n'],
                    'low_n': stat['low_n'],
                    'significant_uncorrected': stat['p_val'] < 0.05,
                    'significant_t_threshold': abs(stat['t_stat']) > 2.0
                })
        
        time_series_df = pd.DataFrame(time_series_data)
        time_series_df.to_excel(writer, sheet_name='Time_Series_Comprehensive', index=False)
        
        # Cluster analysis (detailed)
        cluster_data = []
        for i, cluster in enumerate(results['clusters']):
            cluster_info = {
                'cluster_id': i + 1,
                'start_time': cluster['start_time'],
                'end_time': cluster['end_time'],
                'duration': cluster['end_time'] - cluster['start_time'],
                'direction': cluster['direction'],
                'cluster_statistic': abs(cluster['t_sum']),
                'max_t_in_cluster': cluster['max_t'],
                'size_timepoints': cluster['size'],
                'p_value_corrected': cluster.get('p_value_max', 1),
                'p_value_uncorrected': cluster.get('p_value_all', 1),
                'significant_corrected': cluster.get('p_value_max', 1) < 0.05,
                'significant_uncorrected': cluster.get('p_value_all', 1) < 0.05,
                'time_points_in_cluster': '; '.join(cluster['time_points'])
            }
            cluster_data.append(cluster_info)
        
        if cluster_data:
            cluster_df = pd.DataFrame(cluster_data)
            cluster_df.to_excel(writer, sheet_name='Cluster_Analysis', index=False)
        
        # Effect size analysis
        effect_sizes = []
        for time in results['time_columns']:
            if time in results['stats']:
                stat = results['stats'][time]
                effect_sizes.append({
                    'time': float(time),
                    'cohens_d': stat['cohens_d'],
                    'effect_magnitude': 'negligible' if abs(stat['cohens_d']) < 0.2 else
                                      'small' if abs(stat['cohens_d']) < 0.5 else
                                      'medium' if abs(stat['cohens_d']) < 0.8 else 'large',
                    'mean_difference': stat['mean_diff'],
                    'ci_lower': stat['ci_lower'],
                    'ci_upper': stat['ci_upper'],
                    'ci_includes_zero': stat['ci_lower'] <= 0 <= stat['ci_upper']
                })
        
        effect_df = pd.DataFrame(effect_sizes)
        effect_df.to_excel(writer, sheet_name='Effect_Size_Analysis', index=False)
        
        # Analysis summary
        sig_clusters = sum(1 for c in results['clusters'] if c.get('p_value_max', 1) < 0.05)
        sig_timepoints = sum(1 for time in results['time_columns'] 
                           if time in results['stats'] and results['stats'][time]['p_val'] < 0.05)
        
        summary_info = pd.DataFrame({
            'metric': [
                'Total Participants High Reward',
                'Total Participants Low Reward', 
                'Total Time Points Analyzed',
                'Permutations Run',
                'T-threshold Used',
                'Clusters Found',
                'Significant Clusters (corrected)',
                'Significant Time Points (uncorrected)',
                'Mean Effect Size (Cohen\'s d)',
                'Max Effect Size (Cohen\'s d)',
                'Overall Mean Difference',
                'Analysis Type',
                'Pseudoreplication Corrected'
            ],
            'value': [
                len(results['high_reward']),
                len(results['low_reward']),
                len([time for time in results['time_columns'] if time in results['stats']]),
                results['n_permutations'],
                results['t_threshold'],
                len(results['clusters']),
                sig_clusters,
                sig_timepoints,
                np.mean([results['stats'][t]['cohens_d'] for t in results['time_columns'] if t in results['stats']]),
                np.max([abs(results['stats'][t]['cohens_d']) for t in results['time_columns'] if t in results['stats']]),
                np.mean([results['stats'][t]['mean_diff'] for t in results['time_columns'] if t in results['stats']]),
                'Participant-level aggregation',
                'YES'
            ]
        })
        summary_info.to_excel(writer, sheet_name='Analysis_Summary', index=False)
        
        # Permutation test results
        if results['null_cluster_stats']:
            perm_summary = pd.DataFrame({
                'metric': [
                    'Null Clusters Generated',
                    'Mean Null Cluster Statistic',
                    'Std Null Cluster Statistic',
                    'Max Null Cluster Statistic',
                    '95th Percentile Null',
                    'Max Cluster Stat Per Permutation (Mean)',
                    'Max Cluster Stat Per Permutation (95th Percentile)'
                ],
                'value': [
                    len(results['null_cluster_stats']),
                    np.mean(results['null_cluster_stats']),
                    np.std(results['null_cluster_stats']),
                    np.max(results['null_cluster_stats']),
                    np.percentile(results['null_cluster_stats'], 95),
                    np.mean(results['max_cluster_stats_per_perm']),
                    np.percentile(results['max_cluster_stats_per_perm'], 95)
                ]
            })
            perm_summary.to_excel(writer, sheet_name='Permutation_Summary', index=False)
        
        # Participant aggregated data
        results['aggregated_df'].to_excel(writer, sheet_name='Participant_Data', index=False)
    
    print(f"Comprehensive results saved to {filename}")

def print_comprehensive_results(results: Dict):
    """
    Print detailed analysis results.
    """
    clusters = results['clusters']
    stats_dict = results['stats']
    
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE CORRECTED 44-SUBJECT ANALYSIS RESULTS")
    print(f"{'='*70}")
    
    print(f"\n*** SAMPLE INFORMATION ***")
    print(f"High reward participants: {len(results['high_reward'])}")
    print(f"Low reward participants: {len(results['low_reward'])}")
    print(f"Time points analyzed: {len([t for t in results['time_columns'] if t in stats_dict])}")
    print(f"Permutations run: {results['n_permutations']}")
    print(f"Analysis type: Participant-level aggregation (CORRECTED)")
    
    print(f"\n*** CLUSTER ANALYSIS ***")
    print(f"Total clusters found: {len(clusters)}")
    
    sig_clusters = [c for c in clusters if c.get('p_value_max', 1) < 0.05]
    print(f"Significant clusters (p < 0.05): {len(sig_clusters)}")
    
    if clusters:
        print(f"\nDetailed cluster information:")
        for i, cluster in enumerate(clusters):
            p_corrected = cluster.get('p_value_max', 1)
            p_uncorrected = cluster.get('p_value_all', 1)
            is_sig = p_corrected < 0.05
            
            print(f"\n  Cluster {i+1}:")
            print(f"    Time window: {cluster['start_time']:.2f}s to {cluster['end_time']:.2f}s")
            print(f"    Duration: {cluster['end_time'] - cluster['start_time']:.2f}s")
            print(f"    Direction: {cluster['direction']}")
            print(f"    Cluster statistic: {abs(cluster['t_sum']):.2f}")
            print(f"    Maximum |t| in cluster: {cluster['max_t']:.2f}")
            print(f"    Size: {cluster['size']} time points")
            print(f"    p-value (corrected): {p_corrected:.4f} {'***' if is_sig else ''}")
            print(f"    p-value (uncorrected): {p_uncorrected:.4f}")
    else:
        print("  No clusters found above threshold")
    
    print(f"\n*** EFFECT SIZE ANALYSIS ***")
    cohens_d_values = [stats_dict[t]['cohens_d'] for t in results['time_columns'] if t in stats_dict]
    mean_diffs = [stats_dict[t]['mean_diff'] for t in results['time_columns'] if t in stats_dict]
    
    print(f"Mean Cohen's d across time: {np.mean(cohens_d_values):.3f}")
    print(f"Maximum |Cohen's d|: {np.max(np.abs(cohens_d_values)):.3f}")
    print(f"Mean difference (high - low): {np.mean(mean_diffs):.2f}")
    print(f"Range of differences: {np.min(mean_diffs):.2f} to {np.max(mean_diffs):.2f}")
    
    # Effect size interpretation
    max_effect = np.max(np.abs(cohens_d_values))
    if max_effect < 0.2:
        effect_interp = "negligible"
    elif max_effect < 0.5:
        effect_interp = "small"
    elif max_effect < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    
    print(f"Maximum effect size magnitude: {effect_interp}")
    
    print(f"\n*** STATISTICAL POWER ANALYSIS ***")
    # Estimate power based on observed effect
    observed_mean_effect = np.mean(np.abs(cohens_d_values))
    current_n = len(results['high_reward'])
    
    # Required N for 80% power
    required_n = (2.8 / observed_mean_effect) ** 2 if observed_mean_effect > 0 else float('inf')
    
    print(f"Observed effect size: d = {observed_mean_effect:.3f}")
    print(f"Current sample size: N = {current_n} per group")
    if required_n != float('inf'):
        print(f"Required N for 80% power: ~{required_n:.0f} per group")
        print(f"Current power (approximate): ~{min(100, 100 * (observed_mean_effect * np.sqrt(current_n/2) / 2.8)**2):.0f}%")
    else:
        print("Effect too small to estimate required sample size reliably")
    
    print(f"\n*** CONCLUSION ***")
    if sig_clusters:
        print(f"✓ Found {len(sig_clusters)} significant cluster(s) after multiple comparison correction")
        print("✓ Results are statistically reliable (corrected for pseudoreplication)")
    else:
        print("✗ No significant clusters found after multiple comparison correction")
        print("✓ Analysis is methodologically sound (corrected for pseudoreplication)")
        print("• Small effect sizes may require larger sample sizes to detect")
        print("• Current results provide valuable pilot data for future studies")

def main_comprehensive_corrected():
    """
    Main function for comprehensive corrected 44-subject analysis.
    """
    print("Starting comprehensive CORRECTED 44-subject pupil analysis...")
    print("This analysis uses proper participant-level aggregation.\n")
    
    # Set random seed
    np.random.seed(42)
    
    # Check for data file
    timeseries_file = '/Users/zengsiyi/Desktop/stim1_test/aligned_files/stim1_aligned.csv'
    if not os.path.exists(timeseries_file):
        print(f"Error: {timeseries_file} not found!")
        return None
    
    try:
        # Step 1: Load and clean data
        timeseries_clean, time_columns = load_and_clean_data_corrected(timeseries_file)
        
        # Step 2: Aggregate by participant
        aggregated_df = aggregate_by_participant_comprehensive(timeseries_clean, time_columns)
        
        if len(aggregated_df) == 0:
            print("Error: No data after aggregation!")
            return None
        
        # Step 3: Comprehensive cluster analysis
        results = perform_comprehensive_cluster_test(
            aggregated_df, time_columns, 
            t_threshold=2.0, n_permutations=1000
        )
        
        # Step 4: Print detailed results
        print_comprehensive_results(results)
        
        # Step 5: Create all visualizations
        print(f"\n*** CREATING VISUALIZATIONS ***")
        
        print("1. Creating publication-ready figure...")
        create_comprehensive_publication_figure(results)
        
        print("2. Creating comprehensive analysis figure...")
        create_comprehensive_analysis_figure(results)
        
        print("3. Creating permutation visualization...")
        create_permutation_visualization_corrected(results)
        
        # Step 6: Save comprehensive results
        print("4. Saving comprehensive results to Excel...")
        save_comprehensive_results(results)
        
        # Step 7: Save pickle file
        print("5. Saving pickle file...")
        pickle_filename = 'comprehensive_corrected_44subject_results.pkl'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n{'='*70}")
        print("COMPREHENSIVE CORRECTED ANALYSIS COMPLETE!")
        print(f"{'='*70}")
        
        print("\nGenerated files:")
        output_files = [
            'corrected_44subject_publication.png - Publication-ready figure',
            'corrected_44subject_comprehensive.png - Multi-panel comprehensive figure', 
            'corrected_44subject_permutation.png - Detailed permutation test visualization',
            'corrected_44subject_comprehensive_results.xlsx - Complete Excel results',
            'comprehensive_corrected_44subject_results.pkl - Pickle file for further analysis'
        ]
        
        for file in output_files:
            print(f"  ✓ {file}")
        
        # Final summary
        sig_clusters = sum(1 for c in results['clusters'] if c.get('p_value_max', 1) < 0.05)
        cohens_d_values = [results['stats'][t]['cohens_d'] for t in results['time_columns'] if t in results['stats']]
        
        print(f"\nFinal Summary:")
        print(f"  • Participants analyzed: {len(results['high_reward']) + len(results['low_reward'])}")
        print(f"  • Clusters found: {len(results['clusters'])}")
        print(f"  • Significant clusters: {sig_clusters}")
        print(f"  • Mean effect size: d = {np.mean(cohens_d_values):.3f}")
        print(f"  • Analysis approach: Participant-level (CORRECTED)")
        print(f"  • Pseudoreplication: ELIMINATED")
        
        return results
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main_comprehensive_corrected()
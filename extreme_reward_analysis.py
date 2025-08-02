#!/usr/bin/env python3
"""
Extreme Reward Comparison: 0.8 vs 0.2 Pupil Response Analysis
Focuses on the most extreme reward conditions for maximum contrast
Can be easily adjusted for 0.7 vs 0.3 comparison
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

def load_and_clean_extreme_reward_data(timeseries_file: str = '/Users/zengsiyi/Desktop/stim1_test/aligned_files/stim1_aligned.csv', 
                                      high_reward_value: float = 0.8, 
                                      low_reward_value: float = 0.2) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and clean data for extreme reward comparison (0.8 vs 0.2 by default).
    
    Parameters:
    -----------
    timeseries_file : str
        Path to the CSV file
    high_reward_value : float
        The high reward value to analyze (default: 0.8)
    low_reward_value : float
        The low reward value to analyze (default: 0.2)
    """
    print(f"=== EXTREME REWARD COMPARISON: {high_reward_value} vs {low_reward_value} ===")
    print("Participant-level aggregation for maximum reward contrast\n")
    
    # Load data
    print("Step 1: Loading data...")
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
    
    print(f"Unique rp1 values in dataset: {sorted(timeseries_df['rp1'].unique())}")
    print(f"Target high reward value: {high_reward_value}")
    print(f"Target low reward value: {low_reward_value}")
    
    # Filter for only the extreme conditions
    initial_trials = len(timeseries_df)
    extreme_conditions = timeseries_df[
        (timeseries_df['rp1'] == high_reward_value) | 
        (timeseries_df['rp1'] == low_reward_value)
    ].copy()
    
    print(f"\nFiltered to extreme conditions: {len(extreme_conditions)} of {initial_trials} trials")
    
    # Check if we have both conditions
    available_conditions = extreme_conditions['rp1'].unique()
    if high_reward_value not in available_conditions:
        print(f"WARNING: High reward value {high_reward_value} not found in data!")
        print(f"Available values: {available_conditions}")
    if low_reward_value not in available_conditions:
        print(f"WARNING: Low reward value {low_reward_value} not found in data!")
        print(f"Available values: {available_conditions}")
    
    if len(available_conditions) < 2:
        raise ValueError("Need both high and low reward conditions in the data!")
    
    # Clean data
    print("\nStep 2: Cleaning data...")
    extreme_conditions = extreme_conditions.dropna(subset=time_columns)
    print(f"After removing missing data: {len(extreme_conditions)} trials")
    
    # Check trial distribution
    condition_counts = extreme_conditions['rp1'].value_counts()
    print(f"\nTrial distribution:")
    for condition, count in condition_counts.items():
        condition_label = "HIGH" if condition == high_reward_value else "LOW"
        print(f"  {condition} ({condition_label}): {count} trials")
    
    # Check subject distribution
    subject_counts = extreme_conditions['subject_id'].value_counts()
    print(f"\nSubject distribution:")
    print(f"  Total subjects: {len(subject_counts)}")
    print(f"  Trials per subject: min={subject_counts.min()}, max={subject_counts.max()}, mean={subject_counts.mean():.1f}")
    
    return extreme_conditions, time_columns

def aggregate_extreme_conditions(timeseries_df: pd.DataFrame, time_columns: List[str],
                                high_reward_value: float = 0.8, 
                                low_reward_value: float = 0.2) -> pd.DataFrame:
    """
    Aggregate data by participant for extreme reward conditions.
    """
    print(f"\n*** STEP 3: PARTICIPANT-LEVEL AGGREGATION FOR EXTREME CONDITIONS ***")
    print(f"Comparing {high_reward_value} (HIGH) vs {low_reward_value} (LOW)")
    
    # Create condition labels
    timeseries_df = timeseries_df.copy()
    timeseries_df['condition'] = timeseries_df['rp1'].apply(
        lambda x: 'high_reward' if x == high_reward_value else 
                 'low_reward' if x == low_reward_value else 'other'
    )
    
    # Remove any trials that don't match our conditions (shouldn't be any after filtering)
    before_filter = len(timeseries_df)
    timeseries_df = timeseries_df[timeseries_df['condition'] != 'other']
    print(f"Final filtered trials: {len(timeseries_df)} of {before_filter}")
    
    # Report condition distribution
    condition_counts = timeseries_df['condition'].value_counts()
    for condition, count in condition_counts.items():
        reward_val = high_reward_value if condition == 'high_reward' else low_reward_value
        print(f"  {condition} (rp1={reward_val}): {count} trials")
    
    # Aggregate by participant and condition
    print(f"\nAggregating across participants...")
    agg_data = []
    
    unique_subjects = sorted(timeseries_df['subject_id'].unique())
    subjects_with_both = 0
    subjects_with_high_only = 0
    subjects_with_low_only = 0
    
    for subject in unique_subjects:
        subject_data = timeseries_df[timeseries_df['subject_id'] == subject]
        
        has_high = len(subject_data[subject_data['condition'] == 'high_reward']) > 0
        has_low = len(subject_data[subject_data['condition'] == 'low_reward']) > 0
        
        if has_high and has_low:
            subjects_with_both += 1
        elif has_high:
            subjects_with_high_only += 1
        elif has_low:
            subjects_with_low_only += 1
        
        for condition in ['high_reward', 'low_reward']:
            subject_condition_data = subject_data[subject_data['condition'] == condition]
            
            if len(subject_condition_data) > 0:
                # Calculate mean across trials for this subject and condition
                subject_means = subject_condition_data[time_columns].mean()
                
                agg_row = {
                    'subject_id': subject,
                    'condition': condition,
                    'reward_value': high_reward_value if condition == 'high_reward' else low_reward_value,
                    'n_trials': len(subject_condition_data)
                }
                
                # Add time point means
                for time_col in time_columns:
                    agg_row[time_col] = subject_means[time_col]
                
                agg_data.append(agg_row)
    
    aggregated_df = pd.DataFrame(agg_data)
    
    print(f"\nParticipant Analysis:")
    print(f"  Total subjects: {len(unique_subjects)}")
    print(f"  Subjects with both conditions: {subjects_with_both}")
    print(f"  Subjects with HIGH only: {subjects_with_high_only}")
    print(f"  Subjects with LOW only: {subjects_with_low_only}")
    
    # Final sample sizes
    final_high = len(aggregated_df[aggregated_df['condition'] == 'high_reward'])
    final_low = len(aggregated_df[aggregated_df['condition'] == 'low_reward'])
    
    print(f"\n*** FINAL SAMPLE SIZES FOR EXTREME COMPARISON ***")
    print(f"High reward ({high_reward_value}): {final_high} participants")
    print(f"Low reward ({low_reward_value}): {final_low} participants")
    print(f"Reward contrast: {high_reward_value - low_reward_value:.1f} (maximum possible)")
    
    return aggregated_df

def perform_extreme_cluster_test(aggregated_df: pd.DataFrame, time_columns: List[str], 
                               high_reward_value: float = 0.8, low_reward_value: float = 0.2,
                               t_threshold: float = 2.0, n_permutations: int = 1000) -> Dict:
    """
    Perform cluster-based permutation test for extreme reward comparison.
    """
    print(f"\n*** STEP 4: CLUSTER-BASED PERMUTATION TEST ***")
    print(f"Extreme contrast: {high_reward_value} vs {low_reward_value}")
    print(f"Running {n_permutations} permutations on participant-level data")
    
    # Separate by condition
    high_reward = aggregated_df[aggregated_df['condition'] == 'high_reward']
    low_reward = aggregated_df[aggregated_df['condition'] == 'low_reward']
    
    print(f"\nData for extreme comparison:")
    print(f"  High reward ({high_reward_value}): {len(high_reward)} participants")
    print(f"  Low reward ({low_reward_value}): {len(low_reward)} participants")
    print(f"  Time points: {len(time_columns)}")
    print(f"  Reward difference: {high_reward_value - low_reward_value:.1f}")
    
    # Calculate observed statistics
    print(f"\nCalculating observed statistics for extreme contrast...")
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
    
    print(f"\nObserved clusters for extreme contrast:")
    if observed_clusters:
        for i, cluster in enumerate(observed_clusters):
            print(f"  Cluster {i+1}: {cluster['start_time']:.2f}s - {cluster['end_time']:.2f}s")
            print(f"    Direction: {cluster['direction']}")
            print(f"    Cluster statistic: {abs(cluster['t_sum']):.2f}")
            print(f"    Size: {cluster['size']} time points")
            print(f"    Max |t|: {cluster['max_t']:.2f}")
    else:
        print("  No clusters found above threshold")
    
    # Permutation test
    print(f"\nRunning {n_permutations} permutations for extreme contrast...")
    
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
    print(f"\nCalculating cluster p-values for extreme contrast...")
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
        'aggregated_df': aggregated_df,
        'high_reward_value': high_reward_value,
        'low_reward_value': low_reward_value,
        'reward_contrast': high_reward_value - low_reward_value
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

def create_extreme_comparison_figure(results: Dict, save_path: str = None):
    """
    Create publication-ready figure for extreme reward comparison.
    """
    if save_path is None:
        high_val = results['high_reward_value']
        low_val = results['low_reward_value']
        save_path = f'extreme_reward_{high_val}vs{low_val}_analysis.png'
    
    stats_dict = results['stats']
    clusters = results['clusters']
    time_columns = results['time_columns']
    high_val = results['high_reward_value']
    low_val = results['low_reward_value']
    
    # Prepare data
    times = np.array([float(t) for t in time_columns if t in stats_dict])
    high_means = np.array([stats_dict[t]['high_mean'] for t in time_columns if t in stats_dict])
    low_means = np.array([stats_dict[t]['low_mean'] for t in time_columns if t in stats_dict])
    high_ses = np.array([stats_dict[t]['high_se'] for t in time_columns if t in stats_dict])
    low_ses = np.array([stats_dict[t]['low_se'] for t in time_columns if t in stats_dict])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Plot means with error bands
    ax.plot(times, high_means, color='#d62728', linewidth=3.5, 
            label=f'High reward ({high_val}) - N={len(results["high_reward"])}', zorder=3)
    ax.fill_between(times, high_means - high_ses, high_means + high_ses, 
                   alpha=0.3, color='#d62728', zorder=2)
    
    ax.plot(times, low_means, color='#1f77b4', linewidth=3.5, 
            label=f'Low reward ({low_val}) - N={len(results["low_reward"])}', zorder=3)
    ax.fill_between(times, low_means - low_ses, low_means + low_ses, 
                   alpha=0.3, color='#1f77b4', zorder=2)
    
    # Mark stimulus onset
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2.5, 
              alpha=0.7, zorder=1, label='Stimulus onset')
    
    # Highlight significant clusters
    sig_clusters = [c for c in clusters if c.get('p_value_max', 1) < 0.05]
    for i, cluster in enumerate(sig_clusters):
        ax.axvspan(cluster['start_time'], cluster['end_time'], 
                  alpha=0.25, color='gold', zorder=0)
        
        y_pos = ax.get_ylim()[1] * 0.92
        x_pos = (cluster['start_time'] + cluster['end_time']) / 2
        ax.text(x_pos, y_pos, f'***', ha='center', va='top', 
               fontsize=18, fontweight='bold', color='red')
    
    # Add non-significant clusters with light shading
    nonsig_clusters = [c for c in clusters if c.get('p_value_max', 1) >= 0.05]
    for cluster in nonsig_clusters:
        ax.axvspan(cluster['start_time'], cluster['end_time'], 
                  alpha=0.08, color='gray', zorder=0)
    
    # Formatting
    ax.set_xlabel('Time from stimulus onset (s)', fontsize=15)
    ax.set_ylabel('Pupil size (participant means)', fontsize=15)
    
    contrast_value = results['reward_contrast']
    ax.set_title(f'Extreme Reward Contrast: {high_val} vs {low_val} (Δ = {contrast_value:.1f})\n'
                f'Maximum Possible Reward Difference', 
                fontsize=16, fontweight='bold')
    ax.set_xlim(-0.5, 3)
    
    # Enhanced legend
    legend_elements = ax.get_legend_handles_labels()[0][:3]
    if sig_clusters:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gold', alpha=0.25, 
                                           label=f'Significant cluster (p < 0.05)'))
    if nonsig_clusters:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.08, 
                                           label='Non-significant cluster'))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=13, 
             frameon=True, fancybox=True, shadow=True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add effect size information
    cohens_d_values = [stats_dict[t]['cohens_d'] for t in time_columns if t in stats_dict]
    max_effect = np.max(np.abs(cohens_d_values))
    mean_effect = np.mean(cohens_d_values)
    
    # Add text box with key statistics
    textstr = f'Max |Cohen\'s d|: {max_effect:.3f}\nMean Cohen\'s d: {mean_effect:.3f}\nReward contrast: {contrast_value:.1f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Extreme comparison figure saved to {save_path}")
    
    return fig

def print_extreme_comparison_results(results: Dict):
    """
    Print detailed results for extreme reward comparison.
    """
    clusters = results['clusters']
    stats_dict = results['stats']
    high_val = results['high_reward_value']
    low_val = results['low_reward_value']
    contrast = results['reward_contrast']
    
    print(f"\n{'='*70}")
    print(f"EXTREME REWARD COMPARISON RESULTS: {high_val} vs {low_val}")
    print(f"{'='*70}")
    
    print(f"\n*** EXPERIMENTAL DESIGN ***")
    print(f"High reward condition: {high_val}")
    print(f"Low reward condition: {low_val}")
    print(f"Reward contrast: {contrast:.1f} (maximum possible)")
    print(f"High reward participants: {len(results['high_reward'])}")
    print(f"Low reward participants: {len(results['low_reward'])}")
    print(f"Time points analyzed: {len([t for t in results['time_columns'] if t in stats_dict])}")
    print(f"Analysis type: Participant-level aggregation (CORRECTED)")
    
    print(f"\n*** CLUSTER ANALYSIS ***")
    print(f"Total clusters found: {len(clusters)}")
    
    sig_clusters = [c for c in clusters if c.get('p_value_max', 1) < 0.05]
    print(f"Significant clusters (p < 0.05): {len(sig_clusters)}")
    
    if clusters:
        print(f"\nDetailed cluster information:")
        for i, cluster in enumerate(clusters):
            p_corrected = cluster.get('p_value_max', 1)
            is_sig = p_corrected < 0.05
            
            print(f"\n  Cluster {i+1}:")
            print(f"    Time window: {cluster['start_time']:.2f}s to {cluster['end_time']:.2f}s")
            print(f"    Duration: {cluster['end_time'] - cluster['start_time']:.2f}s")
            print(f"    Direction: {cluster['direction']}")
            print(f"    Cluster statistic: {abs(cluster['t_sum']):.2f}")
            print(f"    Maximum |t| in cluster: {cluster['max_t']:.2f}")
            print(f"    Size: {cluster['size']} time points")
            print(f"    p-value (corrected): {p_corrected:.4f} {'***' if is_sig else ''}")
    else:
        print("  No clusters found above threshold")
    
    print(f"\n*** EFFECT SIZE ANALYSIS ***")
    cohens_d_values = [stats_dict[t]['cohens_d'] for t in results['time_columns'] if t in stats_dict]
    mean_diffs = [stats_dict[t]['mean_diff'] for t in results['time_columns'] if t in stats_dict]
    
    print(f"Mean Cohen's d across time: {np.mean(cohens_d_values):.3f}")
    print(f"Maximum |Cohen's d|: {np.max(np.abs(cohens_d_values)):.3f}")
    print(f"Mean difference ({high_val} - {low_val}): {np.mean(mean_diffs):.3f}")
    print(f"Range of differences: {np.min(mean_diffs):.3f} to {np.max(mean_diffs):.3f}")
    
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
    print(f"Effect size with maximum contrast ({contrast:.1f}): More sensitive than grouped analysis")
    
    print(f"\n*** COMPARISON TO GROUPED ANALYSIS ***")
    print(f"Advantage of extreme contrast:")
    print(f"  • Maximum reward difference: {contrast:.1f} vs 0.5 (grouped: 0.7,0.8 vs 0.2,0.3)")
    print(f"  • Cleaner contrast: No within-group variability")
    print(f"  • Higher effect sizes expected due to maximum separation")
    
    print(f"\n*** CONCLUSION ***")
    if sig_clusters:
        print(f"✓ Found {len(sig_clusters)} significant cluster(s) with extreme contrast")
        print("✓ Maximum reward difference provides strongest test of effect")
    else:
        print("✗ No significant clusters found even with maximum contrast")
        print("• This suggests either no effect or very small effect size")
        print("• Consider power analysis for future studies")

def save_extreme_comparison_results(results: Dict, filename: str = None):
    """
    Save comprehensive results for extreme reward comparison.
    """
    if filename is None:
        high_val = results['high_reward_value']
        low_val = results['low_reward_value']
        filename = f'extreme_reward_{high_val}vs{low_val}_results.xlsx'
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Time series statistics
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
                    'mean_diff': stat['mean_diff'],
                    't_stat': stat['t_stat'],
                    'p_val': stat['p_val'],
                    'cohens_d': stat['cohens_d'],
                    'ci_lower': stat['ci_lower'],
                    'ci_upper': stat['ci_upper'],
                    'significant_uncorrected': stat['p_val'] < 0.05
                })
        
        time_series_df = pd.DataFrame(time_series_data)
        time_series_df.to_excel(writer, sheet_name='Time_Series_Stats', index=False)
        
        # Cluster analysis
        if results['clusters']:
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
                    'significant_corrected': cluster.get('p_value_max', 1) < 0.05
                }
                cluster_data.append(cluster_info)
            
            cluster_df = pd.DataFrame(cluster_data)
            cluster_df.to_excel(writer, sheet_name='Cluster_Analysis', index=False)
        
        # Analysis summary
        sig_clusters = sum(1 for c in results['clusters'] if c.get('p_value_max', 1) < 0.05)
        cohens_d_values = [results['stats'][t]['cohens_d'] for t in results['time_columns'] if t in results['stats']]
        
        summary_info = pd.DataFrame({
            'metric': [
                'High Reward Value',
                'Low Reward Value',
                'Reward Contrast',
                'High Reward Participants',
                'Low Reward Participants',
                'Total Time Points',
                'Clusters Found',
                'Significant Clusters',
                'Mean Effect Size (Cohen\'s d)',
                'Max Effect Size (Cohen\'s d)',
                'Analysis Type'
            ],
            'value': [
                results['high_reward_value'],
                results['low_reward_value'],
                results['reward_contrast'],
                len(results['high_reward']),
                len(results['low_reward']),
                len([t for t in results['time_columns'] if t in results['stats']]),
                len(results['clusters']),
                sig_clusters,
                np.mean(cohens_d_values),
                np.max(np.abs(cohens_d_values)),
                'Extreme Contrast Analysis'
            ]
        })
        summary_info.to_excel(writer, sheet_name='Analysis_Summary', index=False)
        
        # Participant data
        results['aggregated_df'].to_excel(writer, sheet_name='Participant_Data', index=False)
    
    print(f"Extreme comparison results saved to {filename}")

def main_extreme_reward_analysis(high_reward_value: float = 0.8, 
                                low_reward_value: float = 0.2,
                                timeseries_file: str = '/Users/zengsiyi/Desktop/stim1_test/aligned_files/stim1_aligned.csv'):
    """
    Main function for extreme reward comparison analysis.
    
    Parameters:
    -----------
    high_reward_value : float
        The high reward value to analyze (default: 0.8)
        For 0.7 vs 0.3 analysis, set this to 0.7
    low_reward_value : float  
        The low reward value to analyze (default: 0.2)
        For 0.7 vs 0.3 analysis, set this to 0.3
    timeseries_file : str
        Path to the CSV file
    """
    print(f"Starting extreme reward analysis: {high_reward_value} vs {low_reward_value}")
    print("This analysis focuses on maximum reward contrast for clearest effects.\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Check for data file
    if not os.path.exists(timeseries_file):
        print(f"Error: {timeseries_file} not found!")
        return None
    
    try:
        # Step 1: Load and clean data for extreme conditions
        extreme_data, time_columns = load_and_clean_extreme_reward_data(
            timeseries_file, high_reward_value, low_reward_value
        )
        
        if len(extreme_data) == 0:
            print("Error: No data found for specified reward values!")
            return None
        
        # Step 2: Aggregate by participant
        aggregated_df = aggregate_extreme_conditions(
            extreme_data, time_columns, high_reward_value, low_reward_value
        )
        
        if len(aggregated_df) == 0:
            print("Error: No data after aggregation!")
            return None
        
        # Step 3: Cluster analysis
        results = perform_extreme_cluster_test(
            aggregated_df, time_columns, high_reward_value, low_reward_value,
            t_threshold=2.0, n_permutations=1000
        )
        
        # Step 4: Print results
        print_extreme_comparison_results(results)
        
        # Step 5: Create visualization
        print(f"\n*** CREATING VISUALIZATION ***")
        create_extreme_comparison_figure(results)
        
        # Step 6: Save results
        print("*** SAVING RESULTS ***")
        save_extreme_comparison_results(results)
        
        # Step 7: Save pickle file
        pickle_filename = f'extreme_reward_{high_reward_value}vs{low_reward_value}_results.pkl'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n{'='*70}")
        print("EXTREME REWARD ANALYSIS COMPLETE!")
        print(f"{'='*70}")
        
        print(f"\nGenerated files:")
        print(f"  ✓ extreme_reward_{high_reward_value}vs{low_reward_value}_analysis.png")
        print(f"  ✓ extreme_reward_{high_reward_value}vs{low_reward_value}_results.xlsx")
        print(f"  ✓ {pickle_filename}")
        
        # Summary
        sig_clusters = sum(1 for c in results['clusters'] if c.get('p_value_max', 1) < 0.05)
        cohens_d_values = [results['stats'][t]['cohens_d'] for t in results['time_columns'] if t in results['stats']]
        
        print(f"\nFinal Summary:")
        print(f"  • Reward contrast: {high_reward_value} vs {low_reward_value} (Δ = {results['reward_contrast']:.1f})")
        print(f"  • Participants: {len(results['high_reward']) + len(results['low_reward'])}")
        print(f"  • Significant clusters: {sig_clusters}")
        print(f"  • Mean effect size: d = {np.mean(cohens_d_values):.3f}")
        print(f"  • Max effect size: d = {np.max(np.abs(cohens_d_values)):.3f}")
        
        return results
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# INSTRUCTIONS FOR DIFFERENT COMPARISONS
# =============================================================================

def instructions_for_different_comparisons():
    """
    Print instructions for running different reward comparisons.
    """
    print("""
    =============================================================================
    INSTRUCTIONS FOR DIFFERENT REWARD COMPARISONS
    =============================================================================
    
    This script can be easily adjusted for different reward value comparisons:
    
    1. FOR 0.8 vs 0.2 COMPARISON (DEFAULT - MAXIMUM CONTRAST):
       results = main_extreme_reward_analysis(
           high_reward_value=0.8,
           low_reward_value=0.2
       )
    
    2. FOR 0.7 vs 0.3 COMPARISON (MODERATE CONTRAST):
       results = main_extreme_reward_analysis(
           high_reward_value=0.7,
           low_reward_value=0.3
       )
    
    3. FOR ANY OTHER PAIR:
       results = main_extreme_reward_analysis(
           high_reward_value=YOUR_HIGH_VALUE,
           low_reward_value=YOUR_LOW_VALUE
       )
    
    =============================================================================
    ADVANTAGES OF DIFFERENT APPROACHES:
    =============================================================================
    
    ORIGINAL GROUPED ANALYSIS (0.7,0.8 vs 0.2,0.3):
    ✓ More trials per condition (higher power)
    ✓ More generalizable across reward levels
    ✗ Smaller effect size due to within-group variability
    ✗ Less precise contrast
    
    EXTREME CONTRAST (0.8 vs 0.2):
    ✓ Maximum reward difference (0.6)
    ✓ Cleanest contrast, highest effect sizes
    ✓ Most sensitive test of reward effects
    ✗ Fewer trials (lower power)
    ✗ Less generalizable
    
    MODERATE CONTRAST (0.7 vs 0.3):
    ✓ Good balance of contrast (0.4) and trial count
    ✓ Still cleaner than grouped analysis
    ✓ Reasonable effect sizes
    ◦ Intermediate power and generalizability
    
    =============================================================================
    RECOMMENDATIONS:
    =============================================================================
    
    1. Run 0.8 vs 0.2 first (maximum sensitivity)
    2. If significant, run 0.7 vs 0.3 (replication with different values)
    3. Compare results to grouped analysis for completeness
    4. Use the approach that best matches your research question:
       - Extreme contrast: "Do maximum reward differences affect pupils?"
       - Grouped analysis: "Do high vs low reward categories affect pupils?"
    
    """)

if __name__ == "__main__":
    # Print instructions
    instructions_for_different_comparisons()
    
    print("Choose your analysis:")
    print("1. Run 0.8 vs 0.2 analysis (maximum contrast)")
    print("2. Run 0.7 vs 0.3 analysis (moderate contrast)")
    print("3. See instructions only")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        print("\n" + "="*50)
        print("RUNNING 0.8 vs 0.2 ANALYSIS")
        print("="*50)
        results = main_extreme_reward_analysis(
            high_reward_value=0.8,
            low_reward_value=0.2
        )
    
    elif choice == "2":
        print("\n" + "="*50)
        print("RUNNING 0.7 vs 0.3 ANALYSIS")
        print("="*50)
        results = main_extreme_reward_analysis(
            high_reward_value=0.7,
            low_reward_value=0.3
        )
    
    elif choice == "3":
        print("\nInstructions displayed above. Run the script again to perform analysis.")
    
    else:
        print("Invalid choice. Running default 0.8 vs 0.2 analysis...")
        results = main_extreme_reward_analysis(
            high_reward_value=0.8,
            low_reward_value=0.2
        )
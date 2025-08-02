#!/usr/bin/env python3
"""
Value Difference Magnitude Analysis: Large vs Small Value Differences
Compares pupil responses between:
- Large difference group: rp1 = 0.8, 0.2 (extreme values)
- Small difference group: rp1 = 0.7, 0.3 (moderate values)

This tests whether participants show different responses to stimuli with 
large vs small value differences from the reference point.
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

def load_and_clean_value_difference_data(timeseries_file: str = 'stim1_aligned.csv') -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and clean data for value difference magnitude analysis.
    Groups stimuli by their distance from reference point rather than absolute value.
    """
    print("=== VALUE DIFFERENCE MAGNITUDE ANALYSIS ===")
    print("Comparing large vs small value differences from reference point\n")
    
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
    
    # Define reference point (middle value)
    reference_point = 0.5  # Midpoint between 0 and 1
    print(f"Using reference point: {reference_point}")
    
    # Calculate value differences and group assignment
    print(f"\nStep 2: Grouping by value difference magnitude...")
    
    # Large difference group: 0.8, 0.2 (distance = 0.3, 0.3)
    large_diff_values = [0.8, 0.2]
    large_diff_distances = [abs(v - reference_point) for v in large_diff_values]
    
    # Small difference group: 0.7, 0.3 (distance = 0.2, 0.2) 
    small_diff_values = [0.7, 0.3]
    small_diff_distances = [abs(v - reference_point) for v in small_diff_values]
    
    print(f"Large difference group:")
    for val, dist in zip(large_diff_values, large_diff_distances):
        print(f"  rp1 = {val}, distance from reference = {dist}")
    
    print(f"Small difference group:")
    for val, dist in zip(small_diff_values, small_diff_distances):
        print(f"  rp1 = {val}, distance from reference = {dist}")
    
    # Filter for our target values
    target_values = large_diff_values + small_diff_values
    initial_trials = len(timeseries_df)
    timeseries_df = timeseries_df[timeseries_df['rp1'].isin(target_values)].copy()
    
    print(f"\nFiltered to target values: {len(timeseries_df)} of {initial_trials} trials")
    
    # Clean data
    print(f"\nStep 3: Cleaning data...")
    timeseries_df = timeseries_df.dropna(subset=time_columns)
    print(f"After removing missing data: {len(timeseries_df)} trials")
    
    # Check distribution
    value_counts = timeseries_df['rp1'].value_counts().sort_index()
    print(f"\nTrial distribution by rp1 value:")
    for value, count in value_counts.items():
        group = "LARGE DIFF" if value in large_diff_values else "SMALL DIFF"
        dist = abs(value - reference_point)
        print(f"  rp1 = {value} ({group}, distance = {dist}): {count} trials")
    
    # Check subject distribution
    subject_counts = timeseries_df['subject_id'].value_counts()
    print(f"\nSubject distribution:")
    print(f"  Total subjects: {len(subject_counts)}")
    print(f"  Trials per subject: min={subject_counts.min()}, max={subject_counts.max()}, mean={subject_counts.mean():.1f}")
    
    return timeseries_df, time_columns

def aggregate_by_value_difference(timeseries_df: pd.DataFrame, time_columns: List[str]) -> pd.DataFrame:
    """
    Aggregate data by participant for value difference magnitude groups.
    """
    print(f"\n*** STEP 4: PARTICIPANT-LEVEL AGGREGATION BY VALUE DIFFERENCE ***")
    print("Grouping trials by magnitude of difference from reference point")
    
    # Define reference point and groups
    reference_point = 0.5
    large_diff_values = [0.8, 0.2]  # Distance = 0.3
    small_diff_values = [0.7, 0.3]  # Distance = 0.2
    
    # Create condition labels based on value difference magnitude
    timeseries_df = timeseries_df.copy()
    timeseries_df['condition'] = timeseries_df['rp1'].apply(
        lambda x: 'large_difference' if x in large_diff_values else 
                 'small_difference' if x in small_diff_values else 'other'
    )
    
    # Remove any trials that don't fit our conditions (shouldn't be any after filtering)
    before_filter = len(timeseries_df)
    timeseries_df = timeseries_df[timeseries_df['condition'] != 'other']
    print(f"Final filtered trials: {len(timeseries_df)} of {before_filter}")
    
    # Report condition distribution
    condition_counts = timeseries_df['condition'].value_counts()
    for condition, count in condition_counts.items():
        values = large_diff_values if condition == 'large_difference' else small_diff_values
        distances = [abs(v - reference_point) for v in values]
        print(f"  {condition} (rp1={values}, distances={distances}): {count} trials")
    
    # Show breakdown by individual values within each group
    print(f"\nDetailed breakdown:")
    for condition in ['large_difference', 'small_difference']:
        condition_data = timeseries_df[timeseries_df['condition'] == condition]
        value_counts = condition_data['rp1'].value_counts().sort_index()
        print(f"  {condition}:")
        for value, count in value_counts.items():
            print(f"    rp1 = {value}: {count} trials")
    
    # Aggregate by participant and condition
    print(f"\nAggregating across participants...")
    agg_data = []
    
    unique_subjects = sorted(timeseries_df['subject_id'].unique())
    subjects_with_both = 0
    subjects_with_large_only = 0
    subjects_with_small_only = 0
    
    for subject in unique_subjects:
        subject_data = timeseries_df[timeseries_df['subject_id'] == subject]
        
        has_large = len(subject_data[subject_data['condition'] == 'large_difference']) > 0
        has_small = len(subject_data[subject_data['condition'] == 'small_difference']) > 0
        
        if has_large and has_small:
            subjects_with_both += 1
        elif has_large:
            subjects_with_large_only += 1
        elif has_small:
            subjects_with_small_only += 1
        
        for condition in ['large_difference', 'small_difference']:
            subject_condition_data = subject_data[subject_data['condition'] == condition]
            
            if len(subject_condition_data) > 0:
                # Calculate mean across trials for this subject and condition
                subject_means = subject_condition_data[time_columns].mean()
                
                # Calculate additional metrics for this subject-condition
                condition_values = subject_condition_data['rp1'].values
                mean_distance = np.mean([abs(v - 0.5) for v in condition_values])
                
                agg_row = {
                    'subject_id': subject,
                    'condition': condition,
                    'n_trials': len(subject_condition_data),
                    'mean_distance_from_ref': mean_distance,
                    'rp1_values_in_condition': ', '.join([str(v) for v in sorted(set(condition_values))])
                }
                
                # Add time point means
                for time_col in time_columns:
                    agg_row[time_col] = subject_means[time_col]
                
                agg_data.append(agg_row)
    
    aggregated_df = pd.DataFrame(agg_data)
    
    print(f"\nParticipant Analysis:")
    print(f"  Total subjects: {len(unique_subjects)}")
    print(f"  Subjects with both conditions: {subjects_with_both}")
    print(f"  Subjects with LARGE difference only: {subjects_with_large_only}")
    print(f"  Subjects with SMALL difference only: {subjects_with_small_only}")
    
    # Final sample sizes
    final_large = len(aggregated_df[aggregated_df['condition'] == 'large_difference'])
    final_small = len(aggregated_df[aggregated_df['condition'] == 'small_difference'])
    
    print(f"\n*** FINAL SAMPLE SIZES FOR VALUE DIFFERENCE ANALYSIS ***")
    print(f"Large difference condition (0.8, 0.2): {final_large} participants")
    print(f"Small difference condition (0.7, 0.3): {final_small} participants")
    print(f"Difference in distances: 0.3 vs 0.2 = 0.1 difference in magnitude")
    
    return aggregated_df

def perform_value_difference_cluster_test(aggregated_df: pd.DataFrame, time_columns: List[str], 
                                        t_threshold: float = 2.0, n_permutations: int = 1000) -> Dict:
    """
    Perform cluster-based permutation test for value difference magnitude comparison.
    """
    print(f"\n*** STEP 5: CLUSTER-BASED PERMUTATION TEST ***")
    print(f"Testing: Large value differences vs Small value differences")
    print(f"Running {n_permutations} permutations on participant-level data")
    
    # Separate by condition
    large_diff = aggregated_df[aggregated_df['condition'] == 'large_difference']
    small_diff = aggregated_df[aggregated_df['condition'] == 'small_difference']
    
    print(f"\nData for value difference comparison:")
    print(f"  Large difference (0.8, 0.2): {len(large_diff)} participants")
    print(f"  Small difference (0.7, 0.3): {len(small_diff)} participants")
    print(f"  Time points: {len(time_columns)}")
    print(f"  Hypothesis: Different responses to extreme vs moderate value distances")
    
    # Calculate observed statistics
    print(f"\nCalculating observed statistics for value difference comparison...")
    observed_stats = {}
    
    for time in time_columns:
        large_values = large_diff[time].values
        small_values = small_diff[time].values
        
        # Remove NaN values
        large_values = large_values[~np.isnan(large_values)]
        small_values = small_values[~np.isnan(small_values)]
        
        if len(large_values) == 0 or len(small_values) == 0:
            continue
            
        # Calculate comprehensive statistics
        large_mean = np.mean(large_values)
        small_mean = np.mean(small_values)
        large_std = np.std(large_values, ddof=1)
        small_std = np.std(small_values, ddof=1)
        large_se = large_std / np.sqrt(len(large_values))
        small_se = small_std / np.sqrt(len(small_values))
        
        # Statistical tests
        t_stat, p_val = stats.ttest_ind(large_values, small_values)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(large_values) - 1) * large_std**2 + 
                             (len(small_values) - 1) * small_std**2) / 
                            (len(large_values) + len(small_values) - 2))
        cohens_d = (large_mean - small_mean) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for difference
        diff = large_mean - small_mean
        se_diff = np.sqrt(large_se**2 + small_se**2)
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        
        observed_stats[time] = {
            'large_mean': large_mean,
            'small_mean': small_mean,
            'large_std': large_std,
            'small_std': small_std,
            'large_se': large_se,
            'small_se': small_se,
            'mean_diff': diff,
            't_stat': t_stat,
            'p_val': p_val,
            'cohens_d': cohens_d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'large_n': len(large_values),
            'small_n': len(small_values)
        }
    
    # Find observed clusters
    observed_clusters = find_clusters_comprehensive(observed_stats, time_columns, t_threshold)
    
    print(f"\nObserved clusters for value difference comparison:")
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
    print(f"\nRunning {n_permutations} permutations for value difference comparison...")
    
    all_participant_data = aggregated_df.copy()
    null_cluster_stats = []
    max_cluster_stats_per_perm = []
    
    for perm_i in tqdm(range(n_permutations), desc="Permutations"):
        # Shuffle condition labels
        perm_data = all_participant_data.copy()
        shuffled_conditions = np.random.permutation(perm_data['condition'].values)
        perm_data['condition'] = shuffled_conditions
        
        # Calculate permuted statistics
        perm_large = perm_data[perm_data['condition'] == 'large_difference']
        perm_small = perm_data[perm_data['condition'] == 'small_difference']
        
        perm_stats = {}
        for time in time_columns:
            if time not in observed_stats:
                continue
                
            large_vals = perm_large[time].values
            small_vals = perm_small[time].values
            
            large_vals = large_vals[~np.isnan(large_vals)]
            small_vals = small_vals[~np.isnan(small_vals)]
            
            if len(large_vals) == 0 or len(small_vals) == 0:
                continue
            
            t_stat, _ = stats.ttest_ind(large_vals, small_vals)
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
    print(f"\nCalculating cluster p-values for value difference comparison...")
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
        'large_diff': large_diff,
        'small_diff': small_diff,
        'time_columns': time_columns,
        'n_permutations': n_permutations,
        't_threshold': t_threshold,
        'aggregated_df': aggregated_df,
        'analysis_type': 'value_difference_magnitude'
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
                    'direction': 'large > small' if t_stat > 0 else 'small > large',
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
                    'direction': 'large > small' if t_stat > 0 else 'small > large',
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

def create_value_difference_figure(results: Dict, save_path: str = 'value_difference_magnitude_analysis.png'):
    """
    Create publication-ready figure for value difference magnitude comparison.
    """
    stats_dict = results['stats']
    clusters = results['clusters']
    time_columns = results['time_columns']
    
    # Prepare data
    times = np.array([float(t) for t in time_columns if t in stats_dict])
    large_means = np.array([stats_dict[t]['large_mean'] for t in time_columns if t in stats_dict])
    small_means = np.array([stats_dict[t]['small_mean'] for t in time_columns if t in stats_dict])
    large_ses = np.array([stats_dict[t]['large_se'] for t in time_columns if t in stats_dict])
    small_ses = np.array([stats_dict[t]['small_se'] for t in time_columns if t in stats_dict])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    # Plot means with error bands
    ax.plot(times, large_means, color='#e74c3c', linewidth=3.5, 
            label=f'Large difference (0.8, 0.2) - N={len(results["large_diff"])}', zorder=3)
    ax.fill_between(times, large_means - large_ses, large_means + large_ses, 
                   alpha=0.3, color='#e74c3c', zorder=2)
    
    ax.plot(times, small_means, color='#2ecc71', linewidth=3.5, 
            label=f'Small difference (0.7, 0.3) - N={len(results["small_diff"])}', zorder=3)
    ax.fill_between(times, small_means - small_ses, small_means + small_ses, 
                   alpha=0.3, color='#2ecc71', zorder=2)
    
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
    
    ax.set_title('Value Difference Magnitude Analysis\n'
                'Large Differences (0.8, 0.2) vs Small Differences (0.7, 0.3)', 
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
    textstr = f'Max |Cohen\'s d|: {max_effect:.3f}\nMean Cohen\'s d: {mean_effect:.3f}\nDistance diff: 0.3 vs 0.2'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Value difference magnitude figure saved to {save_path}")
    
    return fig

def print_value_difference_results(results: Dict):
    """
    Print detailed results for value difference magnitude comparison.
    """
    clusters = results['clusters']
    stats_dict = results['stats']
    
    print(f"\n{'='*70}")
    print(f"VALUE DIFFERENCE MAGNITUDE ANALYSIS RESULTS")
    print(f"{'='*70}")
    
    print(f"\n*** EXPERIMENTAL DESIGN ***")
    print(f"Large difference group: rp1 = 0.8, 0.2 (distances = 0.3, 0.3 from 0.5)")
    print(f"Small difference group: rp1 = 0.7, 0.3 (distances = 0.2, 0.2 from 0.5)")
    print(f"Hypothesis: Different pupil responses to extreme vs moderate value distances")
    print(f"Large difference participants: {len(results['large_diff'])}")
    print(f"Small difference participants: {len(results['small_diff'])}")
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
    print(f"Mean difference (large - small): {np.mean(mean_diffs):.3f}")
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
    
    print(f"\n*** INTERPRETATION ***")
    print(f"Testing whether pupil responses differ based on value distance magnitude:")
    print(f"  • Large distances (0.3 from reference): More extreme values")
    print(f"  • Small distances (0.2 from reference): More moderate values")
    print(f"  • This tests sensitivity to value extremity rather than absolute reward")
    
    print(f"\n*** CONCLUSION ***")
    if sig_clusters:
        print(f"✓ Found {len(sig_clusters)} significant cluster(s)")
        print("✓ Pupils respond differently to extreme vs moderate value distances")
        print("✓ Evidence for value distance sensitivity in pupil responses")
    else:
        print("✗ No significant clusters found")
        print("• Pupils may not distinguish between extreme vs moderate value distances")
        print("• Or effect size too small to detect with current sample")

def save_value_difference_results(results: Dict, filename: str = 'value_difference_magnitude_results.xlsx'):
    """
    Save comprehensive results for value difference magnitude analysis.
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Time series statistics
        time_series_data = []
        for time in results['time_columns']:
            if time in results['stats']:
                stat = results['stats'][time]
                time_series_data.append({
                    'time': float(time),
                    'large_diff_mean': stat['large_mean'],
                    'small_diff_mean': stat['small_mean'],
                    'large_diff_se': stat['large_se'],
                    'small_diff_se': stat['small_se'],
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
                'Large Difference Participants',
                'Small Difference Participants',
                'Large Diff Values',
                'Small Diff Values',
                'Large Diff Distances',
                'Small Diff Distances',
                'Total Time Points',
                'Clusters Found',
                'Significant Clusters',
                'Mean Effect Size (Cohen\'s d)',
                'Max Effect Size (Cohen\'s d)',
                'Analysis Type'
            ],
            'value': [
                len(results['large_diff']),
                len(results['small_diff']),
                '0.8, 0.2',
                '0.7, 0.3',
                '0.3, 0.3',
                '0.2, 0.2',
                len([t for t in results['time_columns'] if t in results['stats']]),
                len(results['clusters']),
                sig_clusters,
                np.mean(cohens_d_values),
                np.max(np.abs(cohens_d_values)),
                'Value Difference Magnitude'
            ]
        })
        summary_info.to_excel(writer, sheet_name='Analysis_Summary', index=False)
        
        # Participant data
        results['aggregated_df'].to_excel(writer, sheet_name='Participant_Data', index=False)
    
    print(f"Value difference magnitude results saved to {filename}")

def main_value_difference_analysis(timeseries_file: str = 'stim1_aligned.csv'):
    """
    Main function for value difference magnitude analysis.
    
    This analysis tests whether pupil responses differ between:
    - Large value differences: rp1 = 0.8, 0.2 (distance 0.3 from reference 0.5)
    - Small value differences: rp1 = 0.7, 0.3 (distance 0.2 from reference 0.5)
    """
    print("Starting value difference magnitude analysis...")
    print("Testing responses to extreme vs moderate value distances.\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Check for data file
    if not os.path.exists(timeseries_file):
        print(f"Error: {timeseries_file} not found!")
        return None
    
    try:
        # Step 1: Load and clean data for value difference groups
        value_diff_data, time_columns = load_and_clean_value_difference_data(timeseries_file)
        
        if len(value_diff_data) == 0:
            print("Error: No data found for specified value difference groups!")
            return None
        
        # Step 2: Aggregate by participant and value difference magnitude
        aggregated_df = aggregate_by_value_difference(value_diff_data, time_columns)
        
        if len(aggregated_df) == 0:
            print("Error: No data after aggregation!")
            return None
        
        # Step 3: Cluster analysis
        results = perform_value_difference_cluster_test(
            aggregated_df, time_columns, 
            t_threshold=2.0, n_permutations=1000
        )
        
        # Step 4: Print results
        print_value_difference_results(results)
        
        # Step 5: Create visualization
        print(f"\n*** CREATING VISUALIZATION ***")
        create_value_difference_figure(results)
        
        # Step 6: Save results
        print("*** SAVING RESULTS ***")
        save_value_difference_results(results)
        
        # Step 7: Save pickle file
        pickle_filename = 'value_difference_magnitude_results.pkl'
        with open(pickle_filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n{'='*70}")
        print("VALUE DIFFERENCE MAGNITUDE ANALYSIS COMPLETE!")
        print(f"{'='*70}")
        
        print(f"\nGenerated files:")
        print(f"  ✓ value_difference_magnitude_analysis.png")
        print(f"  ✓ value_difference_magnitude_results.xlsx")
        print(f"  ✓ {pickle_filename}")
        
        # Summary
        sig_clusters = sum(1 for c in results['clusters'] if c.get('p_value_max', 1) < 0.05)
        cohens_d_values = [results['stats'][t]['cohens_d'] for t in results['time_columns'] if t in results['stats']]
        
        print(f"\nFinal Summary:")
        print(f"  • Large difference group (0.8, 0.2): {len(results['large_diff'])} participants")
        print(f"  • Small difference group (0.7, 0.3): {len(results['small_diff'])} participants")
        print(f"  • Significant clusters: {sig_clusters}")
        print(f"  • Mean effect size: d = {np.mean(cohens_d_values):.3f}")
        print(f"  • Max effect size: d = {np.max(np.abs(cohens_d_values)):.3f}")
        print(f"  • Analysis focus: Value distance magnitude sensitivity")
        
        return results
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# COMPARISON WITH OTHER ANALYSES
# =============================================================================

def compare_analysis_approaches():
    """
    Print comparison between different analysis approaches.
    """
    print("""
    =============================================================================
    COMPARISON OF PUPIL ANALYSIS APPROACHES
    =============================================================================
    
    1. ORIGINAL GROUPED ANALYSIS (0.7,0.8 vs 0.2,0.3):
       • Tests: High reward category vs Low reward category
       • Groups: Multiple values per condition
       • Focus: Overall reward level differences
       • Advantage: More trials per condition (higher power)
       • Question: "Do high vs low reward categories affect pupils?"
    
    2. EXTREME CONTRAST ANALYSIS (0.8 vs 0.2):
       • Tests: Highest reward vs Lowest reward
       • Groups: Single value per condition
       • Focus: Maximum reward difference
       • Advantage: Cleanest contrast, highest effect sizes
       • Question: "Do maximum reward differences affect pupils?"
    
    3. MODERATE CONTRAST ANALYSIS (0.7 vs 0.3):
       • Tests: Moderate high vs Moderate low
       • Groups: Single value per condition  
       • Focus: Intermediate reward difference
       • Advantage: Balanced contrast and power
       • Question: "Do moderate reward differences affect pupils?"
    
    4. VALUE DIFFERENCE MAGNITUDE ANALYSIS (THIS SCRIPT):
       • Tests: Large distance vs Small distance from reference
       • Groups: Extreme values (0.8,0.2) vs Moderate values (0.7,0.3)
       • Focus: Value extremity/distance rather than absolute reward
       • Advantage: Tests different psychological mechanism
       • Question: "Do pupils respond to value extremity/distance?"
    
    =============================================================================
    THEORETICAL IMPLICATIONS
    =============================================================================
    
    Different results across approaches suggest different mechanisms:
    
    • If GROUPED > EXTREME: Categorical reward processing
    • If EXTREME > GROUPED: Sensitive to maximum contrasts only
    • If VALUE DISTANCE significant: Extremity/distance coding
    • If none significant: Very small effects or no reward sensitivity
    
    The VALUE DIFFERENCE MAGNITUDE analysis tests whether pupils encode:
    - Distance from reference point (extremity)
    - Rather than absolute reward values
    - This could reflect risk sensitivity or salience coding
    
    """)

if __name__ == "__main__":
    # Print comparison information
    compare_analysis_approaches()
    
    print("Starting Value Difference Magnitude Analysis...")
    print("="*60)
    
    # Run the analysis
    results = main_value_difference_analysis()
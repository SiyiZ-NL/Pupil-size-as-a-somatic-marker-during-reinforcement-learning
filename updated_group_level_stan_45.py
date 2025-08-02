#!/usr/bin/env python3
# encoding: utf-8

"""
Updated Group-Level STAN Fitting Script for 45-Subject Separate Alphas Model
Based on the improved parameter recovery model, optimized for larger dataset
"""

import os
import sys
import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as op
import scipy.stats as stats
import seaborn as sns
import math
import cmdstanpy
from cmdstanpy import CmdStanModel

# Import the model creation function
from full_separate_alphas_model_45 import create_full_separate_alphas_model_no_reset

def prepare_data_for_stan_no_reset(df):
    """
    Transform the CSV data for Stan with NO Q-value reset between blocks.
    Q-values only reset at the start of each subject (not each block).
    Adapted for 45-subject dataset structure.
    """
    # First, make sure there are no missing values in critical columns
    critical_cols = ['subject_id', 'block_id', 'rp1', 'rp2', 'reward_p', 'outcome']
    for col in critical_cols:
        if df[col].isna().any():
            print(f"Warning: NaN values found in {col} column. Removing rows with NaN values.")
            df = df.dropna(subset=[col])
    
    transformed_data = []
    
    # Get unique subject IDs and sort them for consistent mapping
    subjects = sorted(df['subject_id'].unique())
    subject_mapping = {subj: i+1 for i, subj in enumerate(subjects)}
    
    print(f"Processing {len(subjects)} subjects: {subjects[:5]}{'...' if len(subjects) > 5 else ''}")
    
    # Create init marker for ONLY the first trial of each SUBJECT (not each block)
    df['init'] = 0
    for subject_id in subjects:
        subject_data = df[df['subject_id'] == subject_id]
        if len(subject_data) > 0:
            # Sort by block and trial to find the very first trial for this subject
            subject_data_sorted = subject_data.sort_values(['block_id', 'trial_id'])
            first_trial_idx = subject_data_sorted.index[0]
            df.loc[first_trial_idx, 'init'] = 1
    
    # Extract the necessary columns for each trial
    for _, row in df.iterrows():
        # Round reward probabilities to fix floating point precision
        prob1 = round(row['rp1'], 1)
        prob2 = round(row['rp2'], 1)
        
        # Determine the indices for these probabilities
        # 0 = 0.8, 1 = 0.2, 2 = 0.7, 3 = 0.3
        idx1 = 0 if prob1 == 0.8 else (1 if prob1 == 0.2 else (2 if prob1 == 0.7 else 3))
        idx2 = 0 if prob2 == 0.8 else (1 if prob2 == 0.2 else (2 if prob2 == 0.7 else 3))
        
        # Determine which probability was chosen
        chosen_prob = round(row['reward_p'], 1)
        chosen_idx = idx1 if chosen_prob == prob1 else idx2
        
        # Get reward (0=no reward, 1=reward)
        reward = 1 if row['outcome'] == 1 else 0
        
        # Get subject number (converted to integer index)
        subject = subject_mapping[row['subject_id']]
        
        # Get block/run number (1-4)
        block = int(row['block_id'])
        
        # Get initialization flag (1=first trial of subject ONLY, 0=otherwise)
        init = int(row['init'])
        
        transformed_data.append([idx1, idx2, chosen_idx, reward, subject, block, init])
    
    return np.array(transformed_data), subject_mapping

def run_separate_alphas_stan(data_df, iterations=2000, chains=4, model_name='separate_alphas_RL_45sub', output_dir='./results_separate_alphas_45'): 
    """Run Stan on reinforcement learning data using the separate alphas model with reduced shrinkage
    
    Arguments:
        data_df {pd.DataFrame} -- DataFrame containing the behavioral data
        iterations {int} -- Number of STAN iterations (increased for 45-subject model)
        chains {int} -- Number of STAN chains (increased for better convergence)
        model_name {str} -- Name for saving model files
        output_dir {str} -- Directory to save output files
        
    Returns:
        fit -- Stan fit object
        posterior -- Posterior samples
        subject_mapping -- Subject ID mapping
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare data for STAN model
    print("Preparing data for Stan...")
    processed_data, subject_mapping = prepare_data_for_stan_no_reset(data_df)
    
    # Extract data for STAN
    n_s = len(subject_mapping)  # Number of subjects
    n_t = len(processed_data)   # Total number of trials
    
    print(f"Stan data prepared: {n_s} subjects, {n_t} total trials")
    
    # Ensure data is in the correct format
    option1 = processed_data[:, 0].astype(int)  # First presented option
    option2 = processed_data[:, 1].astype(int)  # Second presented option
    chosen = processed_data[:, 2].astype(int)   # Chosen option
    reward = processed_data[:, 3].astype(int)   # Reward (0/1)
    subject = processed_data[:, 4].astype(int)  # Subject ID
    block = processed_data[:, 5].astype(int)    # Block/run
    init = processed_data[:, 6].astype(int)     # Init flag
    
    # Data to feed into the model
    stan_data = {
        'n_s': n_s, 
        'n_t': n_t, 
        'Option1': option1,
        'Option2': option2,
        'Chosen': chosen,
        'Reward': reward, 
        'Subject': subject,
        'Block': block,
        'Init': init
    }
    
    # Create Stan model file using the improved separate alphas model
    print("Creating separate alphas Stan model...")
    model_dir = os.path.join(output_dir, 'models')
    stan_model_path = create_full_separate_alphas_model_no_reset(model_dir)
    
    # Initialize or compile the model
    print("Compiling separate alphas Stan model...")
    
    # Check if CmdStan is installed
    try:
        cmdstan_path = cmdstanpy.cmdstan_path()
        print(f"CmdStan installation found at: {cmdstan_path}")
    except ValueError:
        print("CmdStan not found. Attempting to install CmdStan...")
        # Try to install CmdStan
        import subprocess
        try:
            subprocess.run([sys.executable, "-m", "cmdstanpy.install_cmdstan"], check=True)
            print("CmdStan installation completed.")
        except subprocess.CalledProcessError:
            print("Failed to install CmdStan automatically. Please install manually:")
            print("    python -m cmdstanpy.install_cmdstan")
            return None, None, subject_mapping
    
    # Compile the model
    print("Compiling model (this may take a few minutes)...")
    model = CmdStanModel(stan_file=stan_model_path)
    
    # Improved initialization for 45-subject separate alphas model
    print("Setting up improved initialization for 45-subject model...")
    init_values = []
    for chain in range(chains):
        init_dict = {
            'mu_b_pr': 0.0,
            'mu_ag_pr': 0.0,
            'mu_al_pr': 0.0,
            'sd_b': 0.5,
            'sd_ag': 0.4,
            'sd_al': 0.4,
            'b_ind_raw': [0.0] * n_s,
            'ag_ind_raw': [0.0] * n_s,
            'al_ind_raw': [0.0] * n_s
        }
        init_values.append(init_dict)
    
    # Fit the model with settings optimized for 45 subjects
    print(f"Running Stan with {iterations} iterations and {chains} chains...")
    print("Using optimized settings for 45-subject dataset...")
    print("This may take 30-60 minutes depending on your computer...")
    
    fit = model.sample(
        data=stan_data,
        iter_sampling=iterations,
        iter_warmup=iterations,  # Equal warmup for complex model
        chains=chains,
        adapt_delta=0.99,        # High adapt_delta for 45 subjects
        max_treedepth=20,        # Increased tree depth
        inits=init_values,
        show_progress=True,
        parallel_chains=min(chains, 4)  # Use parallel processing if available
    )
    
    # Save the fit results
    fit_path = os.path.join(output_dir, f'{model_name}_fit')
    fit.save_csvfiles(dir=fit_path)
    
    # Extract posterior samples
    print("Extracting posterior samples...")
    posterior = fit.draws_pd()
    
    # Save posterior samples with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    posterior_path = os.path.join(output_dir, f'separate_alphas_posterior_45sub_{timestamp}.pkl')
    with open(posterior_path, 'wb') as f:
        pickle.dump(posterior, f)
    
    # Reverse the subject mapping for easier interpretation
    reverse_subject_mapping = {v: k for k, v in subject_mapping.items()}
    
    # Save the subject mapping with timestamp
    mapping_file = os.path.join(output_dir, f'subject_mapping_45sub_{timestamp}.pkl')
    with open(mapping_file, 'wb') as f:
        pickle.dump({"subject_mapping": subject_mapping, 
                    "reverse_subject_mapping": reverse_subject_mapping}, f)
    
    # Check convergence
    print("Checking model convergence...")
    if 'divergent__' in posterior.columns:
        divergent_rate = posterior['divergent__'].mean() * 100
        print(f"Divergent transitions: {divergent_rate:.1f}%")
        if divergent_rate > 5:
            print("⚠️  Warning: High divergent transition rate. Consider increasing adapt_delta.")
        elif divergent_rate > 1:
            print("⚠️  Moderate divergent transitions. Results should be interpreted carefully.")
        else:
            print("✓ Good convergence!")
    
    # Check R-hat for key parameters
    try:
        # Get summary statistics if available
        rhat_cols = [col for col in posterior.columns if 'rhat' in col.lower()]
        if rhat_cols:
            max_rhat = posterior[rhat_cols].max().max()
            print(f"Maximum R-hat: {max_rhat:.3f}")
            if max_rhat > 1.1:
                print("⚠️  Warning: Some parameters have R-hat > 1.1. Consider more iterations.")
            else:
                print("✓ Good R-hat values!")
    except:
        print("R-hat information not available in posterior samples")
    
    # Plot group parameters
    print("Creating plots...")
    create_separate_alphas_plots(posterior, output_dir, model_name, iterations, n_s)
    
    print(f"✓ Model fitting complete! Results saved to {output_dir}")
    print(f"✓ Posterior samples: {posterior_path}")
    print(f"✓ Subject mapping: {mapping_file}")
    
    return fit, posterior, subject_mapping


def create_separate_alphas_plots(posterior, output_dir, model_name, iterations, n_subjects):
    """Create plots for the separate alphas model - optimized for 45 subjects"""
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Group-level parameters plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Separate Alphas Model: Group-Level Parameters (N={n_subjects})', fontsize=16)
    
    params = ['mu_b', 'mu_ag', 'mu_al']
    param_labels = ['Beta (Inverse Temperature)', 'Alpha Gain (Learning Rate)', 'Alpha Loss (Learning Rate)']
    colors = ['purple', 'orange', 'green']
    
    for i, (param, label, color) in enumerate(zip(params, param_labels, colors)):
        if i < 3:  # First three subplots
            row, col = (i // 2, i % 2) if i < 2 else (1, 0)
            ax = axes[row, col]
            
            values = posterior[param]
            
            # Create histogram with KDE
            sns.histplot(values, kde=True, stat="density", color=color, alpha=0.6, ax=ax)
            
            # Add statistics
            mean_val = values.mean()
            median_val = values.median()
            ci_lower, ci_upper = np.percentile(values, [2.5, 97.5])
            
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2)
            ax.axvline(median_val, color='blue', linestyle='--', linewidth=2)
            ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='gray')
            
            ax.set_title(f'Group {label}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.text(0.02, 0.98, f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\n95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            ax.grid(True, alpha=0.3)
    
    # Alpha comparison plot
    ax = axes[1, 1]
    
    # Extract alpha values
    alpha_gain_values = posterior['mu_ag']
    alpha_loss_values = posterior['mu_al']
    
    # Create side-by-side comparison
    data_for_plot = pd.DataFrame({
        'Alpha Gain': alpha_gain_values,
        'Alpha Loss': alpha_loss_values
    })
    
    # Melt for easier plotting
    melted_data = data_for_plot.melt(var_name='Learning Type', value_name='Learning Rate')
    
    # Create violin plot
    sns.violinplot(data=melted_data, x='Learning Type', y='Learning Rate', ax=ax)
    sns.stripplot(data=melted_data, x='Learning Type', y='Learning Rate', 
                 size=2, alpha=0.3, color='black', ax=ax)
    
    ax.set_title('Learning Rate Comparison: Gains vs Losses')
    ax.set_ylabel('Learning Rate')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    gain_mean = alpha_gain_values.mean()
    loss_mean = alpha_loss_values.mean()
    asymmetry = loss_mean - gain_mean
    
    ax.text(0.02, 0.98, f'Gain: {gain_mean:.3f}\nLoss: {loss_mean:.3f}\nAsymmetry: {asymmetry:+.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'separate_alphas_group_params_{model_name}_{iterations}IT_N{n_subjects}.pdf'))
    plt.close()
    
    print(f"✓ Group-level plots saved to {output_dir}")


def evaluate_separate_alphas_model(posterior=None, posterior_path=None, 
                                  model_name='separate_alphas_RL_45sub', 
                                  output_dir='./results_separate_alphas_45', 
                                  n_subjects=None,
                                  calculate_modes=True):
    """Evaluate the fitted separate alphas model and extract individual parameters
    
    Arguments:
        posterior {pd.DataFrame} -- Posterior samples
        posterior_path {str} -- Path to the posterior samples file
        model_name {str} -- Name of the model
        output_dir {str} -- Directory where results are saved
        n_subjects {int} -- Number of subjects
        calculate_modes {bool} -- Whether to calculate parameter modes
        
    Returns:
        par_modes -- DataFrame with parameter modes for each subject
        par_scales -- DataFrame with parameter scales for each subject
    """
    # Load posterior if not provided
    if posterior is None:
        if posterior_path is None:
            # Find the most recent posterior file
            posterior_files = [f for f in os.listdir(output_dir) 
                             if f.startswith('separate_alphas_posterior_45sub_') and f.endswith('.pkl')]
            
            if not posterior_files:
                print("No separate alphas posterior files found.")
                return None, None
            
            # Sort by modification time to get the latest
            posterior_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
            posterior_path = os.path.join(output_dir, posterior_files[0])
        
        # Check if the file exists
        if not os.path.exists(posterior_path):
            print(f"Error: Posterior file not found at {posterior_path}")
            return None, None
            
        with open(posterior_path, 'rb') as f:
            posterior = pickle.load(f)
    
    # Load subject mapping (find the most recent one)
    mapping_files = [f for f in os.listdir(output_dir) 
                    if f.startswith('subject_mapping_45sub_') and f.endswith('.pkl')]
    
    if not mapping_files:
        print("Error: No subject mapping files found")
        return None, None
    
    mapping_files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
    mapping_file = os.path.join(output_dir, mapping_files[0])
    
    with open(mapping_file, 'rb') as f:
        mapping_data = pickle.load(f)
        reverse_subject_mapping = mapping_data["reverse_subject_mapping"]
    
    # Determine number of subjects
    if n_subjects is None:
        n_subjects = len(reverse_subject_mapping)
    
    # Get subject names (now should be integers for 45-subject dataset)
    names = [reverse_subject_mapping[i+1] for i in range(n_subjects)]
    
    print(f"Evaluating separate alphas model for {n_subjects} subjects")
    
    # Print group-level parameter estimates
    print(f"\nGroup-level parameter estimates:")
    print(f"Beta (inverse temperature): {posterior['mu_b'].mean():.3f} ± {posterior['mu_b'].std():.3f}")
    print(f"Alpha Gain (learning rate for gains): {posterior['mu_ag'].mean():.3f} ± {posterior['mu_ag'].std():.3f}")
    print(f"Alpha Loss (learning rate for losses): {posterior['mu_al'].mean():.3f} ± {posterior['mu_al'].std():.3f}")
    
    # Calculate asymmetry
    alpha_diff = posterior['mu_al'] - posterior['mu_ag']
    print(f"Learning asymmetry (loss - gain): {alpha_diff.mean():.3f} ± {alpha_diff.std():.3f}")
    
    asymmetry_positive = (alpha_diff > 0).mean() * 100
    print(f"Probability of loss > gain learning: {asymmetry_positive:.1f}%")
    
    if alpha_diff.mean() > 0.02:
        print("→ Strong evidence for higher learning from losses")
    elif alpha_diff.mean() > 0.005:
        print("→ Moderate evidence for higher learning from losses")
    elif alpha_diff.mean() < -0.02:
        print("→ Strong evidence for higher learning from gains")
    elif alpha_diff.mean() < -0.005:
        print("→ Moderate evidence for higher learning from gains")
    else:
        print("→ Symmetric learning from gains and losses")
    
    # Calculate parameter modes for each subject
    if calculate_modes:
        print("Calculating individual parameter modes...")
        modes = np.zeros((n_subjects, 3))
        scales = np.zeros((n_subjects, 3))
        
        # Extract individual parameters
        for s in range(n_subjects):
            if (s + 1) % 10 == 0:  # Progress indicator for 45 subjects
                print(f"  Processing subject {s+1}/{n_subjects}")
            
            # Extract parameter distributions for this subject
            b_dist = posterior[f'b_ind[{s+1}]']
            ag_dist = posterior[f'ag_ind[{s+1}]']
            al_dist = posterior[f'al_ind[{s+1}]']
            
            # Calculate location and scale
            loc_b, scale_b = stats.norm.fit(b_dist)
            loc_ag, scale_ag = stats.norm.fit(ag_dist)
            loc_al, scale_al = stats.norm.fit(al_dist)
            
            # Store in arrays
            modes[s, 0] = loc_b
            modes[s, 1] = loc_ag
            modes[s, 2] = loc_al
            
            scales[s, 0] = scale_b
            scales[s, 1] = scale_ag
            scales[s, 2] = scale_al
        
        # Create DataFrames
        par_modes = pd.DataFrame(modes, columns=['beta', 'alpha_gain', 'alpha_loss'], index=names)
        par_scales = pd.DataFrame(scales, columns=['beta', 'alpha_gain', 'alpha_loss'], index=names)
        
        # Add asymmetry column
        par_modes['learning_asymmetry'] = par_modes['alpha_loss'] - par_modes['alpha_gain']
        
        # Save parameter modes and scales
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        par_mode_pkl = os.path.join(output_dir, f'separate_alphas_par_modes_45sub_{timestamp}.pkl')
        par_scale_pkl = os.path.join(output_dir, f'separate_alphas_par_scales_45sub_{timestamp}.pkl')
        
        with open(par_mode_pkl, 'wb') as f:
            pickle.dump(par_modes, f)
        
        with open(par_scale_pkl, 'wb') as f:
            pickle.dump(par_scales, f)
        
        print(f"\nParameter modes and scales saved:")
        print(f"  Modes: {par_mode_pkl}")
        print(f"  Scales: {par_scale_pkl}")
        
        # Print summary of individual differences
        print(f"\nIndividual parameter ranges:")
        beta_min = par_modes['beta'].min()
        beta_max = par_modes['beta'].max()
        ag_min = par_modes['alpha_gain'].min()
        ag_max = par_modes['alpha_gain'].max()
        al_min = par_modes['alpha_loss'].min()
        al_max = par_modes['alpha_loss'].max()
        
        print(f"Beta: {beta_min:.3f} - {beta_max:.3f}")
        print(f"Alpha Gain: {ag_min:.3f} - {ag_max:.3f}")
        print(f"Alpha Loss: {al_min:.3f} - {al_max:.3f}")
        
        # Calculate individual asymmetries
        individual_asymmetry = par_modes['learning_asymmetry']
        asym_mean = individual_asymmetry.mean()
        asym_min = individual_asymmetry.min()
        asym_max = individual_asymmetry.max()
        asym_positive = (individual_asymmetry > 0).sum()
        
        print(f"\nIndividual learning asymmetries:")
        print(f"Mean: {asym_mean:.3f}")
        print(f"Range: {asym_min:.3f} to {asym_max:.3f}")
        print(f"Subjects with loss > gain learning: {asym_positive}/{n_subjects}")
        
        return par_modes, par_scales
    
    return None, None


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Run separate alphas hierarchical reinforcement learning model for 45 subjects')
    parser.add_argument('--data', type=str, required=True, help='Path to behavioral data CSV')
    parser.add_argument('--output', type=str, default='./results_separate_alphas_45', help='Directory to save output files')
    parser.add_argument('--iterations', type=int, default=2000, help='Number of STAN iterations')
    parser.add_argument('--chains', type=int, default=4, help='Number of STAN chains')
    parser.add_argument('--model_name', type=str, default='separate_alphas_RL_45sub', help='Model name')
    
    args = parser.parse_args()
    
    # Load data
    data_df = pd.read_csv(args.data)
    
    # Convert outcome column to ensure it's 0 or 1
    data_df['outcome'] = data_df['outcome'].apply(lambda x: 1 if x == 1 else 0)
    
    print(f"Loaded {len(data_df)} trials from {len(data_df['subject_id'].unique())} subjects")
    
    # Run separate alphas STAN model
    fit, posterior, subject_mapping = run_separate_alphas_stan(
        data_df,
        iterations=args.iterations,
        chains=args.chains,
        model_name=args.model_name,
        output_dir=args.output
    )
    
    # Evaluate model only if fit was successful
    if fit is not None and posterior is not None:
        par_modes, par_scales = evaluate_separate_alphas_model(
            posterior=posterior,
            model_name=args.model_name,
            output_dir=args.output,
            n_subjects=len(subject_mapping),
            calculate_modes=True
        )
        
        print("\n" + "="*70)
        print("SEPARATE ALPHAS MODEL ANALYSIS COMPLETE!")
        print("="*70)
    else:
        print("Analysis failed. Please check the errors above.")

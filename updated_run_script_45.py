#!/usr/bin/env python3
# encoding: utf-8

"""
Updated Run Script for 45-Subject Dataset
Simple script to run the separate alphas Stan model with reduced shrinkage
"""

import os
import sys
import pandas as pd

# Set data path for 45-subject dataset
DATA_PATH = "/Users/zengsiyi/Desktop/sommark_data_Siyi/cleaned_combined_behavioral_data_FILTERED_45.csv"
# Set output directory - using new name to distinguish from 8-subject results
OUTPUT_DIR = "/Users/zengsiyi/Desktop/sommark_data_Siyi/unified_results_45sub"

def main():
    print("UPDATED SEPARATE ALPHAS BAYESIAN Q-LEARNING MODEL - 45 SUBJECTS")
    print("="*70)
    print("Running improved model with:")
    print("✓ Separate alpha_gain and alpha_loss parameters")
    print("✓ Reduced hierarchical shrinkage")
    print("✓ Non-centered parameterization")
    print("✓ Better parameter recovery")
    print("✓ Scaled for 45 subjects")
    print()
    
    print("Installing CmdStan if needed...")
    # Try to install CmdStan
    import subprocess
    try:
        subprocess.run([sys.executable, "-m", "cmdstanpy.install_cmdstan"], check=True)
        print("✓ CmdStan installation completed or already installed.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install CmdStan: {e}")
        print("Please install manually with:")
        print("    python -m cmdstanpy.install_cmdstan")
        return
    
    # Now import from the updated script
    from updated_group_level_stan_45 import run_separate_alphas_stan, evaluate_separate_alphas_model
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"Loading data from {DATA_PATH}...")
    # Load data
    data_df = pd.read_csv(DATA_PATH)
    
    # Check the 'outcome' column for values that aren't 0 or 1
    unique_outcomes = data_df['outcome'].unique()
    print(f"Unique values in outcome column: {unique_outcomes}")
    
    # Convert outcome column to ensure it's 0 or 1
    data_df['outcome'] = data_df['outcome'].apply(lambda x: 1 if x == 1 else 0)
    
    print(f"Loaded {len(data_df)} trials from {len(data_df['subject_id'].unique())} subjects")
    
    # Data validation
    print("\nValidating data structure...")
    critical_cols = ['subject_id', 'block_id', 'rp1', 'rp2', 'reward_p', 'outcome']
    missing_counts = data_df[critical_cols].isna().sum()
    
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  Warning: {count} missing values in {col}")
    
    # Remove rows with missing critical values
    initial_rows = len(data_df)
    data_df = data_df.dropna(subset=critical_cols)
    final_rows = len(data_df)
    
    if final_rows < initial_rows:
        print(f"  Removed {initial_rows - final_rows} rows with missing values")
    
    print(f"  Final dataset: {final_rows} trials from {len(data_df['subject_id'].unique())} subjects")
    
    # Run separate alphas STAN model with settings optimized for 45 subjects
    print("\nRunning updated separate alphas Stan model...")
    print("Using optimized settings for 45-subject dataset...")
    
    fit, posterior, subject_mapping = run_separate_alphas_stan(
        data_df,
        iterations=2000,      # Increased for larger dataset
        chains=4,             # More chains for better convergence with more subjects
        model_name='separate_alphas_RL_45sub',
        output_dir=OUTPUT_DIR
    )
    
    if fit is not None and posterior is not None:
        print("\n" + "="*70)
        print("MODEL FITTING COMPLETE!")
        print("="*70)
        
        # Evaluate the model
        print("\nEvaluating model and extracting individual parameters...")
        par_modes, par_scales = evaluate_separate_alphas_model(
            posterior=posterior,
            model_name='separate_alphas_RL_45sub',
            output_dir=OUTPUT_DIR,
            n_subjects=len(subject_mapping),
            calculate_modes=True
        )
        
        if par_modes is not None:
            print(f"\n✓ Individual parameter estimates successfully extracted for {len(subject_mapping)} subjects!")
            print(f"✓ Results saved to: {OUTPUT_DIR}")
            
            print(f"\nNext steps:")
            print(f"1. Run model validation: python updated_model_validation_45.py")
            print(f"2. Generate trial-wise Q-values: python updated_trial_wise_qvalues_45.py")
            print(f"3. Run parameter recovery test: python comprehensive_parameter_recovery_45.py")
            print(f"4. Run learning curve analysis: python comprehensive_learning_curves_45.py")
            print(f"5. Calculate model fit metrics: python model_fit_assessment_45.py")
            
        else:
            print("Warning: Failed to extract individual parameters")
    
    else:
        print("\n❌ MODEL FITTING FAILED!")
        print("Please check the errors above and try again.")
        print("\nTroubleshooting tips:")
        print("- Check data format and missing values")
        print("- Consider reducing iterations for initial testing")
        print("- Monitor memory usage with 45 subjects")

if __name__ == "__main__":
    main()

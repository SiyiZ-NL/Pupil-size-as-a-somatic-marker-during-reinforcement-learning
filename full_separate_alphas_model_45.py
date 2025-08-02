#!/usr/bin/env python3
# encoding: utf-8

import os

def create_full_separate_alphas_model_no_reset(output_dir='./models'):
    """
    Create the full model with separate alpha_gain and alpha_loss 
    using NO Q-value reset between blocks - optimized for 45 subjects
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stan_model_code = """
    data{
      int<lower=1> n_s;                       // number of subjects
      int<lower=1> n_t;                       // total number of trials
      array[n_t] int<lower=0,upper=3> Option1;      // first presented option (0=0.8, 1=0.2, 2=0.7, 3=0.3)
      array[n_t] int<lower=0,upper=3> Option2;      // second presented option
      array[n_t] int<lower=0,upper=3> Chosen;       // chosen option
      array[n_t] int<lower=0,upper=1> Reward;       // reward (1=yes, 0=no)
      array[n_t] int<lower=1,upper=n_s> Subject;    // subject ID
      array[n_t] int<lower=0,upper=1> Init;         // initialization flag for new SUBJECT only (not blocks)
      array[n_t] int<lower=1,upper=4> Block;        // block/run number (1-4)
    }
      
    parameters{
      // group level mean parameters
      real mu_b_pr;                // inverse gain parameter (probit scale)
      real mu_ag_pr;               // alpha_gain (probit scale)
      real mu_al_pr;               // alpha_loss (probit scale)
     
      // group level standard deviation - OPTIMIZED FOR 45 SUBJECTS
      real<lower=0> sd_b;          // SD for inverse gain
      real<lower=0> sd_ag;         // SD for alpha_gain - allow individual variation
      real<lower=0> sd_al;         // SD for alpha_loss - allow individual variation
      
      // individual level parameters (non-centered parameterization)
      array[n_s] real b_ind_raw;          // raw individual inverse gain
      array[n_s] real ag_ind_raw;         // raw individual alpha_gain
      array[n_s] real al_ind_raw;         // raw individual alpha_loss
    }
    
    transformed parameters{
      // group level mean parameters after transformation
      real<lower=0,upper=10> mu_b;        // inverse gain parameter (constrained range)
      real<lower=0,upper=1> mu_ag;        // alpha_gain
      real<lower=0,upper=1> mu_al;        // alpha_loss
     
      // individual level parameters after transformation
      array[n_s] real<lower=0,upper=10> b_ind;     // inverse gain parameter
      array[n_s] real<lower=0,upper=1> ag_ind;     // alpha_gain
      array[n_s] real<lower=0,upper=1> al_ind;     // alpha_loss
     
      // Non-centered parameterization
      array[n_s] real b_ind_pr;
      array[n_s] real ag_ind_pr;
      array[n_s] real al_ind_pr;
      
      // Apply non-centered transformation
      for (s in 1:n_s) {
        b_ind_pr[s] = mu_b_pr + sd_b * b_ind_raw[s];
        ag_ind_pr[s] = mu_ag_pr + sd_ag * ag_ind_raw[s];
        al_ind_pr[s] = mu_al_pr + sd_al * al_ind_raw[s];
      }
      
      // Apply probit transformations (with constrained beta range for stability)
      mu_b = Phi(mu_b_pr) * 10;      // Range 0-10 for numerical stability
      mu_ag = Phi(mu_ag_pr);
      mu_al = Phi(mu_al_pr);
      
      for (s in 1:n_s) {
        b_ind[s] = Phi(b_ind_pr[s]) * 10;    // Range 0-10 for numerical stability
        ag_ind[s] = Phi(ag_ind_pr[s]);
        al_ind[s] = Phi(al_ind_pr[s]);
      }
    }
    
    model{
      // define general variables
      int si;                  // subject index
      array[4] real prQ;       // Q-values for the 4 probability options (0.8, 0.2, 0.7, 0.3)
      array[2] real Qchoice;   // Q-values for the presented options
      real epsilon;            // small constant for choice probabilities
      real alpha;              // learning rate (depends on reward)
      vector[2] pchoice;       // choice probability vector
      
      // PRIORS OPTIMIZED FOR 45 SUBJECTS
      mu_b_pr ~ normal(0, 0.8);        // Group mean priors
      mu_ag_pr ~ normal(0, 0.8);       
      mu_al_pr ~ normal(0, 0.8);       
     
      // ALLOW INDIVIDUAL VARIATION WHILE MAINTAINING STABILITY
      sd_b ~ normal(0, 0.7);           // Allow beta variation across 45 subjects
      sd_ag ~ normal(0, 0.5);          // Allow alpha_gain variation
      sd_al ~ normal(0, 0.5);          // Allow alpha_loss variation
      
      // Standard normal priors for raw parameters (non-centered)
      for (s in 1:n_s) {
        b_ind_raw[s] ~ normal(0, 1);
        ag_ind_raw[s] ~ normal(0, 1);
        al_ind_raw[s] ~ normal(0, 1);
      }
      
      // small constant to avoid zero probabilities
      epsilon = 0.00001;
      
      // loop through trials
      for (t in 1:n_t) {
        // initialize Q-values ONLY at start of NEW SUBJECT (not blocks)
        if (Init[t]==1){
          si = Subject[t];
          
          // Initialize Q-values for all options to 0.5
          for (v in 0:3) {
            prQ[v+1] = 0.5;  // Note: Stan arrays are 1-indexed, so using v+1
          }
        }
   
        // Get Q-values for the two presented options
        Qchoice[1] = prQ[Option1[t]+1];  // First presented option (adding 1 for 1-indexed arrays)
        Qchoice[2] = prQ[Option2[t]+1];  // Second presented option (adding 1 for 1-indexed arrays)
        
        // Calculate choice probabilities using softmax
        pchoice[1] = 1/(1+exp(b_ind[si]*(Qchoice[2]-Qchoice[1])));
        pchoice[2] = 1-pchoice[1];
        
        // Add small epsilon to avoid zero probabilities
        pchoice[1] = epsilon/2+(1-epsilon)*pchoice[1];
        pchoice[2] = epsilon/2+(1-epsilon)*pchoice[2];
        
        // Model the probability of choosing the first option
        if (Chosen[t] == Option1[t]) {
          target += bernoulli_lpmf(1 | pchoice[1]);
        } else {
          target += bernoulli_lpmf(0 | pchoice[1]);
        }
        
        // Determine learning rate based on reward outcome (SEPARATE ALPHAS)
        alpha = Reward[t] * ag_ind[si] + (1-Reward[t]) * al_ind[si];
        
        // Update the Q-value of the chosen option (CUMULATIVE ACROSS BLOCKS)
        prQ[Chosen[t]+1] = prQ[Chosen[t]+1] + alpha*(Reward[t]-prQ[Chosen[t]+1]);
      }
    }
    
    generated quantities{
      // Calculate log likelihood for model comparison
      array[n_t] real log_lik;
      
      {
        // define general variables
        int si;
        array[4] real prQ;
        array[2] real Qchoice;
        real epsilon;
        real alpha;
        vector[2] pchoice;
        
        // small constant to avoid zero probabilities
        epsilon = 0.00001;
        
        // loop through trials
        for (t in 1:n_t) {
          // initialize Q-values ONLY at start of NEW SUBJECT (not blocks)
          if (Init[t]==1){
            si = Subject[t];
            
            // Initialize Q-values for all options to 0.5
            for (v in 0:3) {
              prQ[v+1] = 0.5;  // Note: Stan arrays are 1-indexed
            }
          }
     
          // Get Q-values for the two presented options
          Qchoice[1] = prQ[Option1[t]+1];
          Qchoice[2] = prQ[Option2[t]+1];
          
          // Calculate choice probabilities using softmax
          pchoice[1] = 1/(1+exp(b_ind[si]*(Qchoice[2]-Qchoice[1])));
          pchoice[2] = 1-pchoice[1];
          
          // Add small epsilon to avoid zero probabilities
          pchoice[1] = epsilon/2+(1-epsilon)*pchoice[1];
          pchoice[2] = epsilon/2+(1-epsilon)*pchoice[2];
          
          // Calculate log likelihood
          if (Chosen[t] == Option1[t]) {
            log_lik[t] = bernoulli_lpmf(1 | pchoice[1]);
          } else {
            log_lik[t] = bernoulli_lpmf(0 | pchoice[1]);
          }
          
          // Determine learning rate based on reward outcome
          alpha = Reward[t] * ag_ind[si] + (1-Reward[t]) * al_ind[si];
          
          // Update the Q-value of the chosen option (CUMULATIVE ACROSS BLOCKS)
          prQ[Chosen[t]+1] = prQ[Chosen[t]+1] + alpha*(Reward[t]-prQ[Chosen[t]+1]);
        }
      }
    }
    """
    
    stan_model_path = os.path.join(output_dir, 'stan_RL_separate_alphas_no_reset_45sub.stan')
    
    with open(stan_model_path, 'w') as f:
        f.write(stan_model_code)
    
    print(f"✓ Created 45-subject separate alphas NO-RESET model at: {stan_model_path}")
    return stan_model_path

def main():
    """Create the full model with separate alphas for 45 subjects"""
    print("Creating full model with separate alpha_gain and alpha_loss for 45 subjects...")
    print("Using reduced shrinkage approach optimized for larger dataset")
    
    # Create model directory
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Create the full model
    model_path = create_full_separate_alphas_model_no_reset(model_dir)
    
    print("\n" + "="*70)
    print("FULL SEPARATE ALPHAS MODEL CREATED FOR 45 SUBJECTS!")
    print("="*70)
    print("\nKey features:")
    print("✓ Separate alpha_gain and alpha_loss parameters")
    print("✓ Reduced hierarchical shrinkage")
    print("✓ Non-centered parameterization")
    print("✓ Optimized priors for 45-subject dataset")
    print("✓ Beta range 0-10 for numerical stability")
    print("✓ Enhanced individual variation modeling")
    
    print(f"\nModel file: {model_path}")
    
    print("\nNext steps:")
    print("1. Run the model: python updated_run_script_45.py")
    print("2. Validate results: python updated_model_validation_45.py")
    print("3. Run parameter recovery testing")
    print("4. Analyze learning curves and model fit")
    
    return model_path

if __name__ == "__main__":
    main()

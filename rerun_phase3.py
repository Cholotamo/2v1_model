# This script is to rerun the phase 3 experiment without the need to train the models again.
# It will use the existing checkpoints and run the final model evaluation.

import os
import sys
import logging
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Before importing from phase3_experiment.py, we need to temporarily modify the results_dir
# to point to the existing directory instead of creating a new one.
# This is a bit of a hack, but it allows us to reuse all the existing functions.

# Store the original directory path
EXISTING_RESULTS_DIR = '/Users/julian/Desktop/Developer/2v1_model/phase3_experiment_20250316_113604'

# Define a function to patch the module's results_dir before importing
def patch_module():
    # First, add the parent directory to the Python path
    sys.path.append('/Users/julian/Desktop/Developer/2v1_model')
    
    # Import the module as a whole
    import phase3_experiment
    
    # Patch the results_dir to use our existing directory
    phase3_experiment.results_dir = EXISTING_RESULTS_DIR
    
    # Configure logging to append to the existing log file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{EXISTING_RESULTS_DIR}/experiment.log', mode='a'),
            logging.StreamHandler()
        ]
    )
    
    # Return the patched module
    return phase3_experiment

# Patch and import the module
phase3_experiment = patch_module()

# Now we can import specific functions from the patched module
from phase3_experiment import (
    logger,
    load_checkpoint,
    save_checkpoint,
    final_model_evaluation  # This will use the fixed version in phase3_experiment.py
)

def main():
    try:
        logger.info("============= STARTING RERUN OF PHASE3 EXPERIMENT =============")
        logger.info(f"Using existing results directory: {EXISTING_RESULTS_DIR}")
        
        # Step 1: Load dataset
        logger.info("Step 1: Loading dataset checkpoint")
        dataset_checkpoint = load_checkpoint('dataset')
        if dataset_checkpoint is None:
            raise FileNotFoundError("Dataset checkpoint not found")
        X, y = dataset_checkpoint['X'], dataset_checkpoint['y']
        logger.info(f"Dataset loaded: X shape={X.shape}, y shape={y.shape}")
        
        # Step 2: Verify feature selection checkpoint
        logger.info("Step 2: Verifying feature selection checkpoint")
        feature_selection_results = load_checkpoint('feature_selection')
        if feature_selection_results is None:
            raise FileNotFoundError("Feature selection checkpoint not found")
        logger.info("Feature selection checkpoint verified")
        
        # Step 3: Load model evaluation results
        logger.info("Step 3: Loading model evaluation results")
        results_df = load_checkpoint('model_evaluation')
        if results_df is None:
            raise FileNotFoundError("Model evaluation checkpoint not found")
        logger.info(f"Model evaluation results loaded: {len(results_df)} models")
        
        # Step 4: Check ensemble results
        logger.info("Step 4: Checking ensemble results")
        ensemble_result = load_checkpoint('ensemble')
        if ensemble_result:
            logger.info("Ensemble results found and loaded")
        else:
            logger.info("No ensemble results found")
        
        # Force regeneration of final evaluation by removing existing checkpoint
        final_eval_path = f'{EXISTING_RESULTS_DIR}/checkpoints/final_evaluation.pkl'
        if os.path.exists(final_eval_path):
            logger.info("Removing existing final_evaluation checkpoint to force regeneration")
            os.remove(final_eval_path)
        
        # Step 5: Run final evaluation using the function from the original script
        logger.info("Step 5: Running final model evaluation")
        best_model = final_model_evaluation(results_df, X, y)
        
        # Check if report was generated
        report_path = f'{EXISTING_RESULTS_DIR}/final_report.md'
        if os.path.exists(report_path):
            logger.info(f"Final report successfully generated at: {report_path}")
            
            # Report best model info
            if best_model is not None:
                model_name = best_model['Model']
                if isinstance(model_name, pd.Series):
                    model_name = model_name.iloc[0]
                    
                feature_method = best_model['Feature Selection']
                if isinstance(feature_method, pd.Series):
                    feature_method = feature_method.iloc[0]
                    
                test_acc = best_model['Test Accuracy']
                if isinstance(test_acc, pd.Series):
                    test_acc = test_acc.iloc[0]
                    
                logger.info(f"Best model: {model_name} with {feature_method}")
                logger.info(f"Best model test accuracy: {float(test_acc):.4f}")
        else:
            logger.error(f"Final report was not generated at: {report_path}")
        
        logger.info("============= RERUN COMPLETED SUCCESSFULLY =============")
        return best_model
        
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()
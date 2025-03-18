import os
import pandas as pd
import numpy as np
import pickle
import sys

# Add the package to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from march_madness.config import (STARTING_SEASON, CURRENT_SEASON, 
                                 TRAINING_SEASONS, VALIDATION_SEASON)
from march_madness.utils.data_processors import load_or_prepare_modeling_data
from march_madness.data.loaders import load_mens_data, load_womens_data
from march_madness.utils.helpers import train_and_predict_model
from march_madness.models.training import time_based_cross_validation
from march_madness.utils.data_processors import prepare_modeling_data

def main():
    print("Training NCAA Basketball Tournament Prediction Models")
    
    # Define directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    cache_dir = os.path.join(project_root, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train March Madness prediction models')
    parser.add_argument('--skip-data-load', action='store_true', 
                        help='Skip loading raw data (assume modeling data is cached)')
    parser.add_argument('--force-prepare', action='store_true',
                        help='Force data preparation even if cache exists')
    args = parser.parse_args()
    
    # Load or prepare modeling data
    if args.skip_data_load:
        print("Skipping raw data loading, using cached modeling data...")
        mens_data = womens_data = None
    else:
        # Load raw data (only needed if cache doesn't exist or force_prepare=True)
        print("Loading raw data...")
        mens_data = load_mens_data(STARTING_SEASON, data_dir=data_dir)
        womens_data = load_womens_data(STARTING_SEASON, data_dir=data_dir)
    
    # Get modeling data (from cache if available, or process it)
    print("Getting modeling data for men's tournament...")
    mens_modeling_data = load_or_prepare_modeling_data(
        mens_data, "men's", STARTING_SEASON, CURRENT_SEASON, 
        TRAINING_SEASONS + [VALIDATION_SEASON],
        cache_dir=cache_dir,
        force_prepare=args.force_prepare
    )
    
    print("Getting modeling data for women's tournament...")
    womens_modeling_data = load_or_prepare_modeling_data(
        womens_data, "women's", STARTING_SEASON, CURRENT_SEASON, 
        TRAINING_SEASONS + [VALIDATION_SEASON],
        cache_dir=cache_dir,
        force_prepare=args.force_prepare
    )

    print("\n=== Performing cross-validation for men's tournament models ===")
    mens_cv_results = time_based_cross_validation(
        mens_modeling_data, 
        "men's",
        prepare_modeling_data,  # Feature engineering function 
        train_and_predict_model # Model training function
    )
    
    print("\n=== Performing cross-validation for women's tournament models ===")
    womens_cv_results = time_based_cross_validation(
        womens_modeling_data,
        "women's",
        prepare_modeling_data,
        train_and_predict_model
    )
    
    # Train models
    print("Training Men's model...")
    mens_model, mens_feature_cols, mens_scaler, mens_dropped_features = train_and_predict_model(
        mens_modeling_data, "men's", TRAINING_SEASONS, VALIDATION_SEASON, []
    )
    
    print("Training Women's model...")
    womens_model, womens_feature_cols, womens_scaler, womens_dropped_features = train_and_predict_model(
        womens_modeling_data, "women's", TRAINING_SEASONS, VALIDATION_SEASON, []
    )

    # Save models and related objects
    os.makedirs('models', exist_ok=True)
    
    with open('models/mens_model.pkl', 'wb') as f:
        pickle.dump(mens_model, f)
    
    with open('models/mens_metadata.pkl', 'wb') as f:
        pickle.dump({
            'feature_cols': mens_feature_cols,
            'scaler': mens_scaler,
            'dropped_features': mens_dropped_features
        }, f)
    
    with open('models/womens_model.pkl', 'wb') as f:
        pickle.dump(womens_model, f)
    
    with open('models/womens_metadata.pkl', 'wb') as f:
        pickle.dump({
            'feature_cols': womens_feature_cols,
            'scaler': womens_scaler,
            'dropped_features': womens_dropped_features
        }, f)
    
    print("Models trained and saved successfully!")

if __name__ == "__main__":
    main()
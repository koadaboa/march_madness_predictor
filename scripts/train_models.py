#!/usr/bin/env python
# Script to train models
import os
import pandas as pd
import numpy as np
import pickle
import sys

# Add the package to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from march_madness.config import (STARTING_SEASON, CURRENT_SEASON, 
                                 TRAINING_SEASONS, VALIDATION_SEASON)
from march_madness.data.loaders import (load_mens_data, load_womens_data, 
                                       filter_data_dict_by_seasons)
from march_madness.utils.helpers import prepare_modeling_data, train_and_predict_model

def main():
    print("Training NCAA Basketball Tournament Prediction Models")
    # Define the data directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    
    # Load training data
    mens_train_data = filter_data_dict_by_seasons(
        load_mens_data(STARTING_SEASON, data_dir = data_dir), 
        TRAINING_SEASONS + [VALIDATION_SEASON]
    )
    
    womens_train_data = filter_data_dict_by_seasons(
        load_womens_data(STARTING_SEASON, data_dir = data_dir), 
        TRAINING_SEASONS + [VALIDATION_SEASON]
    )
    
    # Prepare modeling data
    print("Preparing Men's modeling data...")
    mens_modeling_data = prepare_modeling_data(
        mens_train_data, "men's", STARTING_SEASON, CURRENT_SEASON, 
        TRAINING_SEASONS + [VALIDATION_SEASON]
    )
    
    print("Preparing Women's modeling data...")
    womens_modeling_data = prepare_modeling_data(
        womens_train_data, "women's", STARTING_SEASON, CURRENT_SEASON, 
        TRAINING_SEASONS + [VALIDATION_SEASON]
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
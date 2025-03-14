#!/usr/bin/env python
# Script to generate predictions
import os
import pandas as pd
import numpy as np
import pickle
import sys

# Add the package to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from march_madness.config import (STARTING_SEASON, CURRENT_SEASON, 
                                 TRAINING_SEASONS, VALIDATION_SEASON, 
                                 PREDICTION_SEASONS)
from march_madness.data.loaders import (load_mens_data, load_womens_data, 
                                       filter_data_dict_by_seasons)
from march_madness.utils.helpers import prepare_modeling_data, train_and_predict_model
from march_madness.models.prediction import combine_predictions

def main():
    print("Generating NCAA Basketball Tournament Predictions")
    
    # Load prediction data
    mens_predict_data = filter_data_dict_by_seasons(
        load_mens_data(STARTING_SEASON), 
        PREDICTION_SEASONS
    )
    
    womens_predict_data = filter_data_dict_by_seasons(
        load_womens_data(STARTING_SEASON), 
        PREDICTION_SEASONS
    )
    
    # Ensure prediction data does not contain tournament results
    if 'df_tourney' in mens_predict_data:
        mens_predict_data['df_tourney'] = pd.DataFrame(columns=mens_predict_data['df_tourney'].columns)
    
    if 'df_tourney' in womens_predict_data:
        womens_predict_data['df_tourney'] = pd.DataFrame(columns=womens_predict_data['df_tourney'].columns)
    
    # Prepare prediction data
    print("Preparing Men's prediction data...")
    mens_prediction_data = prepare_modeling_data(
        mens_predict_data, "men's", STARTING_SEASON, CURRENT_SEASON, 
        PREDICTION_SEASONS
    )
    
    print("Preparing Women's prediction data...")
    womens_prediction_data = prepare_modeling_data(
        womens_predict_data, "women's", STARTING_SEASON, CURRENT_SEASON, 
        PREDICTION_SEASONS
    )
    
    # Load trained models and metadata
    with open('models/mens_model.pkl', 'rb') as f:
        mens_model = pickle.load(f)
    
    with open('models/mens_metadata.pkl', 'rb') as f:
        mens_metadata = pickle.load(f)
    
    with open('models/womens_model.pkl', 'rb') as f:
        womens_model = pickle.load(f)
    
    with open('models/womens_metadata.pkl', 'rb') as f:
        womens_metadata = pickle.load(f)
    
    # Generate predictions
    print("Generating Men's predictions...")
    mens_predictions = train_and_predict_model(
        mens_prediction_data, "men's", [], None, PREDICTION_SEASONS,
        model=mens_model,
        feature_cols=mens_metadata['feature_cols'],
        scaler=mens_metadata['scaler'],
        dropped_features=mens_metadata['dropped_features']
    )
    
    print("Generating Women's predictions...")
    womens_predictions = train_and_predict_model(
        womens_prediction_data, "women's", [], None, PREDICTION_SEASONS,
        model=womens_model,
        feature_cols=womens_metadata['feature_cols'],
        scaler=womens_metadata['scaler'],
        dropped_features=womens_metadata['dropped_features']
    )
    
    # Combine predictions
    print("Combining predictions...")
    combined_submission = combine_predictions(mens_predictions, womens_predictions)
    
    # Save predictions
    combined_submission.to_csv("submission.csv", index=False)
    print("Predictions generated and saved to 'submission.csv'")

if __name__ == "__main__":
    main()
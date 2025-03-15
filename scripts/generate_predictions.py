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
from march_madness.models.evaluation import evaluate_predictions_against_actual

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

    # ----- MEN'S PREDICTIONS EVALUATION -----
    # Extract men's predictions
    mens_teams = pd.read_csv("data/MTeams.csv")
    mens_team_ids = set(mens_teams['TeamID'].values)

    # Filter predictions to include only men's teams
    mens_predictions = combined_submission[
        combined_submission['ID'].apply(lambda x: 
            int(x.split('_')[1]) in mens_team_ids or 
            int(x.split('_')[2]) in mens_team_ids
        )
    ]

    # Convert to the format needed for evaluation - using list comprehension
    mens_pred_rows = []
    for _, row in mens_predictions.iterrows():
        parts = row['ID'].split('_')
        season, team1, team2 = int(parts[0]), int(parts[1]), int(parts[2])
        mens_pred_rows.append({
            'Season': season,
            'Team1ID': team1,
            'Team2ID': team2,
            'Pred': row['Pred'],
            'MatchupID': row['ID']
        })

    # Create DataFrame from list of dictionaries
    mens_pred_df = pd.DataFrame(mens_pred_rows)

    # Load actual results
    mens_results = pd.read_csv("data/MNCAATourneyDetailedResults.csv")
    mens_2024_results = mens_results[mens_results['Season'] == 2024]

    if len(mens_2024_results) > 0:
        print(f"Found {len(mens_2024_results)} actual men's tournament games for 2024")
        # Evaluate predictions
        mens_evaluation = evaluate_predictions_against_actual(mens_pred_df, mens_2024_results, gender="men's")
    else:
        print("No 2024 men's tournament results found in the data")

    # ----- WOMEN'S PREDICTIONS EVALUATION -----
    # Extract women's predictions
    womens_teams = pd.read_csv("data/WTeams.csv")
    womens_team_ids = set(womens_teams['TeamID'].values)

    # Filter predictions to include only women's teams
    womens_predictions = combined_submission[
        combined_submission['ID'].apply(lambda x: 
            int(x.split('_')[1]) in womens_team_ids or 
            int(x.split('_')[2]) in womens_team_ids
        )
    ]

    # Convert to the format needed for evaluation
    womens_pred_rows = []
    for _, row in womens_predictions.iterrows():
        parts = row['ID'].split('_')
        season, team1, team2 = int(parts[0]), int(parts[1]), int(parts[2])
        womens_pred_rows.append({
            'Season': season,
            'Team1ID': team1,
            'Team2ID': team2,
            'Pred': row['Pred'],
            'MatchupID': row['ID']
        })

    # Create DataFrame from list of dictionaries
    womens_pred_df = pd.DataFrame(womens_pred_rows)

    # Load actual women's results
    womens_results = pd.read_csv("data/WNCAATourneyDetailedResults.csv")
    womens_2024_results = womens_results[womens_results['Season'] == 2024]

    if len(womens_2024_results) > 0:
        print(f"Found {len(womens_2024_results)} actual women's tournament games for 2024")
        # Evaluate predictions
        womens_evaluation = evaluate_predictions_against_actual(womens_pred_df, womens_2024_results, gender="women's")
        
        # Print comparison if both evaluations are available
        if 'mens_evaluation' in locals() and mens_evaluation and womens_evaluation:
            print("\n===== Comparison of Men's vs Women's Prediction Performance =====")
            print(f"Men's Accuracy: {mens_evaluation['accuracy']:.4f}, Brier Score: {mens_evaluation['brier_score']:.4f}")
            print(f"Women's Accuracy: {womens_evaluation['accuracy']:.4f}, Brier Score: {womens_evaluation['brier_score']:.4f}")
    else:
        print("No 2024 women's tournament results found in the data")

if __name__ == "__main__":
    main()
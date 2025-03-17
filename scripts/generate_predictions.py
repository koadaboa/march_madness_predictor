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
from march_madness.models.evaluation import evaluate_predictions_against_actual, evaluate_predictions_by_tournament_round

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

    # ----- TOURNAMENT PREDICTIONS EVALUATION -----
    # Load team data to separate men's and women's predictions
    print("\nBeginning comprehensive tournament evaluation...")
    mens_teams = pd.read_csv("data/MTeams.csv")
    womens_teams = pd.read_csv("data/WTeams.csv")

    mens_team_ids = set(mens_teams['TeamID'].values)
    womens_team_ids = set(womens_teams['TeamID'].values)

    # Filter predictions to include only men's teams
    mens_predictions = combined_submission[
        combined_submission['ID'].apply(lambda x: 
            int(x.split('_')[1]) in mens_team_ids or 
            int(x.split('_')[2]) in mens_team_ids
        )
    ]

    # Convert to the format needed for evaluation
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

    # Load all tournament results and seed data
    mens_results = pd.read_csv("data/MNCAATourneyDetailedResults.csv")
    womens_results = pd.read_csv("data/WNCAATourneyDetailedResults.csv")
    mens_seeds = pd.read_csv("data/MNCAATourneySeeds.csv")
    womens_seeds = pd.read_csv("data/WNCAATourneySeeds.csv")

    print(f"Evaluating performance for {len(PREDICTION_SEASONS)} prediction seasons: {PREDICTION_SEASONS}")

    # Dictionaries to store combined evaluation metrics
    mens_combined_metrics = {
        'total_games': 0,
        'correct_predictions': 0,
        'total_brier_score': 0,
        'total_log_loss': 0,
        'rounds_data': {},
        'upset_correct': 0,
        'upset_total': 0,
        'non_upset_correct': 0,
        'non_upset_total': 0
    }

    womens_combined_metrics = {
        'total_games': 0,
        'correct_predictions': 0,
        'total_brier_score': 0,
        'total_log_loss': 0,
        'rounds_data': {},
        'upset_correct': 0,
        'upset_total': 0,
        'non_upset_correct': 0,
        'non_upset_total': 0
    }

    # Create a column to store all matched games for detailed analysis
    all_mens_matched_games = []
    all_womens_matched_games = []

    # Evaluate each prediction season
    for season in PREDICTION_SEASONS:
        print(f"\n===== Evaluating Season {season} =====")
        
        # Filter results for this season
        mens_season_results = mens_results[mens_results['Season'] == season]
        womens_season_results = womens_results[womens_results['Season'] == season]
        
        # Filter seeds for this season
        mens_season_seeds = mens_seeds[mens_seeds['Season'] == season]
        womens_season_seeds = womens_seeds[womens_seeds['Season'] == season]
        
        # Men's evaluation
        if len(mens_season_results) > 0:
            print(f"Found {len(mens_season_results)} actual men's tournament games for season {season}")
            
            # Basic evaluation
            mens_season_eval = evaluate_predictions_against_actual(mens_pred_df, mens_season_results, gender="men's")
            
            if mens_season_eval:
                # Update combined metrics
                games_count = len(mens_season_eval['results_df'])
                mens_combined_metrics['total_games'] += games_count
                mens_combined_metrics['correct_predictions'] += sum(mens_season_eval['results_df']['Correct'])
                mens_combined_metrics['total_brier_score'] += mens_season_eval['brier_score'] * games_count
                mens_combined_metrics['total_log_loss'] += mens_season_eval['log_loss'] * games_count
                
                # Store all matched games for further analysis
                all_mens_matched_games.append(mens_season_eval['results_df'])
            
            # Round-specific evaluation
            mens_round_eval = evaluate_predictions_by_tournament_round(
                mens_pred_df, mens_season_results, mens_season_seeds, gender="men's"
            )
            
            if mens_round_eval:
                # Update rounds data
                for _, row in mens_round_eval['round_metrics'].iterrows():
                    round_name = row['Round']
                    if round_name not in mens_combined_metrics['rounds_data']:
                        mens_combined_metrics['rounds_data'][round_name] = {
                            'games': 0, 'correct': 0, 'brier_sum': 0
                        }
                    
                    mens_combined_metrics['rounds_data'][round_name]['games'] += row['Count']
                    mens_combined_metrics['rounds_data'][round_name]['correct'] += row['Count'] * row['Accuracy']
                    mens_combined_metrics['rounds_data'][round_name]['brier_sum'] += row['Brier'] * row['Count']
                
                # Update upset metrics
                upset_df = mens_round_eval['results_df'][mens_round_eval['results_df']['Upset']]
                non_upset_df = mens_round_eval['results_df'][~mens_round_eval['results_df']['Upset']]
                
                mens_combined_metrics['upset_correct'] += sum(upset_df['Correct'])
                mens_combined_metrics['upset_total'] += len(upset_df)
                mens_combined_metrics['non_upset_correct'] += sum(non_upset_df['Correct'])
                mens_combined_metrics['non_upset_total'] += len(non_upset_df)
        else:
            print(f"No men's tournament results found for season {season}")
        
        # Women's evaluation
        if len(womens_season_results) > 0:
            print(f"Found {len(womens_season_results)} actual women's tournament games for season {season}")
            
            # Basic evaluation
            womens_season_eval = evaluate_predictions_against_actual(womens_pred_df, womens_season_results, gender="women's")
            
            if womens_season_eval:
                # Update combined metrics
                games_count = len(womens_season_eval['results_df'])
                womens_combined_metrics['total_games'] += games_count
                womens_combined_metrics['correct_predictions'] += sum(womens_season_eval['results_df']['Correct'])
                womens_combined_metrics['total_brier_score'] += womens_season_eval['brier_score'] * games_count
                womens_combined_metrics['total_log_loss'] += womens_season_eval['log_loss'] * games_count
                
                # Store all matched games for further analysis
                all_womens_matched_games.append(womens_season_eval['results_df'])
            
            # Round-specific evaluation
            womens_round_eval = evaluate_predictions_by_tournament_round(
                womens_pred_df, womens_season_results, womens_season_seeds, gender="women's"
            )
            
            if womens_round_eval:
                # Update rounds data
                for _, row in womens_round_eval['round_metrics'].iterrows():
                    round_name = row['Round']
                    if round_name not in womens_combined_metrics['rounds_data']:
                        womens_combined_metrics['rounds_data'][round_name] = {
                            'games': 0, 'correct': 0, 'brier_sum': 0
                        }
                    
                    womens_combined_metrics['rounds_data'][round_name]['games'] += row['Count']
                    womens_combined_metrics['rounds_data'][round_name]['correct'] += row['Count'] * row['Accuracy']
                    womens_combined_metrics['rounds_data'][round_name]['brier_sum'] += row['Brier'] * row['Count']
                
                # Update upset metrics
                upset_df = womens_round_eval['results_df'][womens_round_eval['results_df']['Upset']]
                non_upset_df = womens_round_eval['results_df'][~womens_round_eval['results_df']['Upset']]
                
                womens_combined_metrics['upset_correct'] += sum(upset_df['Correct'])
                womens_combined_metrics['upset_total'] += len(upset_df)
                womens_combined_metrics['non_upset_correct'] += sum(non_upset_df['Correct'])
                womens_combined_metrics['non_upset_total'] += len(non_upset_df)
        else:
            print(f"No women's tournament results found for season {season}")

    # Calculate final aggregate metrics
    if mens_combined_metrics['total_games'] > 0:
        mens_accuracy = mens_combined_metrics['correct_predictions'] / mens_combined_metrics['total_games']
        mens_brier = mens_combined_metrics['total_brier_score'] / mens_combined_metrics['total_games']
        mens_log_loss = mens_combined_metrics['total_log_loss'] / mens_combined_metrics['total_games']
        
        # Calculate round-specific metrics
        mens_round_metrics = []
        for round_name, data in mens_combined_metrics['rounds_data'].items():
            if data['games'] > 0:
                mens_round_metrics.append({
                    'Round': round_name,
                    'Count': data['games'],
                    'Accuracy': data['correct'] / data['games'],
                    'Brier': data['brier_sum'] / data['games']
                })
        
        mens_rounds_df = pd.DataFrame(mens_round_metrics).sort_values('Round')
        
        # Calculate upset metrics
        mens_upset_accuracy = mens_combined_metrics['upset_correct'] / mens_combined_metrics['upset_total'] if mens_combined_metrics['upset_total'] > 0 else 0
        mens_non_upset_accuracy = mens_combined_metrics['non_upset_correct'] / mens_combined_metrics['non_upset_total'] if mens_combined_metrics['non_upset_total'] > 0 else 0
        
        print("\n===== MEN'S TOURNAMENT OVERALL EVALUATION (ALL SEASONS) =====")
        print(f"Total Games: {mens_combined_metrics['total_games']}")
        print(f"Overall Accuracy: {mens_accuracy:.4f}")
        print(f"Overall Brier Score: {mens_brier:.4f}")
        print(f"Overall Log Loss: {mens_log_loss:.4f}")
        
        print("\nPerformance by Round:")
        print(mens_rounds_df.to_string(index=False))
        
        print("\nUpset Detection:")
        print(f"  Total Upsets: {mens_combined_metrics['upset_total']} ({mens_combined_metrics['upset_total']/mens_combined_metrics['total_games']*100:.1f}%)")
        print(f"  Accuracy on Upsets: {mens_upset_accuracy:.4f}")
        print(f"  Accuracy on Non-Upsets: {mens_non_upset_accuracy:.4f}")

    if womens_combined_metrics['total_games'] > 0:
        womens_accuracy = womens_combined_metrics['correct_predictions'] / womens_combined_metrics['total_games']
        womens_brier = womens_combined_metrics['total_brier_score'] / womens_combined_metrics['total_games']
        womens_log_loss = womens_combined_metrics['total_log_loss'] / womens_combined_metrics['total_games']
        
        # Calculate round-specific metrics
        womens_round_metrics = []
        for round_name, data in womens_combined_metrics['rounds_data'].items():
            if data['games'] > 0:
                womens_round_metrics.append({
                    'Round': round_name,
                    'Count': data['games'],
                    'Accuracy': data['correct'] / data['games'],
                    'Brier': data['brier_sum'] / data['games']
                })
        
        womens_rounds_df = pd.DataFrame(womens_round_metrics).sort_values('Round')
        
        # Calculate upset metrics
        womens_upset_accuracy = womens_combined_metrics['upset_correct'] / womens_combined_metrics['upset_total'] if womens_combined_metrics['upset_total'] > 0 else 0
        womens_non_upset_accuracy = womens_combined_metrics['non_upset_correct'] / womens_combined_metrics['non_upset_total'] if womens_combined_metrics['non_upset_total'] > 0 else 0
        
        print("\n===== WOMEN'S TOURNAMENT OVERALL EVALUATION (ALL SEASONS) =====")
        print(f"Total Games: {womens_combined_metrics['total_games']}")
        print(f"Overall Accuracy: {womens_accuracy:.4f}")
        print(f"Overall Brier Score: {womens_brier:.4f}")
        print(f"Overall Log Loss: {womens_log_loss:.4f}")
        
        print("\nPerformance by Round:")
        print(womens_rounds_df.to_string(index=False))
        
        print("\nUpset Detection:")
        print(f"  Total Upsets: {womens_combined_metrics['upset_total']} ({womens_combined_metrics['upset_total']/womens_combined_metrics['total_games']*100:.1f}%)")
        print(f"  Accuracy on Upsets: {womens_upset_accuracy:.4f}")
        print(f"  Accuracy on Non-Upsets: {womens_non_upset_accuracy:.4f}")

    # Compare men's and women's performance
    if mens_combined_metrics['total_games'] > 0 and womens_combined_metrics['total_games'] > 0:
        print("\n===== COMPARISON OF MEN'S VS WOMEN'S PREDICTION PERFORMANCE (ALL SEASONS) =====")
        print(f"Men's Accuracy: {mens_accuracy:.4f}, Brier Score: {mens_brier:.4f}")
        print(f"Women's Accuracy: {womens_accuracy:.4f}, Brier Score: {womens_brier:.4f}")
        
        # Create all_mens_matched_df and all_womens_matched_df if there are any matched games
        if all_mens_matched_games:
            all_mens_matched_df = pd.concat(all_mens_matched_games, ignore_index=True)
            print(f"\nMen's Tournament: Analyzed {len(all_mens_matched_df)} games across {len(PREDICTION_SEASONS)} seasons")
        
        if all_womens_matched_games:
            all_womens_matched_df = pd.concat(all_womens_matched_games, ignore_index=True)
            print(f"Women's Tournament: Analyzed {len(all_womens_matched_df)} games across {len(PREDICTION_SEASONS)} seasons")

if __name__ == "__main__":
    main()
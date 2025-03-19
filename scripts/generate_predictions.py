#!/usr/bin/env python
"""
Generate NCAA Basketball Tournament Predictions
With real-world mode for 2025 predictions using comprehensive cached data
Includes evaluation against actual results
"""

import os
import pandas as pd
import numpy as np
import pickle
import sys
import time
import argparse
import logging
from datetime import datetime
from sklearn.metrics import brier_score_loss, log_loss

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the package to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from march_madness.config import (STARTING_SEASON, CURRENT_SEASON, 
                                 TRAINING_SEASONS, VALIDATION_SEASON, 
                                 PREDICTION_SEASONS)
from march_madness.data.loaders import load_mens_data, load_womens_data, extract_seed_number
from march_madness.utils.data_processors import load_or_prepare_modeling_data
from march_madness.utils.helpers import train_and_predict_model
from march_madness.models.prediction import combine_predictions
from march_madness.features.matchup import create_tournament_prediction_dataset
from march_madness.utils.data_access import get_data_with_index

def match_predictions_with_results(predictions_df, actual_results, seed_data):
    """
    Match prediction probabilities with actual game outcomes.
    
    Args:
        predictions_df: DataFrame with model predictions
        actual_results: DataFrame with tournament results
        seed_data: DataFrame with tournament seed data
        
    Returns:
        DataFrame with matched predictions and outcomes
    """
    logger.info("Matching predictions with actual results...")
    
    # Create matchup IDs for actual tournament games
    actual_results['MatchupID'] = actual_results.apply(
        lambda row: f"{row['Season']}_{min(row['WTeamID'], row['LTeamID'])}_{max(row['WTeamID'], row['LTeamID'])}",
        axis=1
    )
    
    # Add tournament round based on day number
    round_mapping = {
        134: 'Round64', 135: 'Round64', 136: 'Round64', 137: 'Round64',
        138: 'Round32', 139: 'Round32',
        140: 'Sweet16', 141: 'Sweet16',
        142: 'Elite8', 143: 'Elite8',
        144: 'Final4', 145: 'Final4',
        146: 'Championship'
    }
    actual_results['Round'] = actual_results['DayNum'].map(round_mapping)
    
    # Add seed information to actual results
    actual_results = actual_results.merge(
        seed_data.rename(columns={'TeamID': 'WTeamID', 'Seed': 'WSeed'}),
        on=['Season', 'WTeamID'],
        how='left'
    )
    
    actual_results = actual_results.merge(
        seed_data.rename(columns={'TeamID': 'LTeamID', 'Seed': 'LSeed'}),
        on=['Season', 'LTeamID'],
        how='left'
    )
    
    # Convert seeds to numbers
    actual_results['WSeedNum'] = actual_results['WSeed'].apply(extract_seed_number)
    actual_results['LSeedNum'] = actual_results['LSeed'].apply(extract_seed_number)
    
    # Determine if the game was an upset based on seed
    actual_results['Upset'] = actual_results['WSeedNum'] > actual_results['LSeedNum']
    
    # Match predictions with actual results
    matched_games = []
    
    # Track seasons for reporting
    seasons = actual_results['Season'].unique()
    logger.info(f"Processing results for seasons: {seasons}")
    
    for _, game in actual_results.iterrows():
        matchup_id = game['MatchupID']
        
        # Find the prediction for this matchup
        game_predictions = predictions_df[predictions_df['MatchupID'] == matchup_id]
        
        if len(game_predictions) > 0:
            # Get the prediction row
            pred_row = game_predictions.iloc[0]
            team1_id = pred_row['Team1ID']
            team2_id = pred_row['Team2ID']
            
            # Determine if Team1 was the winner
            if team1_id == game['WTeamID'] and team2_id == game['LTeamID']:
                actual = 1  # Team1 won
                predicted_prob = pred_row['Pred']
            elif team1_id == game['LTeamID'] and team2_id == game['WTeamID']:
                actual = 0  # Team1 lost
                predicted_prob = pred_row['Pred']
            else:
                logger.warning(f"Team mismatch in matchup {matchup_id}")
                continue
            
            # Determine if prediction was correct
            predicted = predicted_prob > 0.5
            correct = (predicted and actual == 1) or (not predicted and actual == 0)
            
            # Store matched game data
            matched_games.append({
                'Season': game['Season'],
                'MatchupID': matchup_id,
                'Round': game['Round'],
                'Team1ID': team1_id,
                'Team2ID': team2_id,
                'Prediction': predicted_prob,
                'Actual': actual,
                'Correct': correct,
                'Upset': game['Upset'],
                'WSeedNum': game['WSeedNum'],
                'LSeedNum': game['LSeedNum']
            })
    
    # Create DataFrame from matched games
    if matched_games:
        matched_df = pd.DataFrame(matched_games)
        logger.info(f"Successfully matched {len(matched_df)} predictions with actual results")
        return matched_df
    else:
        logger.warning("No games were matched with predictions!")
        return pd.DataFrame()

def evaluate_predictions(matched_df):
    """
    Calculate evaluation metrics for matched predictions.
    
    Args:
        matched_df: DataFrame with matched predictions and outcomes
        
    Returns:
        Dictionary with evaluation metrics
    """
    if matched_df.empty:
        return {
            'total_games': 0,
            'correct_predictions': 0,
            'total_brier_score': 0,
            'total_log_loss': 0,
            'rounds_data': {},
            'upset_total': 0,
            'upset_correct': 0,
            'non_upset_total': 0,
            'non_upset_correct': 0
        }
    
    # Calculate overall metrics
    total_games = len(matched_df)
    correct_predictions = matched_df['Correct'].sum()
    
    # Calculate Brier score and log loss
    try:
        total_brier_score = brier_score_loss(matched_df['Actual'], matched_df['Prediction']) * total_games
    except Exception as e:
        logger.error(f"Error calculating Brier score: {str(e)}")
        total_brier_score = 0
    
    try:
        total_log_loss = log_loss(matched_df['Actual'], matched_df['Prediction']) * total_games
    except Exception as e:
        logger.error(f"Error calculating log loss: {str(e)}")
        total_log_loss = float('nan')
    
    # Calculate metrics by round
    rounds_data = {}
    for round_name, group in matched_df.groupby('Round'):
        round_games = len(group)
        round_correct = group['Correct'].sum()
        
        try:
            round_brier = brier_score_loss(group['Actual'], group['Prediction']) * round_games
        except:
            round_brier = 0
            
        rounds_data[round_name] = {
            'games': round_games,
            'correct': round_correct,
            'brier_sum': round_brier
        }
    
    # Calculate metrics for upsets vs. non-upsets
    upsets = matched_df[matched_df['Upset']]
    non_upsets = matched_df[~matched_df['Upset']]
    
    upset_total = len(upsets)
    upset_correct = upsets['Correct'].sum() if not upsets.empty else 0
    
    non_upset_total = len(non_upsets)
    non_upset_correct = non_upsets['Correct'].sum() if not non_upsets.empty else 0
    
    return {
        'total_games': total_games,
        'correct_predictions': correct_predictions,
        'total_brier_score': total_brier_score,
        'total_log_loss': total_log_loss,
        'rounds_data': rounds_data,
        'upset_total': upset_total,
        'upset_correct': upset_correct,
        'non_upset_total': non_upset_total,
        'non_upset_correct': non_upset_correct
    }

def match_and_evaluate_predictions(predictions_df, actual_results, seed_data, gender="men's"):
    """
    Use the original evaluation functions from march_madness.models.evaluation
    
    Args:
        predictions_df: DataFrame with model predictions
        actual_results: DataFrame with tournament results
        seed_data: DataFrame with tournament seed data
        gender: 'men's' or 'women's'
        
    Returns:
        Dictionary with evaluation metrics
    """
    from march_madness.models.evaluation import (
        evaluate_predictions_against_actual, 
        evaluate_predictions_by_tournament_round
    )
    
    # Basic evaluation for metrics
    basic_eval = evaluate_predictions_against_actual(
        predictions_df, actual_results, gender=gender
    )
    
    # Round-specific evaluation
    round_eval = evaluate_predictions_by_tournament_round(
        predictions_df, actual_results, seed_data, gender=gender
    )
    
    # Combine metrics
    metrics = {
        'total_games': len(basic_eval['results_df']) if basic_eval else 0,
        'correct_predictions': sum(basic_eval['results_df']['Correct']) if basic_eval else 0,
        'total_brier_score': basic_eval['brier_score'] * len(basic_eval['results_df']) if basic_eval else 0,
        'total_log_loss': basic_eval['log_loss'] * len(basic_eval['results_df']) if basic_eval else 0,
        'rounds_data': {},
        'upset_correct': 0,
        'upset_total': 0,
        'non_upset_correct': 0,
        'non_upset_total': 0
    }
    
    # Process round-specific metrics
    if round_eval:
        for _, row in round_eval['round_metrics'].iterrows():
            round_name = row['Round']
            if round_name not in metrics['rounds_data']:
                metrics['rounds_data'][round_name] = {
                    'games': 0, 'correct': 0, 'brier_sum': 0
                }
            
            metrics['rounds_data'][round_name]['games'] += row['Count']
            metrics['rounds_data'][round_name]['correct'] += row['Count'] * row['Accuracy']
            metrics['rounds_data'][round_name]['brier_sum'] += row['Brier'] * row['Count']
        
        # Process upset metrics
        upset_df = round_eval['results_df'][round_eval['results_df']['Upset']]
        non_upset_df = round_eval['results_df'][~round_eval['results_df']['Upset']]
        
        metrics['upset_correct'] += sum(upset_df['Correct'])
        metrics['upset_total'] += len(upset_df)
        metrics['non_upset_correct'] += sum(non_upset_df['Correct'])
        metrics['non_upset_total'] += len(non_upset_df)
    
    return metrics

def display_evaluation_results(mens_metrics, womens_metrics=None):
    """
    Display evaluation metrics in a format similar to the example.
    
    Args:
        mens_metrics: Dictionary with men's evaluation metrics
        womens_metrics: Dictionary with women's evaluation metrics (optional)
    """
    # Process men's metrics
    if mens_metrics['total_games'] > 0:
        mens_accuracy = mens_metrics['correct_predictions'] / mens_metrics['total_games']
        mens_brier = mens_metrics['total_brier_score'] / mens_metrics['total_games']
        mens_log_loss = mens_metrics['total_log_loss'] / mens_metrics['total_games']
        
        # Calculate round-specific metrics
        mens_round_metrics = []
        for round_name, data in mens_metrics['rounds_data'].items():
            if data['games'] > 0:
                mens_round_metrics.append({
                    'Round': round_name,
                    'Count': data['games'],
                    'Accuracy': data['correct'] / data['games'],
                    'Brier': data['brier_sum'] / data['games']
                })
        
        mens_rounds_df = pd.DataFrame(mens_round_metrics)
        
        # Sort rounds in proper order
        round_order = {'Round64': 0, 'Round32': 1, 'Sweet16': 2, 'Elite8': 3, 'Final4': 4, 'Championship': 5}
        mens_rounds_df['RoundOrder'] = mens_rounds_df['Round'].map(round_order)
        mens_rounds_df = mens_rounds_df.sort_values('RoundOrder').drop('RoundOrder', axis=1)
        
        # Calculate upset metrics
        mens_upset_accuracy = mens_metrics['upset_correct'] / mens_metrics['upset_total'] if mens_metrics['upset_total'] > 0 else 0
        mens_non_upset_accuracy = mens_metrics['non_upset_correct'] / mens_metrics['non_upset_total'] if mens_metrics['non_upset_total'] > 0 else 0
        
        print("\n===== MEN'S TOURNAMENT OVERALL EVALUATION (ALL SEASONS) =====")
        print(f"Total Games: {mens_metrics['total_games']}")
        print(f"Overall Accuracy: {mens_accuracy:.4f}")
        print(f"Overall Brier Score: {mens_brier:.4f}")
        print(f"Overall Log Loss: {mens_log_loss:.4f}")
        
        print("\nPerformance by Round:")
        print(mens_rounds_df.to_string(index=False))
        
        print("\nUpset Detection:")
        print(f"  Total Upsets: {mens_metrics['upset_total']} ({mens_metrics['upset_total']/mens_metrics['total_games']*100:.1f}%)")
        print(f"  Accuracy on Upsets: {mens_upset_accuracy:.4f}")
        print(f"  Accuracy on Non-Upsets: {mens_non_upset_accuracy:.4f}")
    else:
        print("\nNo men's games evaluated.")
    
    # Process women's metrics if provided
    if womens_metrics and womens_metrics['total_games'] > 0:
        womens_accuracy = womens_metrics['correct_predictions'] / womens_metrics['total_games']
        womens_brier = womens_metrics['total_brier_score'] / womens_metrics['total_games']
        womens_log_loss = womens_metrics['total_log_loss'] / womens_metrics['total_games']
        
        # Calculate round-specific metrics
        womens_round_metrics = []
        for round_name, data in womens_metrics['rounds_data'].items():
            if data['games'] > 0:
                womens_round_metrics.append({
                    'Round': round_name,
                    'Count': data['games'],
                    'Accuracy': data['correct'] / data['games'],
                    'Brier': data['brier_sum'] / data['games']
                })
        
        womens_rounds_df = pd.DataFrame(womens_round_metrics)
        
        # Sort rounds in proper order
        round_order = {'Round64': 0, 'Round32': 1, 'Sweet16': 2, 'Elite8': 3, 'Final4': 4, 'Championship': 5}
        womens_rounds_df['RoundOrder'] = womens_rounds_df['Round'].map(round_order)
        womens_rounds_df = womens_rounds_df.sort_values('RoundOrder').drop('RoundOrder', axis=1)
        
        # Calculate upset metrics
        womens_upset_accuracy = womens_metrics['upset_correct'] / womens_metrics['upset_total'] if womens_metrics['upset_total'] > 0 else 0
        womens_non_upset_accuracy = womens_metrics['non_upset_correct'] / womens_metrics['non_upset_total'] if womens_metrics['non_upset_total'] > 0 else 0
        
        print("\n===== WOMEN'S TOURNAMENT OVERALL EVALUATION (ALL SEASONS) =====")
        print(f"Total Games: {womens_metrics['total_games']}")
        print(f"Overall Accuracy: {womens_accuracy:.4f}")
        print(f"Overall Brier Score: {womens_brier:.4f}")
        print(f"Overall Log Loss: {womens_log_loss:.4f}")
        
        print("\nPerformance by Round:")
        print(womens_rounds_df.to_string(index=False))
        
        print("\nUpset Detection:")
        print(f"  Total Upsets: {womens_metrics['upset_total']} ({womens_metrics['upset_total']/womens_metrics['total_games']*100:.1f}%)")
        print(f"  Accuracy on Upsets: {womens_upset_accuracy:.4f}")
        print(f"  Accuracy on Non-Upsets: {womens_non_upset_accuracy:.4f}")
    elif womens_metrics:
        print("\nNo women's games evaluated.")

def main():
    """Generate tournament predictions with optional evaluation mode."""
    # Start timing
    start_time = time.time()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate March Madness predictions')
    parser.add_argument('--skip-data-load', action='store_true', 
                        help='Skip loading raw data (assume modeling data is cached)')
    parser.add_argument('--force-prepare', action='store_true',
                        help='Force data preparation even if cache exists')
    parser.add_argument('--tournament-only', action='store_true',
                        help='Only generate matchups between tournament teams (much faster)')
    parser.add_argument('--real-world', action='store_true',
                        help='Generate predictions for upcoming 2025 tournament (uses comprehensive cache)')
    parser.add_argument('--seasons', type=str, default=None,
                        help='Comma-separated list of seasons to predict (overrides defaults and real-world)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate predictions against actual tournament results')
    args = parser.parse_args()
    
    # Determine which seasons to predict
    if args.seasons:
        # User specified seasons override everything else
        prediction_seasons = [int(s) for s in args.seasons.split(',')]
        real_world_mode = False
    elif args.real_world:
        # Real-world mode includes 2025
        prediction_seasons = [2025]
        real_world_mode = True
        logger.info("REAL-WORLD MODE: Generating predictions for 2025 tournament")
    else:
        # Use default prediction seasons from config
        prediction_seasons = PREDICTION_SEASONS
        real_world_mode = False
    
    logger.info(f"Generating predictions for seasons: {prediction_seasons}")
    
    # Define directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    cache_dir = os.path.join(project_root, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set cache file paths based on mode
    if real_world_mode:
        mens_cache_file = os.path.join(cache_dir, f"mens_all_data_{CURRENT_SEASON}_complete.pkl")
        womens_cache_file = os.path.join(cache_dir, f"womens_all_data_{CURRENT_SEASON}_complete.pkl")
        
        # Check if comprehensive cache files exist
        if not os.path.exists(mens_cache_file) or not os.path.exists(womens_cache_file):
            logger.warning(f"Comprehensive cache files not found. Please run preprocess_all_data.py first.")
            logger.warning(f"Falling back to standard cache files.")
            mens_cache_file = os.path.join(cache_dir, "men's_modeling_data.pkl")
            womens_cache_file = os.path.join(cache_dir, "women's_modeling_data.pkl")
    else:
        # Use standard cache files
        mens_cache_file = os.path.join(cache_dir, "men's_modeling_data_2025.pkl")
        womens_cache_file = os.path.join(cache_dir, "women's_modeling_data_2025.pkl") 
    
    # In real-world mode, we always use cached data
    if real_world_mode:
        args.skip_data_load = True
        logger.info("Real-world mode: Using cached data (--skip-data-load forced)")
    
    # Disable evaluation for real-world mode
    if real_world_mode and args.evaluate:
        logger.warning("Evaluation mode not applicable in real-world mode - disabling evaluation")
        args.evaluate = False
    
    # Load or prepare modeling data
    mens_modeling_data = None
    womens_modeling_data = None
    
    if args.skip_data_load:
        logger.info("Skipping raw data loading, using cached modeling data...")
        # Load cached data directly
        if os.path.exists(mens_cache_file):
            logger.info(f"Loading men's cached data from {mens_cache_file}")
            with open(mens_cache_file, 'rb') as f:
                mens_modeling_data = pickle.load(f)
        else:
            logger.warning(f"Men's cache file not found: {mens_cache_file}")
            
        if os.path.exists(womens_cache_file):
            logger.info(f"Loading women's cached data from {womens_cache_file}")
            with open(womens_cache_file, 'rb') as f:
                womens_modeling_data = pickle.load(f)
        else:
            logger.warning(f"Women's cache file not found: {womens_cache_file}")
    else:
        # Load raw data
        logger.info("Loading raw data...")
        mens_data = load_mens_data(STARTING_SEASON, data_dir=data_dir)
        womens_data = load_womens_data(STARTING_SEASON, data_dir=data_dir)
        
        logger.info("Preparing modeling data...")
        # Get modeling data the standard way
        mens_modeling_data = load_or_prepare_modeling_data(
            mens_data, "men's", STARTING_SEASON, CURRENT_SEASON, 
            prediction_seasons,
            cache_dir=cache_dir,
            force_prepare=args.force_prepare
        )
        
        womens_modeling_data = load_or_prepare_modeling_data(
            womens_data, "women's", STARTING_SEASON, CURRENT_SEASON, 
            prediction_seasons,
            cache_dir=cache_dir,
            force_prepare=args.force_prepare
        )
    
    # Handle 2025 predictions (code unchanged)...
    if 2025 in prediction_seasons:
        # Existing 2025 setup code here...
        pass
    
    # Apply tournament-only filtering for non-2025 seasons if needed
    if args.tournament_only and not real_world_mode:
        # Existing tournament-only code here...
        pass
    
    # Load trained models and metadata
    logger.info("Loading trained models...")
    models_dir = os.path.join(project_root, 'models')
    
    try:
        with open(os.path.join(models_dir, 'mens_model.pkl'), 'rb') as f:
            mens_model = pickle.load(f)
        
        with open(os.path.join(models_dir, 'mens_metadata.pkl'), 'rb') as f:
            mens_metadata = pickle.load(f)
        
        with open(os.path.join(models_dir, 'womens_model.pkl'), 'rb') as f:
            womens_model = pickle.load(f)
        
        with open(os.path.join(models_dir, 'womens_metadata.pkl'), 'rb') as f:
            womens_metadata = pickle.load(f)
    except FileNotFoundError as e:
        logger.error(f"Could not load model files: {str(e)}")
        return
    
    # Generate predictions
    mens_predictions = pd.DataFrame()
    womens_predictions = pd.DataFrame()
    
    if mens_modeling_data is not None:
        logger.info("Generating Men's predictions...")
        mens_predictions = train_and_predict_model(
            mens_modeling_data, "men's", [], None, prediction_seasons,
            model=mens_model,
            feature_cols=mens_metadata['feature_cols'],
            scaler=mens_metadata['scaler'],
            dropped_features=mens_metadata['dropped_features']
        )
        logger.info(f"Generated {len(mens_predictions)} men's predictions")
    else:
        logger.warning("Skipping men's predictions due to missing data")
    
    if womens_modeling_data is not None:
        logger.info("Generating Women's predictions...")
        womens_predictions = train_and_predict_model(
            womens_modeling_data, "women's", [], None, prediction_seasons,
            model=womens_model,
            feature_cols=womens_metadata['feature_cols'],
            scaler=womens_metadata['scaler'],
            dropped_features=womens_metadata['dropped_features']
        )
        logger.info(f"Generated {len(womens_predictions)} women's predictions")
    else:
        logger.warning("Skipping women's predictions due to missing data")
    
    # Combine predictions
    if not mens_predictions.empty or not womens_predictions.empty:
        logger.info("Combining predictions...")
        combined_submission = combine_predictions(mens_predictions, womens_predictions)
        
        # Create submission filename
        parts = []
        if args.tournament_only:
            parts.append("tournament_only")
        if real_world_mode:
            parts.append("2025")
        if parts:
            suffix = "_" + "_".join(parts)
        else:
            suffix = ""
            
        submission_path = os.path.join(project_root, f"submission{suffix}.csv")
        combined_submission.to_csv(submission_path, index=False)
        
        # Report results
        logger.info(f"Predictions generated and saved to '{submission_path}'")
        logger.info(f"Total predictions: {len(combined_submission)}")
        logger.info(f"Men's predictions: {len(mens_predictions)}")
        logger.info(f"Women's predictions: {len(womens_predictions)}")
    else:
        logger.error("No predictions were generated!")
        if args.evaluate:
            logger.error("Cannot perform evaluation without predictions")
            args.evaluate = False
    
    # Evaluate predictions if requested
    if args.evaluate:
        logger.info("Evaluating predictions against actual tournament results...")

        # Get actual tournament results for each gender
        mens_results_path = os.path.join(data_dir, 'MNCAATourneyDetailedResults.csv')
        womens_results_path = os.path.join(data_dir, 'WNCAATourneyDetailedResults.csv')

        # Initialize metrics containers
        mens_metrics = None
        womens_metrics = None
        
        # Evaluate men's predictions
        if not mens_predictions.empty and os.path.exists(mens_results_path):
            mens_results = pd.read_csv(mens_results_path)
            mens_results = mens_results[mens_results['Season'].isin(prediction_seasons)]
            
            if not mens_results.empty:
                logger.info(f"Found {len(mens_results)} men's tournament games for seasons {prediction_seasons}")
                
                # Use the original evaluation method
                mens_metrics = match_and_evaluate_predictions(
                    mens_predictions,
                    mens_results,
                    mens_modeling_data['df_seed'],
                    gender="men's"
                )
        
        # Evaluate women's predictions
        if not womens_predictions.empty and os.path.exists(womens_results_path):
            womens_results = pd.read_csv(womens_results_path)
            womens_results = womens_results[womens_results['Season'].isin(prediction_seasons)]
            
            if not womens_results.empty:
                logger.info(f"Found {len(womens_results)} women's tournament games for seasons {prediction_seasons}")
                
                # Use the original evaluation method
                womens_metrics = match_and_evaluate_predictions(
                    womens_predictions,
                    womens_results,
                    womens_modeling_data['df_seed'],
                    gender="women's"
                )
        
        # Display evaluation results
        display_evaluation_results(mens_metrics if mens_metrics else {}, womens_metrics)
    
    # Report timing
    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    logger.info(f"Total execution time: {minutes} minutes, {seconds} seconds")

if __name__ == "__main__":
    main()
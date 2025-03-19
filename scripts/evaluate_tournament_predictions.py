#!/usr/bin/env python
"""
Evaluate NCAA Tournament predictions against actual tournament results.
This script loads tournament predictions and compares them to actual game outcomes.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
from sklearn.metrics import brier_score_loss, log_loss
import sys

# Add the package to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from march_madness.data.loaders import extract_seed_number

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def map_seed_to_region(seed_str):
    """Map a seed string to a region letter."""
    if isinstance(seed_str, str) and len(seed_str) >= 1:
        return seed_str[0]
    return 'Unknown'

def map_day_to_round(day_num):
    """Map a tournament day number to a round name."""
    round_mapping = {
        134: 'Round64', 135: 'Round64', 136: 'Round64', 137: 'Round64',
        138: 'Round32', 139: 'Round32',
        140: 'Sweet16', 141: 'Sweet16',
        142: 'Elite8', 143: 'Elite8',
        144: 'Final4', 145: 'Final4',
        146: 'Championship'
    }
    return round_mapping.get(day_num, 'Unknown')

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
    actual_results['Round'] = actual_results['DayNum'].map(map_day_to_round)
    
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
    
    # Add regions
    actual_results['WRegion'] = actual_results['WSeed'].apply(map_seed_to_region)
    actual_results['LRegion'] = actual_results['LSeed'].apply(map_seed_to_region)
    
    # Determine if the game was an upset based on seed
    actual_results['Upset'] = actual_results['WSeedNum'] > actual_results['LSeedNum']
    
    # Calculate seed difference
    actual_results['SeedDiff'] = actual_results['WSeedNum'] - actual_results['LSeedNum']
    
    # Match predictions with actual results
    matched_games = []
    
    # Track seasons for reporting
    seasons = actual_results['Season'].unique()
    logger.info(f"Processing results for seasons: {seasons}")
    
    for _, game in actual_results.iterrows():
        matchup_id = game['MatchupID']
        season = game['Season']
        
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
                'Season': season,
                'MatchupID': matchup_id,
                'Round': game['Round'],
                'WTeamID': game['WTeamID'],
                'LTeamID': game['LTeamID'],
                'Team1ID': team1_id,
                'Team2ID': team2_id,
                'Prediction': predicted_prob,
                'Actual': actual,
                'Correct': correct,
                'SeedDiff': game['SeedDiff'],
                'Upset': game['Upset'],
                'WSeedNum': game['WSeedNum'],
                'LSeedNum': game['LSeedNum'],
                'WRegion': game['WRegion'],
                'LRegion': game['LRegion'],
                'ScoreDiff': game['WScore'] - game['LScore']
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
        return {}
    
    # Calculate overall metrics
    accuracy = matched_df['Correct'].mean()
    brier_score = brier_score_loss(matched_df['Actual'], matched_df['Prediction'])
    
    try:
        logloss = log_loss(matched_df['Actual'], matched_df['Prediction'])
    except:
        logloss = float('nan')
    
    # Calculate metrics by round
    rounds_metrics = matched_df.groupby('Round').agg({
        'MatchupID': 'count',
        'Correct': 'mean',
        'Actual': lambda x: x.tolist(),
        'Prediction': lambda x: x.tolist()
    }).reset_index()
    
    # Calculate Brier score by round
    rounds_metrics['BrierScore'] = rounds_metrics.apply(
        lambda row: brier_score_loss(row['Actual'], row['Prediction']) 
                   if len(row['Actual']) > 0 else float('nan'), 
        axis=1
    )
    
    # Calculate metrics for upsets vs. non-upsets
    upset_metrics = matched_df.groupby('Upset').agg({
        'MatchupID': 'count',
        'Correct': 'mean',
        'Actual': lambda x: x.tolist(),
        'Prediction': lambda x: x.tolist()
    }).reset_index()
    
    upset_metrics['BrierScore'] = upset_metrics.apply(
        lambda row: brier_score_loss(row['Actual'], row['Prediction'])
                   if len(row['Actual']) > 0 else float('nan'),
        axis=1
    )
    
    # Analyze prediction confidence
    matched_df['Confidence'] = abs(matched_df['Prediction'] - 0.5) * 2  # Scale to 0-1
    confidence_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    matched_df['ConfidenceBin'] = pd.cut(matched_df['Confidence'], bins=confidence_bins)
    
    confidence_metrics = matched_df.groupby('ConfidenceBin').agg({
        'MatchupID': 'count',
        'Correct': 'mean'
    }).reset_index()
    
    return {
        'overall': {
            'games': len(matched_df),
            'accuracy': accuracy,
            'brier_score': brier_score,
            'log_loss': logloss
        },
        'rounds': rounds_metrics,
        'upsets': upset_metrics,
        'confidence': confidence_metrics,
        'matched_df': matched_df
    }

def create_evaluation_visualizations(evaluation_results, output_dir):
    """
    Create visualization plots for evaluation results.
    
    Args:
        evaluation_results: Dictionary with evaluation metrics
        output_dir: Directory to save visualization files
    """
    if not evaluation_results:
        logger.warning("No evaluation results to visualize")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    matched_df = evaluation_results['matched_df']
    
    # 1. Accuracy by round
    plt.figure(figsize=(10, 6))
    rounds_df = evaluation_results['rounds'].sort_values('Round')
    
    # Create round order
    round_order = ['Round64', 'Round32', 'Sweet16', 'Elite8', 'Final4', 'Championship']
    rounds_df['Round'] = pd.Categorical(rounds_df['Round'], categories=round_order, ordered=True)
    rounds_df = rounds_df.sort_values('Round')
    
    ax = sns.barplot(x='Round', y='Correct', data=rounds_df)
    
    # Add count labels
    for i, row in rounds_df.iterrows():
        ax.text(i, row['Correct'] + 0.02, f"n={row['MatchupID']}", ha='center')
    
    plt.title('Prediction Accuracy by Tournament Round')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(output_dir, 'accuracy_by_round.png'))
    
    # 2. Accuracy for upsets vs. non-upsets
    plt.figure(figsize=(8, 6))
    upset_df = evaluation_results['upsets']
    ax = sns.barplot(x='Upset', y='Correct', data=upset_df)
    
    # Add count labels
    for i, row in upset_df.iterrows():
        ax.text(i, row['Correct'] + 0.02, f"n={row['MatchupID']}", ha='center')
    
    plt.title('Prediction Accuracy for Upsets vs. Non-Upsets')
    plt.xlabel('Upset Game')
    plt.ylabel('Accuracy')
    plt.xticks([0, 1], ['Non-Upset', 'Upset'])
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(output_dir, 'accuracy_by_upset.png'))
    
    # 3. Seed difference vs. prediction accuracy
    plt.figure(figsize=(10, 6))
    
    # Group by absolute seed difference
    matched_df['AbsSeedDiff'] = abs(matched_df['SeedDiff'])
    seed_diff_metrics = matched_df.groupby('AbsSeedDiff').agg({
        'MatchupID': 'count',
        'Correct': 'mean'
    }).reset_index()
    
    # Plot only seed differences with enough games
    seed_diff_metrics = seed_diff_metrics[seed_diff_metrics['MatchupID'] >= 2]
    
    ax = sns.barplot(x='AbsSeedDiff', y='Correct', data=seed_diff_metrics)
    
    # Add count labels
    for i, row in seed_diff_metrics.iterrows():
        ax.text(i, row['Correct'] + 0.02, f"n={row['MatchupID']}", ha='center')
    
    plt.title('Prediction Accuracy by Absolute Seed Difference')
    plt.xlabel('Absolute Seed Difference')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(output_dir, 'accuracy_by_seed_diff.png'))
    
    # 4. Prediction confidence vs. accuracy
    plt.figure(figsize=(10, 6))
    confidence_df = evaluation_results['confidence']
    
    ax = sns.barplot(x='ConfidenceBin', y='Correct', data=confidence_df)
    
    # Add count labels
    for i, row in confidence_df.iterrows():
        ax.text(i, row['Correct'] + 0.02, f"n={row['MatchupID']}", ha='center')
    
    plt.title('Prediction Accuracy by Confidence Level')
    plt.xlabel('Confidence Level (0=lowest, 1=highest)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.savefig(os.path.join(output_dir, 'accuracy_by_confidence.png'))
    
    # 5. Calibration curve
    plt.figure(figsize=(10, 6))
    
    # Create bins for predicted probabilities
    prob_bins = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0
    matched_df['ProbBin'] = pd.cut(matched_df['Prediction'], bins=prob_bins)
    
    calibration_df = matched_df.groupby('ProbBin').agg({
        'MatchupID': 'count',
        'Actual': 'mean',
        'Prediction': 'mean'
    }).reset_index()
    
    # Plot the calibration curve
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.scatter(calibration_df['Prediction'], calibration_df['Actual'], 
                s=calibration_df['MatchupID']*5, alpha=0.7, label='Model')
    
    # Add count labels
    for _, row in calibration_df.iterrows():
        if not pd.isna(row['Prediction']) and not pd.isna(row['Actual']):
            plt.text(row['Prediction'], row['Actual'], f"n={row['MatchupID']}")
    
    plt.title('Calibration Curve: Predicted vs. Actual Probability')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Outcome Rate')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'calibration_curve.png'))
    
    logger.info(f"Visualizations saved to {output_dir}")

def main():
    """Main function to evaluate tournament predictions."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate NCAA Tournament predictions')
    parser.add_argument('--predictions', type=str, default='submission_tournament_only.csv',
                        help='Path to predictions CSV file')
    parser.add_argument('--seasons', type=str, default='2021,2022,2023',
                        help='Comma-separated list of seasons to evaluate')
    parser.add_argument('--gender', type=str, choices=['mens', 'womens', 'both'], default='both',
                        help='Which gender tournament to evaluate')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory for output files')
    parser.add_argument('--men-prefix', type=str, default='M', 
                        help='Prefix for men\'s data files')
    parser.add_argument('--women-prefix', type=str, default='W',
                        help='Prefix for women\'s data files')
    args = parser.parse_args()
    
    # Parse seasons
    seasons = [int(s) for s in args.seasons.split(',')]
    logger.info(f"Evaluating predictions for seasons: {seasons}")
    
    # Paths to required files
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    predictions_path = os.path.join(project_root, args.predictions)
    
    # Load predictions
    logger.info(f"Loading predictions from {predictions_path}")
    try:
        predictions_df = pd.read_csv(predictions_path)
        logger.info(f"Loaded {len(predictions_df)} predictions")
    except Exception as e:
        logger.error(f"Error loading predictions: {str(e)}")
        return
    
    # Extract season, team1, team2 from prediction ID
    predictions_df[['Season', 'Team1ID', 'Team2ID']] = predictions_df['ID'].str.split('_', expand=True)
    predictions_df['Season'] = predictions_df['Season'].astype(int)
    predictions_df['Team1ID'] = predictions_df['Team1ID'].astype(int)
    predictions_df['Team2ID'] = predictions_df['Team2ID'].astype(int)
    
    # Add matchup ID for joining with actual results
    predictions_df['MatchupID'] = predictions_df['ID']
    
    # Filter to requested seasons
    predictions_df = predictions_df[predictions_df['Season'].isin(seasons)]
    logger.info(f"Using {len(predictions_df)} predictions for seasons {seasons}")
    
    # Evaluate men's tournament if requested
    if args.gender in ['mens', 'both']:
        logger.info("Evaluating men's tournament predictions")
        
        # Load men's tournament results
        mens_results_path = os.path.join(data_dir, f'{args.men_prefix}NCAATourneyDetailedResults.csv')
        mens_seeds_path = os.path.join(data_dir, f'{args.men_prefix}NCAATourneySeeds.csv')
        
        try:
            mens_results = pd.read_csv(mens_results_path)
            mens_seeds = pd.read_csv(mens_seeds_path)
            
            # Filter to requested seasons
            mens_results = mens_results[mens_results['Season'].isin(seasons)]
            mens_seeds = mens_seeds[mens_seeds['Season'].isin(seasons)]
            
            logger.info(f"Loaded {len(mens_results)} men's tournament games")
            
            # Identify men's teams (to separate men's vs women's predictions)
            mens_team_ids = set()
            for _, row in mens_seeds.iterrows():
                mens_team_ids.add(row['TeamID'])
            
            # Filter predictions to men's teams only
            mens_predictions = predictions_df[
                predictions_df['Team1ID'].isin(mens_team_ids) | 
                predictions_df['Team2ID'].isin(mens_team_ids)
            ]
            
            logger.info(f"Using {len(mens_predictions)} men's predictions")
            
            # Match predictions with actual results
            mens_matched = match_predictions_with_results(mens_predictions, mens_results, mens_seeds)
            
            # Evaluate the matched predictions
            if not mens_matched.empty:
                mens_eval = evaluate_predictions(mens_matched)
                
                # Print evaluation summary
                logger.info("Men's Tournament Evaluation Summary:")
                logger.info(f"Total Games: {mens_eval['overall']['games']}")
                logger.info(f"Overall Accuracy: {mens_eval['overall']['accuracy']:.4f}")
                logger.info(f"Brier Score: {mens_eval['overall']['brier_score']:.4f}")
                logger.info(f"Log Loss: {mens_eval['overall']['log_loss']:.4f}")
                
                # Save evaluation results
                mens_output_dir = os.path.join(project_root, args.output_dir, 'mens')
                os.makedirs(mens_output_dir, exist_ok=True)
                
                # Save detailed results
                mens_matched.to_csv(os.path.join(mens_output_dir, 'matched_predictions.csv'), index=False)
                
                # Create visualizations
                create_evaluation_visualizations(mens_eval, mens_output_dir)
                
                # Round-specific evaluation
                logger.info("\nPerformance by Round:")
                print(mens_eval['rounds'][['Round', 'MatchupID', 'Correct', 'BrierScore']].to_string(index=False))
                
                # Upset evaluation
                logger.info("\nPerformance on Upsets:")
                print(mens_eval['upsets'][['Upset', 'MatchupID', 'Correct', 'BrierScore']].to_string(index=False))
            else:
                logger.warning("No men's games matched for evaluation!")
        
        except Exception as e:
            logger.error(f"Error evaluating men's predictions: {str(e)}", exc_info=True)
    
    # Evaluate women's tournament if requested
    if args.gender in ['womens', 'both']:
        logger.info("Evaluating women's tournament predictions")
        
        # Load women's tournament results
        womens_results_path = os.path.join(data_dir, f'{args.women_prefix}NCAATourneyDetailedResults.csv')
        womens_seeds_path = os.path.join(data_dir, f'{args.women_prefix}NCAATourneySeeds.csv')
        
        try:
            womens_results = pd.read_csv(womens_results_path)
            womens_seeds = pd.read_csv(womens_seeds_path)
            
            # Filter to requested seasons
            womens_results = womens_results[womens_results['Season'].isin(seasons)]
            womens_seeds = womens_seeds[womens_seeds['Season'].isin(seasons)]
            
            logger.info(f"Loaded {len(womens_results)} women's tournament games")
            
            # Identify women's teams
            womens_team_ids = set()
            for _, row in womens_seeds.iterrows():
                womens_team_ids.add(row['TeamID'])
            
            # Filter predictions to women's teams only
            womens_predictions = predictions_df[
                predictions_df['Team1ID'].isin(womens_team_ids) | 
                predictions_df['Team2ID'].isin(womens_team_ids)
            ]
            
            logger.info(f"Using {len(womens_predictions)} women's predictions")
            
            # Match predictions with actual results
            womens_matched = match_predictions_with_results(womens_predictions, womens_results, womens_seeds)
            
            # Evaluate the matched predictions
            if not womens_matched.empty:
                womens_eval = evaluate_predictions(womens_matched)
                
                # Print evaluation summary
                logger.info("Women's Tournament Evaluation Summary:")
                logger.info(f"Total Games: {womens_eval['overall']['games']}")
                logger.info(f"Overall Accuracy: {womens_eval['overall']['accuracy']:.4f}")
                logger.info(f"Brier Score: {womens_eval['overall']['brier_score']:.4f}")
                logger.info(f"Log Loss: {womens_eval['overall']['log_loss']:.4f}")
                
                # Save evaluation results
                womens_output_dir = os.path.join(project_root, args.output_dir, 'womens')
                os.makedirs(womens_output_dir, exist_ok=True)
                
                # Save detailed results
                womens_matched.to_csv(os.path.join(womens_output_dir, 'matched_predictions.csv'), index=False)
                
                # Create visualizations
                create_evaluation_visualizations(womens_eval, womens_output_dir)
                
                # Round-specific evaluation
                logger.info("\nPerformance by Round:")
                print(womens_eval['rounds'][['Round', 'MatchupID', 'Correct', 'BrierScore']].to_string(index=False))
                
                # Upset evaluation
                logger.info("\nPerformance on Upsets:")
                print(womens_eval['upsets'][['Upset', 'MatchupID', 'Correct', 'BrierScore']].to_string(index=False))
            else:
                logger.warning("No women's games matched for evaluation!")
        
        except Exception as e:
            logger.error(f"Error evaluating women's predictions: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
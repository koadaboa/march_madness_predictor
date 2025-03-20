# Model evaluation functions
import pandas as pd
import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from ..features.matchup import create_tournament_prediction_dataset
from ..models.common import create_feature_interactions


def calibrate_by_expected_round(predictions, X_test, seed_diff_col='SeedDiff', gender=None):
    """
    Calibrate predictions with gender-specific and round-specific adjustments
    based on comprehensive tournament analysis.
    """
    calibrated_preds = np.copy(predictions)
    seed_diffs = X_test[seed_diff_col].values if isinstance(X_test, pd.DataFrame) else X_test[:, seed_diff_col]
    
    # Get team seed information and tournament round if available
    tournament_round = None
    if isinstance(X_test, pd.DataFrame) and 'ExpectedRound' in X_test.columns:
        tournament_round = X_test['ExpectedRound'].values
    
    # Use both seed differences and win rate differences for calibration
    win_rate_diffs = X_test['WinRateDiff'].values if isinstance(X_test, pd.DataFrame) and 'WinRateDiff' in X_test.columns else None
    
    # ======= Gender-specific calibration values based on analysis ========
    if gender == "women's":
        # Women's tournament calibration values - MORE EXTREME FOR FAVORITES
        round_factors = {
            'Championship': {  # Women's championship has 50% accuracy, not 80%
                'heavy_favorite': 1.05,  # Much less confident in favorites
                'favorite': 1.03,
                'slight_favorite': 1.01,
                'even': 1.0,
                'slight_underdog': 0.99,
                'underdog': 0.97,
                'heavy_underdog': 0.95
            },
            'Final4': {  # Women's Final Four extremely unpredictable (25% accuracy)
                'heavy_favorite': 0.85,  # Be even more conservative with favorites
                'favorite': 0.80,
                'slight_favorite': 0.75,
                'even': 1.0,
                'slight_underdog': 1.25,  # Boost underdogs more
                'underdog': 1.30,
                'heavy_underdog': 1.35
            },
            'Elite8': {  # Women's Elite 8 is very predictable (87.5% accuracy)
                'heavy_favorite': 1.30,  # Much higher confidence
                'favorite': 1.25,
                'slight_favorite': 1.20,
                'even': 1.0,
                'slight_underdog': 0.80,
                'underdog': 0.75,
                'heavy_underdog': 0.70
            },
            'Sweet16': {  # Women's Sweet 16 is strong (73% accuracy)
                'heavy_favorite': 1.20,
                'favorite': 1.15,
                'slight_favorite': 1.10,
                'even': 1.0,
                'slight_underdog': 0.90,
                'underdog': 0.85,
                'heavy_underdog': 0.80
            },
            'Round32': {  # Women's Round of 32 is stronger than you thought (75% accuracy)
                'heavy_favorite': 1.25,  # Increase confidence
                'favorite': 1.20,
                'slight_favorite': 1.15,
                'even': 1.0,
                'slight_underdog': 0.85,
                'underdog': 0.80,
                'heavy_underdog': 0.75
            },
            'default': {  # Default factors for other rounds
                'heavy_favorite': 1.15,  # Slightly higher overall confidence
                'favorite': 1.10,
                'slight_favorite': 1.05,
                'even': 1.0,
                'slight_underdog': 0.92,
                'underdog': 0.85,
                'heavy_underdog': 0.80
            }
        }
        
        # Momentum factors
        momentum_factors = {
            'strong_form': 1.18,  # Increase from 1.12
            'good_form': 1.10,    # Increase from 1.06
            'neutral_form': 1.0,
            'poor_form': 0.90,    # Decrease from 0.95
            'weak_form': 0.85  
        }
    else:
        # Men's tournament calibration values
        round_factors = {
            'Championship': {  # Men are predictable in championship (70% accuracy)
                'heavy_favorite': 1.20,  # More confident in heavy favorites
                'favorite': 1.15,
                'slight_favorite': 1.10,
                'even': 1.0,
                'slight_underdog': 0.90,
                'underdog': 0.85,
                'heavy_underdog': 0.80
            },
            'Sweet16': {  # Men's Sweet 16 is very predictable (75% accuracy)
                'heavy_favorite': 1.18,
                'favorite': 1.15,
                'slight_favorite': 1.10,
                'even': 1.0,
                'slight_underdog': 0.90,
                'underdog': 0.85,
                'heavy_underdog': 0.80
            },
            'Final4': {  # Men's Final Four is unpredictable (42% accuracy)
                'heavy_favorite': 0.90,  # More doubt on favorites
                'favorite': 0.95,
                'slight_favorite': 0.98,
                'even': 1.0,
                'slight_underdog': 1.02,
                'underdog': 1.05,
                'heavy_underdog': 1.10   # Boost underdogs
            },
            'Elite8': {  # Men's Elite 8 is also unpredictable (42% accuracy)
                'heavy_favorite': 0.92,
                'favorite': 0.95,
                'slight_favorite': 0.98,
                'even': 1.0,
                'slight_underdog': 1.02,
                'underdog': 1.05,
                'heavy_underdog': 1.08
            },
            'default': {  # Default factors for other rounds
                'heavy_favorite': 1.10,
                'favorite': 1.05,
                'slight_favorite': 1.02,
                'even': 1.0,
                'slight_underdog': 0.98,
                'underdog': 0.95,
                'heavy_underdog': 0.90
            }
        }
        
        # Momentum factors
        momentum_factors = {
            'strong_form': 1.08,
            'good_form': 1.04,
            'neutral_form': 1.0,
            'poor_form': 0.96,
            'weak_form': 0.92
        }
    
    # Create segments based on seed difference, round, and momentum
    segments = []
    for i in range(len(predictions)):
        seed_diff = seed_diffs[i]
        win_rate_diff = win_rate_diffs[i] if win_rate_diffs is not None else 0
        
        # Determine segment based on seed difference
        if seed_diff <= -10:  # Heavy favorite by seed
            seed_segment = 'heavy_favorite'
        elif seed_diff <= -5:
            seed_segment = 'favorite'
        elif seed_diff <= -1:
            seed_segment = 'slight_favorite'
        elif seed_diff < 1:
            seed_segment = 'even'
        elif seed_diff < 5:
            seed_segment = 'slight_underdog'
        elif seed_diff < 10:
            seed_segment = 'underdog'
        else:
            seed_segment = 'heavy_underdog'
            
        # Add round information if available
        current_round = 'default'
        if tournament_round is not None:
            round_name = tournament_round[i]
            if round_name in round_factors:
                current_round = round_name
        
        # Add momentum dimension
        if win_rate_diff >= 0.2:
            momentum = 'strong_form'
        elif win_rate_diff >= 0.1:
            momentum = 'good_form'
        elif win_rate_diff <= -0.2:
            momentum = 'weak_form'
        elif win_rate_diff <= -0.1:
            momentum = 'poor_form'
        else:
            momentum = 'neutral_form'
            
        segments.append((seed_segment, current_round, momentum))
    
    # Apply calibration to each prediction
    for i in range(len(predictions)):
        seed_segment, current_round, momentum = segments[i]
        raw_pred = predictions[i]
        
        # Get calibration factors
        round_factor = round_factors[current_round][seed_segment] if current_round in round_factors else round_factors['default'][seed_segment]
        momentum_factor = momentum_factors[momentum]
        
        # Apply both factors
        adjusted_pred = raw_pred * round_factor * momentum_factor
        
        # Apply bounded constraints based on segment to prevent extreme values
        if seed_segment == 'heavy_favorite':
            # Don't go below 0.55 for heavy favorites
            calibrated_preds[i] = min(0.98, max(0.55, adjusted_pred))
        elif seed_segment == 'favorite':
            calibrated_preds[i] = min(0.95, max(0.50, adjusted_pred))
        elif seed_segment == 'slight_favorite':
            calibrated_preds[i] = min(0.90, max(0.45, adjusted_pred))
        elif seed_segment == 'even':
            # Keep closer to 0.5 for even matchups
            calibrated_preds[i] = min(0.65, max(0.35, adjusted_pred))
        elif seed_segment == 'slight_underdog':
            calibrated_preds[i] = min(0.55, max(0.10, adjusted_pred))
        elif seed_segment == 'underdog':
            calibrated_preds[i] = min(0.50, max(0.05, adjusted_pred))
        else:  # heavy underdog
            # Don't go above 0.45 for heavy underdogs
            calibrated_preds[i] = min(0.45, max(0.02, adjusted_pred))
    
    # Add confidence boosting to push predictions away from 0.5
    for i in range(len(calibrated_preds)):
        current_pred = calibrated_preds[i]
        
        # Only adjust predictions that aren't too extreme already
        if 0.4 < current_pred < 0.6:
            # Calculate distance from 0.5
            distance_from_center = abs(current_pred - 0.5)
            
            # Calculate boost factor - increases the further from 0.5
            boost_factor = distance_from_center * 2.0
            
            # Apply the boost (preserving direction)
            if current_pred > 0.5:
                calibrated_preds[i] = current_pred + (boost_factor * 0.05)
            else:
                calibrated_preds[i] = current_pred - (boost_factor * 0.05)

    # Add confidence boosting for predictions near the middle
    for i in range(len(calibrated_preds)):
        raw_pred = calibrated_preds[i]
        
        # For women's tournament - wider range since model is more accurate
        if 0.35 < raw_pred < 0.65:
            # Calculate distance from 0.5
            distance = abs(raw_pred - 0.5)
            
            # Stronger boost for women's tournament (higher overall accuracy)
            boost_magnitude = distance * 0.7
            
            # Apply the boost in the appropriate direction
            if raw_pred > 0.5:
                calibrated_preds[i] = raw_pred + boost_magnitude
            else:
                calibrated_preds[i] = raw_pred - boost_magnitude

    # Ensure all predictions remain in valid range
    calibrated_preds = np.clip(calibrated_preds, 0.001, 0.999)
    
    return calibrated_preds

def calibrate_mens_predictions(predictions, X_test, seed_diff_col='SeedDiff'):
    """
    Enhanced calibration specifically for men's tournament that addresses
    the issues identified in Final Four, Sweet 16, and upset detection
    
    Args:
        predictions: Raw prediction probabilities
        X_test: Test features DataFrame
        seed_diff_col: Column name for seed difference
        
    Returns:
        Calibrated predictions
    """   
    calibrated_preds = np.copy(predictions)
    
    # Extract important features for calibration
    seed_diffs = X_test[seed_diff_col].values if isinstance(X_test, pd.DataFrame) else X_test[:, seed_diff_col]
    
    # Get tournament round if available
    tournament_round = None
    if isinstance(X_test, pd.DataFrame) and 'ExpectedRound' in X_test.columns:
        tournament_round = X_test['ExpectedRound'].values
    
    # Extract defensive and shooting metrics if available
    def_metrics = {}
    shooting_metrics = {}
    
    for metric in ['Team1DefEfficiency', 'Team2DefEfficiency', 'DefEfficiencyDiff', 
                  'Team1FG3Pct', 'Team2FG3Pct', 'Team1WinRate', 'Team2WinRate',
                  'WinRateDiff', 'NetEfficiencyDiff']:
        if metric in X_test.columns:
            if 'Def' in metric:
                def_metrics[metric] = X_test[metric].values
            elif 'FG3' in metric:
                shooting_metrics[metric] = X_test[metric].values
            elif metric in ['Team1WinRate', 'Team2WinRate', 'WinRateDiff', 'NetEfficiencyDiff']:
                # Store these common metrics
                def_metrics[metric] = X_test[metric].values
    
    # Adjust round factors to be more upset-friendly
    round_factors = {
            'Round64': {
                'heavy_favorite': 1.25,  # Very strong confidence in top seeds
                'favorite': 1.20,
                'slight_favorite': 1.10,
                'even': 1.0,
                'slight_underdog': 0.80,
                'underdog': 0.70,
                'heavy_underdog': 0.60
            },
            'Round32': {
                'heavy_favorite': 1.25,  # Increased from 1.15
                'favorite': 1.20,        # Increased from 1.10
                'slight_favorite': 1.15, # Increased from 1.05
                'even': 1.0,
                'slight_underdog': 0.85, # Decreased from 0.95
                'underdog': 0.80,        # Decreased from 0.90
                'heavy_underdog': 0.75   # Decreased from 0.85
            },
            'Sweet16': {
                'heavy_favorite': 1.0,  # Neutral for Sweet 16
                'favorite': 1.0,
                'slight_favorite': 1.0,
                'even': 1.0,
                'slight_underdog': 1.0,
                'underdog': 1.0,
                'heavy_underdog': 1.0
            },
            'Elite8': {
                'heavy_favorite': 0.95,    # Less extreme than before
                'favorite': 0.97,          # Actually reduce confidence a bit
                'slight_favorite': 0.99,   # Almost neutral
                'even': 1.0,
                'slight_underdog': 1.01,   # Slight boost to underdogs
                'underdog': 1.03,          # Slightly favor underdogs
                'heavy_underdog': 1.05 
            },
            'Final4': {
                'heavy_favorite': 1.0,  # Neutral for Final Four
                'favorite': 1.0,
                'slight_favorite': 1.0,
                'even': 1.0, 
                'slight_underdog': 1.0,
                'underdog': 1.0,
                'heavy_underdog': 1.0
            },
            'Championship': {
                'heavy_favorite': 1.30,  # Very strong favorite bias for Championship
                'favorite': 1.25,
                'slight_favorite': 1.20,
                'even': 1.0,
                'slight_underdog': 0.80,
                'underdog': 0.75,
                'heavy_underdog': 0.70
            },
            'default': {
                'heavy_favorite': 1.15,
                'favorite': 1.10,
                'slight_favorite': 1.05,
                'even': 1.0,
                'slight_underdog': 0.95,
                'underdog': 0.90,
                'heavy_underdog': 0.85
            }
        }
    
    # Apply the calibration logic
    for i in range(len(calibrated_preds)):
        raw_pred = calibrated_preds[i]
        seed_diff = seed_diffs[i]
        
        # Determine seed segment
        if seed_diff <= -10:
            seed_segment = 'heavy_favorite'
        elif seed_diff <= -5:
            seed_segment = 'favorite'
        elif seed_diff <= -1:
            seed_segment = 'slight_favorite'
        elif seed_diff < 1:
            seed_segment = 'even'
        elif seed_diff < 5:
            seed_segment = 'slight_underdog'
        elif seed_diff < 10:
            seed_segment = 'underdog'
        else:
            seed_segment = 'heavy_underdog'
        
        # Determine round
        current_round = 'default'
        if tournament_round is not None:
            round_name = tournament_round[i]
            if round_name in round_factors:
                current_round = round_name
        
        # Get calibration factor
        base_factor = round_factors[current_round][seed_segment]
        
        # Additional adjustments based on other metrics
        defensive_factor = 1.0
        if 'Team1DefEfficiency' in def_metrics and 'Team2DefEfficiency' in def_metrics:
            team1_def = def_metrics['Team1DefEfficiency'][i]
            team2_def = def_metrics['Team2DefEfficiency'][i]
            
            # Lower defensive efficiency means better defense
            if team1_def < team2_def:
                def_advantage = min((team2_def - team1_def) / team2_def, 0.5)
                defensive_factor = 1.0 + (def_advantage * 0.3)
            else:
                def_advantage = min((team1_def - team2_def) / team1_def, 0.5)
                defensive_factor = 1.0 - (def_advantage * 0.3)
        
        # 3-point shooting factor
        shooting_factor = 1.0
        if 'Team1FG3Pct' in shooting_metrics and 'Team2FG3Pct' in shooting_metrics:
            team1_3pt = shooting_metrics['Team1FG3Pct'][i]
            team2_3pt = shooting_metrics['Team2FG3Pct'][i]
            
            # Higher is better for 3-point shooting
            if team1_3pt > team2_3pt:
                shooting_advantage = min((team1_3pt - team2_3pt) / team2_3pt, 0.5)
                shooting_factor = 1.0 + (shooting_advantage * 0.3)
            else:
                shooting_advantage = min((team2_3pt - team1_3pt) / team1_3pt, 0.5)
                shooting_factor = 1.0 - (shooting_advantage * 0.3)
        
        # Combine all factors
        combined_factor = base_factor * defensive_factor * shooting_factor
        
        # Apply the combined factor
        calibrated_preds[i] = raw_pred * combined_factor

        current_round = 'default'
        if tournament_round is not None:
            round_name = tournament_round[i]
            if round_name in round_factors:
                current_round = round_name
        
        # Make more decisive predictions for early rounds (where we have more games)
        if current_round in ['Round64', 'Round32']:
            # More aggressive threshold adjustment
            if calibrated_preds[i] > 0.53:  # If leaning toward Team1 winning
                calibrated_preds[i] = min(0.95, calibrated_preds[i] + 0.07)
            elif calibrated_preds[i] < 0.47:  # If leaning toward Team1 losing
                calibrated_preds[i] = max(0.05, calibrated_preds[i] - 0.07)
        
        # Ensure predictions stay within reasonable bounds
        if seed_segment == 'heavy_favorite':
            calibrated_preds[i] = min(0.95, max(0.6, calibrated_preds[i]))
        elif seed_segment == 'favorite':
            calibrated_preds[i] = min(0.90, max(0.5, calibrated_preds[i]))
        elif seed_segment == 'slight_favorite':
            calibrated_preds[i] = min(0.85, max(0.45, calibrated_preds[i]))
        elif seed_segment == 'even':
            calibrated_preds[i] = min(0.65, max(0.35, calibrated_preds[i]))
        elif seed_segment in ['slight_underdog', 'underdog', 'heavy_underdog']:
            calibrated_preds[i] = min(0.55, max(0.15, calibrated_preds[i]))
    
    # Add some additional confidence boosting
    for i in range(len(calibrated_preds)):
        current_pred = calibrated_preds[i]
        
        # Boost near-50/50 predictions
        if 0.4 < current_pred < 0.6:
            distance = abs(current_pred - 0.5)
            boost_factor = distance * 2.0
            
            # Boost slightly more for rounds known for upsets
            if tournament_round is not None:
                round_name = tournament_round[i]
                if round_name in ['Round64', 'Sweet16']:
                    boost_factor *= 1.5
            
            # Apply boost
            if current_pred > 0.5:
                calibrated_preds[i] = current_pred + (boost_factor * 0.07)
            else:
                calibrated_preds[i] = current_pred - (boost_factor * 0.07)
    
    # Ensure all predictions are within valid range
    calibrated_preds = np.clip(calibrated_preds, 0.001, 0.999)
    
    return calibrated_preds

def validate_model(model, validation_data, feature_cols, scaler, dropped_features, gender, validation_season):
    """
    Validate a trained model on a separate validation season

    Args:
        model: Trained model
        validation_data: Dictionary with validation data
        feature_cols: Feature columns used for training
        scaler: Trained scaler
        dropped_features: Features that were dropped during training
        gender: 'men's' or 'women's'
        validation_season: Season to use for validation

    Returns:
        Dictionary with validation metrics
    """
    print(f"\n=== Validating {gender} model on {validation_season} season ===")

    # Extract necessary dataframes
    df_tourney = validation_data.get('df_tourney', pd.DataFrame())
    df_seed = validation_data.get('df_seed', pd.DataFrame())
    # ... extract other needed dataframes

    # Generate features for validation season
    # IMPORTANT: Use the same feature engineering pipeline as in training
    # but apply it only to validation data

    # Process the validation season data
    # ... (feature engineering for validation season)

    # Create validation matchups
    validation_matchups = create_tournament_prediction_dataset(
        [validation_season],
        # ... same parameters as in training
    )

    # Add matchup IDs
    validation_matchups['MatchupID'] = validation_matchups.apply(
        lambda row: f"{row['Season']}_{min(row['Team1ID'], row['Team2ID'])}_{max(row['Team1ID'], row['Team2ID'])}",
        axis=1
    )

    # Match with actual tournament results
    actual_games = df_tourney[df_tourney['Season'] == validation_season].copy()
    actual_games['MatchupID'] = actual_games.apply(
        lambda row: f"{row['Season']}_{min(row['WTeamID'], row['LTeamID'])}_{max(row['WTeamID'], row['LTeamID'])}",
        axis=1
    )

    # Extract only matchups that actually happened
    valid_matchups = []
    for _, game in actual_games.iterrows():
        matchup_id = game['MatchupID']
        match_rows = validation_matchups[validation_matchups['MatchupID'] == matchup_id]

        for _, row in match_rows.iterrows():
            row_copy = row.copy()
            # Set the actual outcome as the target
            if row['Team1ID'] == game['WTeamID'] and row['Team2ID'] == game['LTeamID']:
                row_copy['Target'] = 1  # Team1 won
            elif row['Team1ID'] == game['LTeamID'] and row['Team2ID'] == game['WTeamID']:
                row_copy['Target'] = 0  # Team1 lost
            else:
                continue  # Should not happen

            valid_matchups.append(pd.DataFrame([row_copy]))

    if not valid_matchups:
        print(f"No matching games found for {validation_season}")
        return None

    validation_df = pd.concat(valid_matchups, ignore_index=True)

    # Prepare features for prediction
    X_val = validation_df[feature_cols].copy()
    y_val = validation_df['Target']

    # Apply the same transformations as during training
    X_val_scaled = scaler.transform(X_val)
    X_val_enhanced = create_feature_interactions(X_val)
    X_val_reduced = X_val_enhanced.drop(dropped_features, axis=1, errors='ignore')

    # Make predictions
    val_preds_proba = model.predict_proba(X_val_reduced)[:, 1]

    # Apply calibration if needed
    val_preds_calibrated = calibrate_by_expected_round(
        val_preds_proba,
        validation_df,
        seed_diff_col='SeedDiff'
    )

    # Calculate validation metrics
    val_log_loss = log_loss(y_val, val_preds_proba)
    val_calibrated_log_loss = log_loss(y_val, val_preds_calibrated)
    val_auc = roc_auc_score(y_val, val_preds_proba)
    val_brier = brier_score_loss(y_val, val_preds_proba)
    val_calibrated_brier = brier_score_loss(y_val, val_preds_calibrated)

    print(f"Validation Log Loss: {val_log_loss:.4f}")
    print(f"Validation Calibrated Log Loss: {val_calibrated_log_loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Validation Brier Score: {val_brier:.4f}")
    print(f"Validation Calibrated Brier Score: {val_calibrated_brier:.4f}")

    return {
        'log_loss': val_log_loss,
        'calibrated_log_loss': val_calibrated_log_loss,
        'auc': val_auc,
        'brier_score': val_brier,
        'calibrated_brier_score': val_calibrated_brier
    }

def evaluate_predictions_against_actual(predictions_df, actual_results_df, gender="men's"):
    """
    Evaluate the accuracy of predictions against actual tournament results

    Args:
        predictions_df: DataFrame with predicted probabilities
        actual_results_df: DataFrame with actual tournament results
        gender: 'men's' or 'women's' to identify which dataset

    Returns:
        Dictionary with evaluation metrics
    """
    # Create a copy of the actual results for processing
    actual_games = actual_results_df.copy()

    # Create a unique ID for each matchup in the actual results
    actual_games['MatchupID'] = actual_games.apply(
        lambda row: f"{row['Season']}_{min(row['WTeamID'], row['LTeamID'])}_{max(row['WTeamID'], row['LTeamID'])}",
        axis=1
    )

    # Create columns to track predicted vs actual outcomes
    matched_predictions = []

    # For each actual game, find and evaluate the prediction
    for _, game in actual_games.iterrows():
        season = game['Season']
        matchup_id = game['MatchupID']

        # Find this matchup in our predictions
        prediction = predictions_df[predictions_df['MatchupID'] == matchup_id]

        if len(prediction) > 0:
            # Get the prediction row
            pred_row = prediction.iloc[0]

            # Determine if team1 was the winner
            team1_id = pred_row['Team1ID']
            team2_id = pred_row['Team2ID']
            team1_won = (game['WTeamID'] == team1_id)

            # Get the predicted probability
            predicted_prob = pred_row['Pred'] if team1_won else (1 - pred_row['Pred'])

            matched_predictions.append({
                'Season': season,
                'MatchupID': matchup_id,
                'Team1ID': team1_id,
                'Team2ID': team2_id,
                'WinnerID': game['WTeamID'],
                'LoserID': game['LTeamID'],
                'PredictedProb': predicted_prob,
                'Correct': (predicted_prob > 0.5),
                'Confidence': abs(predicted_prob - 0.5) * 2  # Scale 0-1, where 1 is highest confidence
            })

    # Convert to DataFrame for analysis
    if matched_predictions:
        results_df = pd.DataFrame(matched_predictions)

        # Calculate evaluation metrics
        accuracy = results_df['Correct'].mean()
        brier_score = brier_score_loss(results_df['Correct'].astype(int), results_df['PredictedProb'])
        log_loss_score = log_loss(results_df['Correct'].astype(int), results_df['PredictedProb'])

        # Calculate accuracy by season
        accuracy_by_season = results_df.groupby('Season')['Correct'].mean()

        # Output results
        print(f"\n===== {gender.capitalize()} Tournament Prediction Evaluation =====")
        print(f"Matched {len(results_df)} games from actual tournament results")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Brier Score: {brier_score:.4f}")
        print(f"Log Loss: {log_loss_score:.4f}")
        print("\nAccuracy by Season:")
        for season, acc in accuracy_by_season.items():
            print(f"  {season}: {acc:.4f}")

        # Additional analyses
        print("\nHigh Confidence Predictions (>80% confidence):")
        high_conf = results_df[results_df['Confidence'] > 0.6]
        if len(high_conf) > 0:
            print(f"  Count: {len(high_conf)}")
            print(f"  Accuracy: {high_conf['Correct'].mean():.4f}")
        else:
            print("  No high confidence predictions found")

        return {
            'accuracy': accuracy,
            'brier_score': brier_score,
            'log_loss': log_loss_score,
            'matched_games': len(results_df),
            'accuracy_by_season': accuracy_by_season.to_dict(),
            'results_df': results_df
        }
    else:
        print(f"No matching predictions found for {gender} tournaments")
        return None

        # Add to march_madness/models/evaluation.py
def evaluate_predictions_by_tournament_round(predictions_df, actual_results_df, seed_data, gender="men's"):
    """
    Evaluate prediction accuracy by tournament round
    """
    # Create a copy of the actual results for processing
    actual_games = actual_results_df.copy()
    
    # Merge with seed data
    actual_games = actual_games.merge(
        seed_data.rename(columns={'TeamID': 'WTeamID', 'Seed': 'WSeed'}),
        on=['Season', 'WTeamID'],
        how='left'
    )
    
    actual_games = actual_games.merge(
        seed_data.rename(columns={'TeamID': 'LTeamID', 'Seed': 'LSeed'}),
        on=['Season', 'LTeamID'],
        how='left'
    )
    
    # Create a unique ID for each matchup in the actual results
    actual_games['MatchupID'] = actual_games.apply(
        lambda row: f"{row['Season']}_{min(row['WTeamID'], row['LTeamID'])}_{max(row['WTeamID'], row['LTeamID'])}",
        axis=1
    )
    
    # Determine the round for each game
    round_mapping = {
        134: 'Round64', 135: 'Round64', 136: 'Round64', 137: 'Round64',
        138: 'Round32', 139: 'Round32',
        140: 'Sweet16', 141: 'Sweet16',
        142: 'Elite8', 143: 'Elite8',
        144: 'Final4',
        146: 'Championship'
    }
    
    actual_games['Round'] = actual_games['DayNum'].map(round_mapping)
    
    # Match predictions with actual games and track by round
    matched_predictions = []
    
    for _, game in actual_games.iterrows():
        season = game['Season']
        matchup_id = game['MatchupID']
        
        # Find this matchup in our predictions
        prediction = predictions_df[predictions_df['MatchupID'] == matchup_id]
        
        if len(prediction) > 0:
            # Get the prediction row
            pred_row = prediction.iloc[0]
            
            # Determine if team1 was the winner
            team1_id = pred_row['Team1ID']
            team2_id = pred_row['Team2ID']
            team1_won = (game['WTeamID'] == team1_id)
            
            # Get the predicted probability
            predicted_prob = pred_row['Pred'] if team1_won else (1 - pred_row['Pred'])
            
            matched_predictions.append({
                'Season': season,
                'MatchupID': matchup_id,
                'Round': game['Round'],
                'WTeamID': game['WTeamID'],
                'LTeamID': game['LTeamID'],
                'WSeed': game['WSeed'],
                'LSeed': game['LSeed'],
                'PredictedProb': predicted_prob,
                'Correct': (predicted_prob > 0.5),
                'Confidence': abs(predicted_prob - 0.5) * 2  # Scale 0-1, where 1 is highest confidence
            })
    
    # Convert to DataFrame and analyze by round
    if matched_predictions:
        results_df = pd.DataFrame(matched_predictions)
        
        # Calculate Brier score and accuracy by round
        round_metrics = results_df.groupby('Round').apply(lambda x: pd.Series({
            'Count': len(x),
            'Accuracy': x['Correct'].mean(),
            'Brier': brier_score_loss(x['Correct'].astype(int), x['PredictedProb']),
            'LogLoss': log_loss(x['Correct'].astype(int), x['PredictedProb']),
            'AvgConfidence': x['Confidence'].mean()
        })).reset_index()
        
        print(f"\n===== {gender.capitalize()} Tournament Prediction Evaluation by Round =====")
        print(f"Matched {len(results_df)} games from actual tournament results")
        print("\nMetrics by Round:")
        print(round_metrics.to_string(index=False))
        
        # Analyze upset metrics
        results_df['WSeedNum'] = results_df['WSeed'].apply(lambda x: int(x[1:3]) if isinstance(x, str) and len(x) >= 3 else 0)
        results_df['LSeedNum'] = results_df['LSeed'].apply(lambda x: int(x[1:3]) if isinstance(x, str) and len(x) >= 3 else 0) 
        results_df['Upset'] = results_df['WSeedNum'] > results_df['LSeedNum']
        
        # Accuracy on upsets vs. non-upsets
        upset_games = results_df[results_df['Upset']]
        non_upset_games = results_df[~results_df['Upset']]
        
        print("\nUpset Detection:")
        print(f"  Total Upsets: {len(upset_games)} ({len(upset_games)/len(results_df)*100:.1f}%)")
        if len(upset_games) > 0:
            print(f"  Accuracy on Upsets: {upset_games['Correct'].mean():.4f}")
        print(f"  Accuracy on Non-Upsets: {non_upset_games['Correct'].mean():.4f}")
        
        return {
            'round_metrics': round_metrics,
            'results_df': results_df
        }
    else:
        print(f"No matching predictions found for {gender} tournaments")
        return None

def evaluate_predictions_by_tournament_round(predictions_df, actual_results_df, seed_data, gender="men's"):
    """
    Evaluate prediction accuracy by tournament round
    """
    # Create a copy of the actual results for processing
    actual_games = actual_results_df.copy()
    
    # Merge with seed data
    actual_games = actual_games.merge(
        seed_data.rename(columns={'TeamID': 'WTeamID', 'Seed': 'WSeed'}),
        on=['Season', 'WTeamID'],
        how='left'
    )
    
    actual_games = actual_games.merge(
        seed_data.rename(columns={'TeamID': 'LTeamID', 'Seed': 'LSeed'}),
        on=['Season', 'LTeamID'],
        how='left'
    )
    
    # Create a unique ID for each matchup in the actual results
    actual_games['MatchupID'] = actual_games.apply(
        lambda row: f"{row['Season']}_{min(row['WTeamID'], row['LTeamID'])}_{max(row['WTeamID'], row['LTeamID'])}",
        axis=1
    )
    
    # Determine the round for each game
    round_mapping = {
        134: 'Round64', 135: 'Round64', 136: 'Round64', 137: 'Round64',
        138: 'Round32', 139: 'Round32',
        140: 'Sweet16', 141: 'Sweet16',
        142: 'Elite8', 143: 'Elite8',
        144: 'Final4',
        146: 'Championship'
    }
    
    actual_games['Round'] = actual_games['DayNum'].map(round_mapping)
    
    # Match predictions with actual games and track by round
    matched_predictions = []
    
    for _, game in actual_games.iterrows():
        season = game['Season']
        matchup_id = game['MatchupID']
        
        # Find this matchup in our predictions
        prediction = predictions_df[predictions_df['MatchupID'] == matchup_id]
        
        if len(prediction) > 0:
            # Get the prediction row
            pred_row = prediction.iloc[0]
            
            # Determine if team1 was the winner
            team1_id = pred_row['Team1ID']
            team2_id = pred_row['Team2ID']
            team1_won = (game['WTeamID'] == team1_id)
            
            # Get the predicted probability
            predicted_prob = pred_row['Pred'] if team1_won else (1 - pred_row['Pred'])
            
            matched_predictions.append({
                'Season': season,
                'MatchupID': matchup_id,
                'Round': game['Round'],
                'WTeamID': game['WTeamID'],
                'LTeamID': game['LTeamID'],
                'WSeed': game['WSeed'],
                'LSeed': game['LSeed'],
                'PredictedProb': predicted_prob,
                'Correct': (predicted_prob > 0.5),
                'Confidence': abs(predicted_prob - 0.5) * 2  # Scale 0-1, where 1 is highest confidence
            })
    
    # Convert to DataFrame and analyze by round
    if matched_predictions:
        results_df = pd.DataFrame(matched_predictions)
        
        # Initialize an empty DataFrame for round metrics
        round_metrics_list = []
        
        # Calculate metrics by round manually instead of using groupby with lambda
        for round_name, round_group in results_df.groupby('Round'):
            correct = round_group['Correct'].astype(int)
            predicted = round_group['PredictedProb']
            
            # Calculate Brier score
            brier = brier_score_loss(correct, predicted)
            
            # Calculate Log Loss safely with error handling
            try:
                logloss = log_loss(correct, predicted)
            except ValueError as e:
                # If all predictions are the same class
                if "y_true contains only one label" in str(e):
                    # If all are correct (all 1's), log_loss should be close to 0
                    # If all are incorrect (all 0's), log_loss should be a high value
                    if correct.mean() == 1.0:  # All correct
                        logloss = 0.001  # Near perfect score
                    else:  # All incorrect
                        logloss = 15.0  # Very bad score
                else:
                    # For other errors, just use a default value
                    logloss = float('nan')
            
            round_metrics_list.append({
                'Round': round_name,
                'Count': len(round_group),
                'Accuracy': round_group['Correct'].mean(),
                'Brier': brier,
                'LogLoss': logloss,
                'AvgConfidence': round_group['Confidence'].mean()
            })
        
        round_metrics = pd.DataFrame(round_metrics_list)
        
        print(f"\n===== {gender.capitalize()} Tournament Prediction Evaluation by Round =====")
        print(f"Matched {len(results_df)} games from actual tournament results")
        print("\nMetrics by Round:")
        print(round_metrics.to_string(index=False))
        
        # Analyze upset metrics
        results_df['WSeedNum'] = results_df['WSeed'].apply(lambda x: int(x[1:3]) if isinstance(x, str) and len(x) >= 3 else 0)
        results_df['LSeedNum'] = results_df['LSeed'].apply(lambda x: int(x[1:3]) if isinstance(x, str) and len(x) >= 3 else 0) 
        results_df['Upset'] = results_df['WSeedNum'] > results_df['LSeedNum']
        
        # Accuracy on upsets vs. non-upsets
        upset_games = results_df[results_df['Upset']]
        non_upset_games = results_df[~results_df['Upset']]
        
        print("\nUpset Detection:")
        print(f"  Total Upsets: {len(upset_games)} ({len(upset_games)/len(results_df)*100:.1f}%)")
        if len(upset_games) > 0:
            print(f"  Accuracy on Upsets: {upset_games['Correct'].mean():.4f}")
        print(f"  Accuracy on Non-Upsets: {non_upset_games['Correct'].mean():.4f}")
        
        return {
            'round_metrics': round_metrics,
            'results_df': results_df
        }
    else:
        print(f"No matching predictions found for {gender} tournaments")
        return None

# Replace all the specialized functions with this simplified approach
def optimal_calibration_for_mens(predictions, X_test, seed_diff_col='SeedDiff'):
    """Optimized calibration that balances accuracy and upset detection"""
    calibrated_preds = np.copy(predictions)
    
    # Get seed differences and round information
    seed_diffs = X_test[seed_diff_col].values if isinstance(X_test, pd.DataFrame) else X_test[:, seed_diff_col]
    tournament_round = None
    if isinstance(X_test, pd.DataFrame) and 'ExpectedRound' in X_test.columns:
        tournament_round = X_test['ExpectedRound'].values
    
    # Apply calibration with different factors for each round
    for i in range(len(calibrated_preds)):
        # Determine seed segment
        seed_diff = seed_diffs[i]
        if seed_diff <= -10:
            seed_segment = 'heavy_favorite'
        elif seed_diff <= -5:
            seed_segment = 'favorite'
        elif seed_diff <= -1:
            seed_segment = 'slight_favorite'
        elif seed_diff < 1:
            seed_segment = 'even'
        elif seed_diff < 5:
            seed_segment = 'slight_underdog'
        elif seed_diff < 10:
            seed_segment = 'underdog'
        else:
            seed_segment = 'heavy_underdog'
        
        # Determine round
        current_round = 'default'
        if tournament_round is not None:
            round_name = tournament_round[i]
            if round_name in ['Round64', 'Round32', 'Sweet16', 'Elite8', 'Final4', 'Championship']:
                current_round = round_name
        
        # Get calibration factor based on round and seed segment
        if current_round == 'Round64':
            if seed_segment in ['heavy_favorite', 'favorite']:
                # Strong boost for favorites in first round
                calibrated_preds[i] = min(0.95, calibrated_preds[i] * 1.15)
            elif seed_segment in ['heavy_underdog', 'underdog']:
                # Reduce confidence in first round underdogs
                calibrated_preds[i] = max(0.10, calibrated_preds[i] * 0.85)
        
        elif current_round == 'Round32':
            # More balanced approach for Round32 (where we had issues)
            if calibrated_preds[i] > 0.65:
                # Cap high confidence predictions
                calibrated_preds[i] = 0.65 + ((calibrated_preds[i] - 0.65) * 0.5)
            elif calibrated_preds[i] < 0.35:
                # Cap low confidence predictions
                calibrated_preds[i] = 0.35 - ((0.35 - calibrated_preds[i]) * 0.5)
        
        elif current_round in ['Elite8', 'Final4', 'Championship']:
            # Later rounds have more upsets - be more balanced
            if calibrated_preds[i] > 0.5:
                # Reduce favorite confidence
                calibrated_preds[i] = 0.5 + ((calibrated_preds[i] - 0.5) * 0.8)
            else:
                # Increase underdog chances
                calibrated_preds[i] = 0.5 - ((0.5 - calibrated_preds[i]) * 0.8)
    
    # Special upset handling based on classic matchups
    if isinstance(X_test, pd.DataFrame) and 'Team1Seed' in X_test.columns and 'Team2Seed' in X_test.columns:
        for i in range(len(calibrated_preds)):
            team1_seed = X_test.iloc[i]['Team1Seed']
            team2_seed = X_test.iloc[i]['Team2Seed']
            
            # Classic upset matchups
            classic_upset = ((team1_seed == 12 and team2_seed == 5) or
                            (team1_seed == 11 and team2_seed == 6) or
                            (team1_seed == 10 and team2_seed == 7) or
                            (team1_seed == 9 and team2_seed == 8))
            
            # Adjust for classic upset matchups
            if classic_upset:
                if calibrated_preds[i] > 0.65:
                    # Cap high confidence against classic upset
                    calibrated_preds[i] = 0.65
                elif 0.45 < calibrated_preds[i] < 0.55:
                    # Give slight edge to upset potential
                    calibrated_preds[i] = 0.45
    
    # Final mild push away from 0.5 for decisive predictions
    for i in range(len(calibrated_preds)):
        if calibrated_preds[i] > 0.55:  # Clear favor to Team1
            calibrated_preds[i] = min(0.95, calibrated_preds[i] + 0.05)
        elif calibrated_preds[i] < 0.45:  # Clear favor to Team2
            calibrated_preds[i] = max(0.05, calibrated_preds[i] - 0.05)
    
    # Ensure all predictions remain in valid range
    return np.clip(calibrated_preds, 0.001, 0.999)
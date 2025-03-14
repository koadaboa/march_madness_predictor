# Model evaluation functions
import pandas as pd
import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from ..features.matchup import create_tournament_prediction_dataset
from ..models.training import create_feature_interactions


def calibrate_by_expected_round(predictions, X_test, seed_diff_col='SeedDiff'):
    calibrated_preds = np.copy(predictions)
    seed_diffs = X_test[seed_diff_col].values if isinstance(X_test, pd.DataFrame) else X_test[:, seed_diff_col]
    
    # Get team seed information if available
    team1_seeds = X_test['Team1Seed'].values if 'Team1Seed' in X_test.columns else None
    team2_seeds = X_test['Team2Seed'].values if 'Team2Seed' in X_test.columns else None
    
    # Use both seed differences and win rate differences for calibration
    win_rate_diffs = X_test['WinRateDiff'].values if 'WinRateDiff' in X_test.columns else None
    
    # Create segments based on both seed and win rate differences
    segments = []
    for i in range(len(predictions)):
        seed_diff = seed_diffs[i]
        win_rate_diff = win_rate_diffs[i] if win_rate_diffs is not None else 0
        
        # Determine segment - more fine-grained than before
        if seed_diff <= -10:  # Heavy favorite by seed
            segment = 'heavy_favorite_seed'
        elif seed_diff <= -5:
            segment = 'favorite_seed'
        elif seed_diff <= -1:
            segment = 'slight_favorite_seed'
        elif seed_diff < 1:
            segment = 'even_seed'
        elif seed_diff < 5:
            segment = 'slight_underdog_seed'
        elif seed_diff < 10:
            segment = 'underdog_seed'
        else:
            segment = 'heavy_underdog_seed'
            
        # Add win rate dimension
        if win_rate_diff >= 0.2:
            segment += '_strong_form'
        elif win_rate_diff >= 0.1:
            segment += '_good_form'
        elif win_rate_diff <= -0.2:
            segment += '_weak_form'
        elif win_rate_diff <= -0.1:
            segment += '_poor_form'
        else:
            segment += '_neutral_form'
            
        segments.append(segment)
    
    # Apply calibration based on segments
    segment_calibrators = {}
    
    # If we have target values, train segment-specific calibrators
    if 'Target' in X_test.columns and not X_test['Target'].isna().any():
        targets = X_test['Target'].values
        unique_segments = set(segments)
        
        for segment in unique_segments:
            mask = np.array(segments) == segment
            if np.sum(mask) >= 10:  # Need enough samples
                segment_probs = predictions[mask]
                segment_targets = targets[mask]
                
                # Try logistic regression calibration first
                try:
                    lr_calibrator = LogisticRegression(C=1.0, solver='liblinear')
                    lr_calibrator.fit(segment_probs.reshape(-1, 1), segment_targets)
                    segment_calibrators[segment] = ('lr', lr_calibrator)
                except:
                    # Fall back to isotonic if logistic fails
                    try:
                        iso_calibrator = IsotonicRegression(out_of_bounds='clip')
                        iso_calibrator.fit(segment_probs, segment_targets)
                        segment_calibrators[segment] = ('iso', iso_calibrator)
                    except:
                        pass
    
    # Apply calibration to each prediction
    for i in range(len(predictions)):
        segment = segments[i]
        if segment in segment_calibrators:
            calib_type, calibrator = segment_calibrators[segment]
            if calib_type == 'lr':
                calibrated_preds[i] = calibrator.predict_proba(np.array([[predictions[i]]]))[0, 1]
            else:
                calibrated_preds[i] = calibrator.transform([predictions[i]])[0]
        else:
            # Apply default calibration based on segment
            raw_pred = predictions[i]
            if 'heavy_favorite' in segment:
                if 'strong_form' in segment:
                    calibrated_preds[i] = min(0.98, raw_pred * 1.05)
                else:
                    calibrated_preds[i] = min(0.95, raw_pred * 1.03)
            elif 'favorite' in segment:
                calibrated_preds[i] = min(0.92, raw_pred * 1.02)
            elif 'slight_favorite' in segment:
                calibrated_preds[i] = min(0.85, raw_pred * 1.01)
            elif 'even' in segment:
                # Keep closer to 0.5 for even matchups
                calibrated_preds[i] = raw_pred * 0.95 + 0.025
            elif 'slight_underdog' in segment:
                calibrated_preds[i] = max(0.15, raw_pred * 0.98)
            elif 'underdog' in segment:
                calibrated_preds[i] = max(0.08, raw_pred * 0.97)
            else:  # heavy underdog
                calibrated_preds[i] = max(0.02, raw_pred * 0.95)
    
    # Ensure all predictions are valid probabilities
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
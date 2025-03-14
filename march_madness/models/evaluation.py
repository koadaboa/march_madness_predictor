# Model evaluation functions
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

def calibrate_by_expected_round(predictions, X_test, seed_diff_col='SeedDiff'):
    """
    Calibrate model predictions by expected tournament round based on seed matchups

    Args:
        predictions: Model predictions to calibrate
        X_test: Test features
        seed_diff_col: Column name for seed difference

    Returns:
        Calibrated predictions
    """
    # Initialize with original predictions
    calibrated_preds = np.copy(predictions)

    # Get seed differences
    if isinstance(X_test, pd.DataFrame):
        seed_diffs = X_test[seed_diff_col].values
        team1_seeds = X_test['Team1Seed'].values if 'Team1Seed' in X_test.columns else None
        team2_seeds = X_test['Team2Seed'].values if 'Team2Seed' in X_test.columns else None
    else:
        # Assume it's the index of the seed difference column
        seed_diffs = X_test[:, seed_diff_col]
        team1_seeds = None
        team2_seeds = None

    # Assign expected rounds if we have seed information
    if team1_seeds is not None and team2_seeds is not None:
        expected_rounds = [determine_expected_round(s1, s2) for s1, s2 in zip(team1_seeds, team2_seeds)]
    else:
        # If we don't have seed info, just use seed differences
        expected_rounds = ['Unknown'] * len(seed_diffs)

    seed_groups = [
        ('heavy_favorite', lambda x: x <= -8),
        ('favorite', lambda x: (x > -8) & (x <= -4)),
        ('slight_favorite', lambda x: (x > -4) & (x <= -1)),
        ('even', lambda x: (x > -1) & (x < 1)),
        ('slight_underdog', lambda x: (x >= 1) & (x < 4)),
        ('underdog', lambda x: (x >= 4) & (x < 8)),
        ('heavy_underdog', lambda x: x >= 8)
    ]

    # Calibrate by expected round and seed group
    for round_name in set(expected_rounds):
        if round_name == 'Unknown':
            continue

        round_mask = np.array(expected_rounds) == round_name

        # For each seed group in this expected round
        for group_name, group_filter in seed_groups:
            group_mask = round_mask & group_filter(seed_diffs)

            # If not enough samples in this group, skip
            if sum(group_mask) < 10:
                continue

            # Get predictions for this group
            group_preds = predictions[group_mask]

            # If we have true values, we can fit a calibrator
            if 'Target' in X_test.columns and not X_test['Target'].isna().any():
                group_targets = X_test.loc[group_mask, 'Target'].values

                # Train isotonic calibration
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(group_preds, group_targets)

                # Apply calibration
                calibrated_preds[group_mask] = calibrator.transform(group_preds)
            else:
                # Apply pre-defined calibration based on historical patterns
                if round_name == 'Round64':
                    if group_name == 'heavy_favorite':
                        # 1 vs 16, 2 vs 15 matchups - boost higher seeds
                        calibrated_preds[group_mask] = 0.95 * group_preds + 0.05
                    elif group_name == 'favorite':
                        # 3 vs 14, 4 vs 13 matchups
                        calibrated_preds[group_mask] = 0.9 * group_preds + 0.05
                    elif group_name == 'slight_favorite':
                        # 5 vs 12, 6 vs 11 matchups - more upset prone
                        calibrated_preds[group_mask] = 0.85 * group_preds + 0.05
                    elif group_name == 'even':
                        # 7 vs 10, 8 vs 9 matchups - very even
                        calibrated_preds[group_mask] = 0.7 * group_preds + 0.15

                elif round_name == 'Round32':
                    if group_name == 'heavy_favorite':
                        # Still favor higher seeds but less dramatically
                        calibrated_preds[group_mask] = 0.9 * group_preds + 0.05
                    elif group_name in ['favorite', 'slight_favorite']:
                        # More competitive matchups
                        calibrated_preds[group_mask] = 0.85 * group_preds + 0.05
                    elif group_name == 'even':
                        # Very competitive
                        calibrated_preds[group_mask] = 0.7 * group_preds + 0.15

                elif round_name in ['Sweet16', 'Elite8']:
                    # Later rounds are more competitive and skill-based
                    if group_name in ['heavy_favorite', 'favorite']:
                        calibrated_preds[group_mask] = 0.85 * group_preds + 0.05
                    else:
                        # Closer matchups
                        calibrated_preds[group_mask] = 0.8 * group_preds + 0.1

                elif round_name in ['Final4', 'Championship']:
                    # Final rounds are very competitive regardless of seed
                    calibrated_preds[group_mask] = 0.7 * group_preds + 0.15

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
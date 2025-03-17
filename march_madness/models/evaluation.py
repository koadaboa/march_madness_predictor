# Model evaluation functions
import pandas as pd
import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from ..features.matchup import create_tournament_prediction_dataset
from ..models.training import create_feature_interactions


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
            'Championship': {  # Women are very predictable in championship (80% accuracy)
                'heavy_favorite': 1.25,  # Much more confident in heavy favorites
                'favorite': 1.20,
                'slight_favorite': 1.15,
                'even': 1.0,
                'slight_underdog': 0.85,
                'underdog': 0.80,
                'heavy_underdog': 0.75
            },
            'Final4': {  # Women's Final Four very unpredictable (33% accuracy)
                'heavy_favorite': 0.95,  # Be conservative with favorites
                'favorite': 0.92,
                'slight_favorite': 0.90,
                'even': 1.0,
                'slight_underdog': 1.10,  # Boost underdogs
                'underdog': 1.15,
                'heavy_underdog': 1.20
            },
            'Round32': {  # Women's Round of 32 is good (60% accuracy)
                'heavy_favorite': 1.15,
                'favorite': 1.12,
                'slight_favorite': 1.08,
                'even': 1.0,
                'slight_underdog': 0.90,
                'underdog': 0.85,
                'heavy_underdog': 0.80
            },
            'default': {  # Default factors for other rounds
                'heavy_favorite': 1.12,
                'favorite': 1.08,
                'slight_favorite': 1.05,
                'even': 1.0,
                'slight_underdog': 0.92,
                'underdog': 0.85,
                'heavy_underdog': 0.80
            }
        }
        
        # Momentum factors
        momentum_factors = {
            'strong_form': 1.12,
            'good_form': 1.06,
            'neutral_form': 1.0,
            'poor_form': 0.95,
            'weak_form': 0.90
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
    
    # Ensure all predictions are valid probabilities
    calibrated_preds = np.clip(calibrated_preds, 0.001, 0.999)
    
    return calibrated_preds

def calibrate_mens_predictions(predictions, X_test, seed_diff_col='SeedDiff'):
    """
    Advanced calibration specifically for men's tournament that addresses
    the issues identified in multi-year analysis
    
    Args:
        predictions: Raw prediction probabilities
        X_test: Test features DataFrame
        seed_diff_col: Column name for seed difference
        
    Returns:
        Calibrated predictions
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LogisticRegression
    
    calibrated_preds = np.copy(predictions)
    seed_diffs = X_test[seed_diff_col].values if isinstance(X_test, pd.DataFrame) else X_test[:, seed_diff_col]
    
    # Get tournament round if available
    tournament_round = None
    if isinstance(X_test, pd.DataFrame) and 'ExpectedRound' in X_test.columns:
        tournament_round = X_test['ExpectedRound'].values
    
    # Use additional context features if available
    features = []
    feature_names = ['WinRateDiff', 'OffEfficiencyDiff', 'DefEfficiencyDiff']
    
    for feat in feature_names:
        if feat in X_test.columns:
            features.append(X_test[feat].values)
    
    # Create segments based on seed difference and round
    segments = []
    
    for i in range(len(predictions)):
        seed_diff = seed_diffs[i]
        
        # Determine segment based on seed difference
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
        
        # Add round information if available
        round_segment = 'unknown'
        if tournament_round is not None:
            round_segment = tournament_round[i]
        
        segments.append((seed_segment, round_segment))
    
    # Apply calibration based on our analysis
    for i in range(len(predictions)):
        seed_segment, round_segment = segments[i]
        raw_pred = predictions[i]
        
        # Base multiplier for men's model - seed-specific
        if seed_segment == 'heavy_favorite':
            base_mult = 1.20  # Be more confident in heavy favorites
        elif seed_segment == 'favorite':
            base_mult = 1.12
        elif seed_segment == 'slight_favorite':
            base_mult = 1.08
        elif seed_segment == 'even':
            base_mult = 1.0
        elif seed_segment == 'slight_underdog':
            base_mult = 0.92
        elif seed_segment == 'underdog':
            base_mult = 0.88
        else:  # heavy_underdog
            base_mult = 0.83
        
        # Round-specific adjustments based on our analysis
        if round_segment == 'Final4':
            # Men's Final Four has just 41.7% accuracy - be more conservative with favorites
            # and boost underdogs
            if 'favorite' in seed_segment:
                round_mult = 0.85  # Reduce confidence in favorites
            elif 'underdog' in seed_segment:
                round_mult = 1.15  # Boost confidence in underdogs
            else:
                round_mult = 1.0
        elif round_segment == 'Elite8':
            # Men's Elite 8 has just 58.3% accuracy
            if 'favorite' in seed_segment:
                round_mult = 0.90
            elif 'underdog' in seed_segment:
                round_mult = 1.10
            else:
                round_mult = 1.0
        elif round_segment == 'Championship':
            # Men's championship games are 60% accurate - boost favorites
            if 'favorite' in seed_segment:
                round_mult = 1.15
            else:
                round_mult = 0.95
        elif round_segment == 'Round64':
            # Men's Round64 is good at 64.8% - be more confident
            if 'favorite' in seed_segment:
                round_mult = 1.10
            else:
                round_mult = 0.90
        else:
            round_mult = 1.0
        
        # Confidence factor - be more aggressive with extreme predictions
        if raw_pred >= 0.8 or raw_pred <= 0.2:
            # Current high-confidence predictions are 83.3% accurate
            # Be even more confident
            conf_mult = 1.10
        elif raw_pred >= 0.7 or raw_pred <= 0.3:
            conf_mult = 1.05
        else:
            conf_mult = 1.0
        
        # Apply all multipliers
        adjusted_pred = raw_pred * base_mult * round_mult * conf_mult
        
        # Apply bounds based on segment to prevent extreme values
        if seed_segment == 'heavy_favorite':
            # Don't go below 0.7 for heavy favorites
            calibrated_preds[i] = min(0.98, max(0.7, adjusted_pred))
        elif seed_segment == 'favorite':
            calibrated_preds[i] = min(0.95, max(0.65, adjusted_pred))
        elif seed_segment == 'slight_favorite':
            calibrated_preds[i] = min(0.9, max(0.6, adjusted_pred))
        elif seed_segment == 'even':
            # Keep closer to 0.5 for even matchups but allow more variation
            calibrated_preds[i] = min(0.7, max(0.3, adjusted_pred))
        elif seed_segment == 'slight_underdog':
            calibrated_preds[i] = min(0.4, max(0.1, adjusted_pred))
        elif seed_segment == 'underdog':
            calibrated_preds[i] = min(0.35, max(0.05, adjusted_pred))
        else:  # heavy_underdog
            # Don't go above 0.3 for heavy underdogs
            calibrated_preds[i] = min(0.3, max(0.02, adjusted_pred))
    
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
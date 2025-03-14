# Model training functions
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def prepare_tournament_data_for_training(tourney_results, features_df, test_seasons=None):
    """
    Prepare tournament data for model training and validation.
    Ensures proper handling of train/test split to prevent data leakage.

    Args:
        tourney_results: DataFrame with tournament results
        features_df: DataFrame with matchup features
        test_seasons: List of seasons to use for testing

    Returns:
        Training and testing data splits
    """
    print("\nPreparing tournament data...")
    features_df_copy = features_df.copy()
    train_data = []
    found_matches = 0

    # Define features to use (exclude identifiers and target)
    feature_cols = [col for col in features_df_copy.columns if col not in
                   ['Season', 'Team1ID', 'Team2ID', 'Target', 'ExpectedRound', 'MatchupID']]

    if test_seasons:
        if not isinstance(test_seasons, list):
            test_seasons = [test_seasons]

        train_seasons = [season for season in features_df_copy['Season'].unique() if season not in test_seasons]
        print(f"Training seasons: {train_seasons}")
        print(f"Testing seasons: {test_seasons}")

        # For training data, match with actual tournament results
        for season in train_seasons:
            season_features = get_data_with_index(features_df_copy, 'Season', season, indexed_suffix='_by_season').copy()
            season_results = get_data_with_index(tourney_results, 'Season', season, indexed_suffix='_by_season')
            season_matches = 0

            for _, result in season_results.iterrows():
                # Find the matchup in our features dataset where Team1 won
                matchup1 = season_features[
                    (season_features['Team1ID'] == result['WTeamID']) &
                    (season_features['Team2ID'] == result['LTeamID'])
                ].copy()

                # Find the matchup in our features dataset where Team1 lost
                matchup2 = season_features[
                    (season_features['Team1ID'] == result['LTeamID']) &
                    (season_features['Team2ID'] == result['WTeamID'])
                ].copy()

                if len(matchup1) > 0:
                    matchup1['Target'] = 1  # Team1 won
                    train_data.append(matchup1)
                    season_matches += 1
                    found_matches += 1

                if len(matchup2) > 0:
                    matchup2['Target'] = 0  # Team1 lost
                    train_data.append(matchup2)
                    season_matches += 1
                    found_matches += 1

            print(f"Season {season}: Found {season_matches} matching feature records")

        print(f"Total matches found: {found_matches}")
        train_df = pd.concat(train_data, ignore_index=True)
        print(f"Training data shape: {train_df.shape}")

        # For validation, extract only the actual games that happened
        test_with_results = []
        test_results = tourney_results[tourney_results['Season'].isin(test_seasons)]

        for _, result in test_results.iterrows():
            # Find the matchup in our test dataset
            result_matchup_id = f"{result['Season']}_{min(result['WTeamID'], result['LTeamID'])}_{max(result['WTeamID'], result['LTeamID'])}"

            # Get all potential matchups that match this actual game
            test_features = features_df_copy[features_df_copy['Season'].isin(test_seasons)]
            test_features = test_features[test_features['MatchupID'] == result_matchup_id].copy()

            # Set target based on who actually won
            for _, row in test_features.iterrows():
                if row['Team1ID'] == result['WTeamID'] and row['Team2ID'] == result['LTeamID']:
                    row = row.copy()
                    row['Target'] = 1  # Team1 won
                    test_with_results.append(pd.DataFrame([row]))
                elif row['Team1ID'] == result['LTeamID'] and row['Team2ID'] == result['WTeamID']:
                    row = row.copy()
                    row['Target'] = 0  # Team1 lost
                    test_with_results.append(pd.DataFrame([row]))

        if test_with_results:
            test_with_results_df = pd.concat(test_with_results, ignore_index=True)
            print(f"Found {len(test_with_results_df)} actual matchups in test seasons for validation")

            X_train = train_df[feature_cols]
            y_train = train_df['Target']
            X_test = test_with_results_df[feature_cols]
            y_test = test_with_results_df['Target']
        else:
            print("No actual matchups found for test seasons")
            X_train = train_df[feature_cols]
            y_train = train_df['Target']
            X_test = pd.DataFrame(columns=feature_cols)
            y_test = pd.Series([])
    else:
        # Use random split if no test_seasons specified
        X = features_df_copy[feature_cols]
        y = features_df_copy['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if len(X_test) > 0:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = np.empty((0, len(feature_cols)))

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test, scaler, feature_cols


def create_feature_interactions(X, important_features=None, max_degree=2):
    """
    Create meaningful feature interactions

    Args:
        X: DataFrame with features
        important_features: List of most important feature names
        max_degree: Maximum degree for interactions

    Returns:
        DataFrame with original features and interactions
    """
    # Start with the original data
    X_enhanced = X.copy()

    # If no important features provided, use all features
    if important_features is None:
        important_features = X.columns.tolist()

    # Select top features for interactions (to avoid explosion of features)
    if len(important_features) > 30:
        interaction_features = important_features[:30]
    else:
        interaction_features = important_features

    # Create interaction terms for pairs of important features
    for i, feat1 in enumerate(interaction_features):
        if feat1 not in X.columns:
            continue

        # 1. Create squared terms for certain features (non-binary features)
        if X[feat1].nunique() > 2:
            feat_name = f"{feat1}_squared"
            X_enhanced[feat_name] = X[feat1] ** 2

        # 2. Create interaction terms with other important features
        for j in range(i+1, len(interaction_features)):
            feat2 = interaction_features[j]

            if feat2 not in X.columns:
                continue

            feat_name = f"{feat1}_x_{feat2}"
            X_enhanced[feat_name] = X[feat1] * X[feat2]

    # Create specific basketball-relevant interactions

    # Efficiency interactions
    if all(f in X.columns for f in ['Team1OffEfficiency', 'Team2DefEfficiency']):
        X_enhanced['Team1Off_vs_Team2Def'] = X['Team1OffEfficiency'] / X['Team2DefEfficiency']

    if all(f in X.columns for f in ['Team2OffEfficiency', 'Team1DefEfficiency']):
        X_enhanced['Team2Off_vs_Team1Def'] = X['Team2OffEfficiency'] / X['Team1DefEfficiency']

    # Seed interactions with other factors
    if all(f in X.columns for f in ['SeedDiff', 'WinRateDiff']):
        X_enhanced['SeedDiff_x_WinRateDiff'] = X['SeedDiff'] * X['WinRateDiff']

    if all(f in X.columns for f in ['SeedDiff', 'SOSPercentileDiff']):
        X_enhanced['SeedDiff_x_SOSDiff'] = X['SeedDiff'] * X['SOSPercentileDiff']

    # Experience interactions
    if all(f in X.columns for f in ['TourneyAppearancesDiff', 'SeedDiff']):
        X_enhanced['SeedDiff_x_ExperienceDiff'] = X['SeedDiff'] * X['TourneyAppearancesDiff']

    # Handle potential division by zero issues
    for col in X_enhanced.columns:
        if col not in X.columns:  # Only process new columns
            X_enhanced[col] = X_enhanced[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return X_enhanced

def drop_redundant_features(X, threshold=0.95):
    """
    Drop highly correlated features to reduce multicollinearity

    Args:
        X: DataFrame with features
        threshold: Correlation threshold for dropping

    Returns:
        DataFrame with reduced features
    """
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop highly correlated features
    X_reduced = X.drop(to_drop, axis=1)

    return X_reduced, to_drop

def handle_class_imbalance(X_train, y_train, method='combined', random_state=42):
    """
    Handle class imbalance in the training data

    Args:
        X_train: Training feature data
        y_train: Training target data
        method: Method for handling imbalance
                ('smote', 'under', 'combined')
        random_state: Random state for reproducibility

    Returns:
        Tuple of (resampled X_train, resampled y_train)
    """
    # Check class distribution
    class_counts = pd.Series(y_train).value_counts()

    # If classes are already relatively balanced, return original data
    if min(class_counts) / max(class_counts) >= 0.4:
        return X_train, y_train

    # Apply the specified resampling method
    if method == 'smote':
        # SMOTE: Synthetic Minority Over-sampling Technique
        resampler = SMOTE(random_state=random_state)
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)

    elif method == 'under':
        # Random Under-Sampling of the majority class
        resampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)

    elif method == 'combined':
        # Combined approach: SMOTE followed by under-sampling
        smote = SMOTE(sampling_strategy=0.8, random_state=random_state)
        under = RandomUnderSampler(sampling_strategy=0.9, random_state=random_state)

        # First apply SMOTE to oversample the minority class
        X_temp, y_temp = smote.fit_resample(X_train, y_train)

        # Then apply under-sampling to the majority class
        X_resampled, y_resampled = under.fit_resample(X_temp, y_temp)

    else:
        raise ValueError(f"Unknown resampling method: {method}")

    return X_resampled, y_resampled

def train_ensemble_model(X_train, y_train, random_state=42):
    """
    Train an ensemble model for tournament predictions

    Args:
        X_train: Training features
        y_train: Training targets
        random_state: Random state for reproducibility

    Returns:
        Trained ensemble model
    """
    # Train XGBoost model
    xgb_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        min_child_weight=3,
        random_state=random_state
    )

    # Train LightGBM model
    lgb_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        min_child_samples=20,
        random_state=random_state
    )

    # Train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=random_state,
        n_jobs=-1
    )

    # Create a voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('rf', rf_model)
        ],
        voting='soft',
        weights=[0.4, 0.4, 0.2]
    )

    # Fit the ensemble
    ensemble.fit(X_train, y_train)

    return ensemble
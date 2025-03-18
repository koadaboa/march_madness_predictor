# march_madness/models/common.py

import pandas as pd
import numpy as np

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
    # Move the function implementation from training.py to here
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

    # Tournament readiness interaction with seed
    if all(f in X.columns for f in ['Team1TournamentReadiness', 'Team2TournamentReadiness', 'SeedDiff']):
        X_enhanced['TournamentReadiness_x_SeedDiff'] = (X['Team1TournamentReadiness'] - X['Team2TournamentReadiness']) * X['SeedDiff']
    
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
    # Move the function implementation from training.py to here
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop highly correlated features
    X_reduced = X.drop(to_drop, axis=1)

    return X_reduced, to_drop
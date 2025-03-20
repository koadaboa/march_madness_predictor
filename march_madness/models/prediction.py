import pandas as pd
import numpy as np
import random
from ..data.loaders import extract_seed_number

def run_tournament_simulation_pre_tournament(seed_data, predictions_df, tournament_slots, num_simulations=1000):
    """
    Run Monte Carlo simulations of the tournament to refine predictions,
    ensuring we only use seed data for bracket structure without tournament results

    Args:
        seed_data: DataFrame with seed information for the current season only
        predictions_df: DataFrame with predictions for all possible matchups
        tournament_slots: DataFrame with tournament slot information
        num_simulations: Number of tournament simulations to run

    Returns:
        DataFrame with simulation-adjusted predictions
    """
    # Verify we're only using seed data for a single season
    if len(seed_data['Season'].unique()) > 1:
        print("WARNING: Seed data spans multiple seasons. Using the most recent season.")
        current_season = seed_data['Season'].max()
        seed_data = seed_data[seed_data['Season'] == current_season]

    # Create a dictionary for predictions lookup
    predictions_lookup = {}
    for _, row in predictions_df.iterrows():
        team1 = row['Team1ID']
        team2 = row['Team2ID']
        pred = row['Pred']

        # Store the prediction for this matchup
        predictions_lookup[f"{team1}_{team2}"] = pred

    # Create a bracket for the current season using seed information
    season = seed_data['Season'].iloc[0]

    # Group seeds by region (assuming seed format like W01, X02, etc.)
    regions = {}
    for _, row in seed_data.iterrows():
        team_id = row['TeamID']
        seed_str = row['Seed']
        region = seed_str[0] if isinstance(seed_str, str) and len(seed_str) >= 1 else 'Unknown'
        seed_num = int(seed_str[1:3]) if isinstance(seed_str, str) and len(seed_str) >= 3 else 16

        if region not in regions:
            regions[region] = []

        regions[region].append((seed_num, team_id))

    # Sort seeds within each region
    for region in regions:
        regions[region].sort()

    # Initialize simulation results tracker
    simulation_results = {}
    for _, row in predictions_df.iterrows():
        team1 = row['Team1ID']
        team2 = row['Team2ID']

        match_key = f"{team1}_{team2}"
        simulation_results[match_key] = {
            'Team1ID': team1,
            'Team2ID': team2,
            'OrigProb': row['Pred'],
            'SimCount': 0,
            'Team1Wins': 0
        }

    # Run simulations
    for sim in range(num_simulations):
        # Initialize tournament bracket
        bracket = []

        # Add all teams from all regions to the initial bracket (Round of 64)
        for region in regions:
            region_teams = regions[region]
            # Create first-round matchups (1 vs 16, 8 vs 9, etc.)
            for i in range(8):
                high_seed, high_team = region_teams[i]  # Higher seed (better team)
                low_seed, low_team = region_teams[15-i]  # Lower seed (worse team)

                bracket.append((high_team, low_team))

        # Simulate the entire tournament
        round_num = 1
        while len(bracket) > 1:
            next_round = []

            # Simulate each matchup in the current round
            for team1, team2 in bracket:
                # Get prediction for this matchup
                match_key = f"{team1}_{team2}"
                reverse_key = f"{team2}_{team1}"

                if match_key in predictions_lookup:
                    win_prob = predictions_lookup[match_key]
                elif reverse_key in predictions_lookup:
                    win_prob = 1 - predictions_lookup[reverse_key]
                else:
                    # If no direct prediction, use seed-based probability
                    team1_seed = extract_seed_number(seed_data[seed_data['TeamID'] == team1]['Seed'].values[0])
                    team2_seed = extract_seed_number(seed_data[seed_data['TeamID'] == team2]['Seed'].values[0])
                    win_prob = calculate_seed_based_probability(team1_seed, team2_seed)

                # Simulate the game
                winner = team1 if random.random() < win_prob else team2
                next_round.append(winner)

                # Update simulation counts
                if match_key in simulation_results:
                    simulation_results[match_key]['SimCount'] += 1
                    if winner == team1:
                        simulation_results[match_key]['Team1Wins'] += 1
                elif reverse_key in simulation_results:
                    simulation_results[reverse_key]['SimCount'] += 1
                    if winner == team2:
                        simulation_results[reverse_key]['Team1Wins'] += 1

            # Update bracket for next round
            bracket = []
            for i in range(0, len(next_round), 2):
                if i + 1 < len(next_round):
                    bracket.append((next_round[i], next_round[i+1]))

            round_num += 1

    # Calculate adjusted predictions based on simulation results
    adjusted_predictions = []

    for match_key, result in simulation_results.items():
        if result['SimCount'] > 0:
            # Calculate simulation-based probability
            sim_prob = result['Team1Wins'] / result['SimCount']

            # Blend with original probability
            # Weight more toward simulation for later games, which have more context
            if result['SimCount'] >= num_simulations / 4:
                # Matchup occurred in many simulations (likely later rounds)
                blend_factor = 0.7  # Weight more heavily on simulations
            else:
                # Early games or rare matchups
                blend_factor = 0.3

            # Blend probabilities
            adjusted_prob = blend_factor * sim_prob + (1 - blend_factor) * result['OrigProb']

            adjusted_predictions.append({
                'Team1ID': result['Team1ID'],
                'Team2ID': result['Team2ID'],
                'OrigProb': result['OrigProb'],
                'SimProb': sim_prob,
                'AdjustedProb': adjusted_prob,
                'SimCount': result['SimCount']
            })

    # Convert to DataFrame
    adjusted_df = pd.DataFrame(adjusted_predictions)

    # Update original predictions with adjusted values
    merged_predictions = predictions_df.copy()

    for _, row in adjusted_df.iterrows():
        team1 = row['Team1ID']
        team2 = row['Team2ID']
        adjusted_prob = row['AdjustedProb']

        # Find the matching row in predictions_df
        match_idx = merged_predictions[
            (merged_predictions['Team1ID'] == team1) &
            (merged_predictions['Team2ID'] == team2)
        ].index

        if len(match_idx) > 0:
            merged_predictions.loc[match_idx, 'Pred'] = adjusted_prob

    return merged_predictions, adjusted_df

def combine_predictions(mens_predictions, womens_predictions):
    """
    Combines predictions for men's and women's tournaments into a single submission file
    Only processes each unique matchup once (team1_id < team2_id)

    Args:
        mens_predictions: DataFrame with men's tournament predictions
        womens_predictions: DataFrame with women's tournament predictions

    Returns:
        DataFrame formatted for Kaggle submission
    """
    # Create submission rows for men's predictions
    mens_rows = []
    for _, row in mens_predictions.iterrows():
        season = row['Season']
        team1_id = row['Team1ID']
        team2_id = row['Team2ID']
        pred = row['Pred']

        # Since we've already ensured team1_id < team2_id during matchup generation,
        # this check is now redundant but kept for safety
        if team1_id < team2_id:
            matchup_id = f"{season}_{team1_id}_{team2_id}"
            mens_rows.append({"ID": matchup_id, "Pred": pred})
        else:
            # This branch should never be taken with the updated matchup generation
            matchup_id = f"{season}_{team2_id}_{team1_id}"
            mens_rows.append({"ID": matchup_id, "Pred": 1 - pred})  # Invert probability when teams are swapped
            print(f"WARNING: Found unexpected matchup order: team1_id={team1_id}, team2_id={team2_id}")

    # Create separate DataFrame
    mens_submission = pd.DataFrame(mens_rows)

    # Create submission rows for women's predictions
    womens_rows = []
    for _, row in womens_predictions.iterrows():
        season = row['Season']
        team1_id = row['Team1ID']
        team2_id = row['Team2ID']
        pred = row['Pred']

        # Same check for women's predictions
        if team1_id < team2_id:
            matchup_id = f"{season}_{team1_id}_{team2_id}"
            womens_rows.append({"ID": matchup_id, "Pred": pred})
        else:
            # This branch should never be taken with the updated matchup generation
            matchup_id = f"{season}_{team2_id}_{team1_id}"
            womens_rows.append({"ID": matchup_id, "Pred": 1 - pred})
            print(f"WARNING: Found unexpected matchup order: team1_id={team1_id}, team2_id={team2_id}")

    # Create separate DataFrame
    womens_submission = pd.DataFrame(womens_rows)

    # Combine predictions
    combined_submission = pd.concat([mens_submission, womens_submission], ignore_index=True)

    # Check for duplicates and remove them
    duplicate_ids = combined_submission['ID'].duplicated()
    if duplicate_ids.any():
        print(f"Removing {duplicate_ids.sum()} duplicate matchup IDs from the submission file")
        # Remove duplicates, keeping the first occurrence
        combined_submission = combined_submission.drop_duplicates(subset=['ID'], keep='first')

    print(f"Combined submission file has {len(combined_submission)} rows:")
    print(f"- Men's predictions: {len(mens_submission)}")
    print(f"- Women's predictions: {len(womens_submission)}")

    return combined_submission

def apply_final_four_seed_rule(predictions_df, seed_data):
    """Apply seed-based rules for Final4 and Championship games"""
    adjusted = predictions_df.copy()
    seed_improved = 0
    
    # Process each prediction row
    for i, row in adjusted.iterrows():
        if 'ExpectedRound' in row and row['ExpectedRound'] in ['Final4', 'Championship']:
            # Get team seeds
            team1_seed_row = seed_data[(seed_data['Season'] == row['Season']) & 
                                       (seed_data['TeamID'] == row['Team1ID'])]
            team2_seed_row = seed_data[(seed_data['Season'] == row['Season']) & 
                                       (seed_data['TeamID'] == row['Team2ID'])]
            
            if len(team1_seed_row) > 0 and len(team2_seed_row) > 0:
                try:
                    # Extract numeric seeds (e.g., 'W01' -> 1)
                    seed1 = int(team1_seed_row['Seed'].iloc[0][1:3]) if len(team1_seed_row['Seed'].iloc[0]) >= 3 else 16
                    seed2 = int(team2_seed_row['Seed'].iloc[0][1:3]) if len(team2_seed_row['Seed'].iloc[0]) >= 3 else 16
                    
                    # Based on historical NCAA Tournament Final Four data (1985-2023):
                    # 1-seeds win ~70% of their Final Four games
                    # 2-seeds win ~60% of their Final Four games
                    # 3+ seeds are at roughly 50% or below
                    
                    if seed1 < seed2:  # Team1 is better seed
                        if seed1 == 1:  # 1-seeds very strong in Final4
                            adjusted.loc[i, 'Pred'] = 0.78 if row['ExpectedRound'] == 'Final4' else 0.72
                            seed_improved += 1
                        elif seed1 == 2:  # 2-seeds also strong
                            adjusted.loc[i, 'Pred'] = 0.68 if row['ExpectedRound'] == 'Final4' else 0.62
                            seed_improved += 1
                        elif seed1 <= 4:  # Other top seeds get some advantage
                            adjusted.loc[i, 'Pred'] = 0.60
                            seed_improved += 1
                    elif seed2 < seed1:  # Team2 is better seed
                        if seed2 == 1:  # 1-seeds very strong in Final4
                            adjusted.loc[i, 'Pred'] = 0.22 if row['ExpectedRound'] == 'Final4' else 0.28
                            seed_improved += 1
                        elif seed2 == 2:  # 2-seeds also strong
                            adjusted.loc[i, 'Pred'] = 0.32 if row['ExpectedRound'] == 'Final4' else 0.38
                            seed_improved += 1
                        elif seed2 <= 4:  # Other top seeds get some advantage
                            adjusted.loc[i, 'Pred'] = 0.40
                            seed_improved += 1
                except:
                    # If there's any error, just preserve the original prediction
                    pass
    
    print(f"Applied seed-based adjustments to {seed_improved} Final4/Championship matchups")
    return adjusted

def apply_late_round_adjustments(predictions_df):
    """Apply final adjustments to improve late round accuracy"""
    adjusted = predictions_df.copy()
    
    # Find potential Final4 and Championship matchups
    for i, row in adjusted.iterrows():
        if row['ExpectedRound'] in ['Final4', 'Championship']:
            # Get current prediction
            current_pred = row['Pred']
            
            # Calculate seed difference
            seed_diff = row['SeedDiff'] if 'SeedDiff' in row else 0
            
            # For Final4 and Championship, highly seeded teams tend to be overvalued
            # Move predictions closer to 0.5, but preserve the direction
            if current_pred > 0.5:
                # Move high predictions down slightly
                adjusted.loc[i, 'Pred'] = 0.8 * current_pred + 0.2 * 0.55
            else:
                # Move low predictions up slightly
                adjusted.loc[i, 'Pred'] = 0.8 * current_pred + 0.2 * 0.45
                
    return adjusted
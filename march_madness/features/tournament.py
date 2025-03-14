import pandas as pd
import numpy as np
from march_madness.config import PREDICTION_SEASONS
from march_madness.utils.data_access import get_data_with_index

def calculate_tournament_history(tourney_df, current_season=None):
    """
    Calculate historical tournament performance for each team
    Ensures no data leakage by using only past seasons for any given season

    Args:
        tourney_df: DataFrame with tournament results
        current_season: Current season for prediction

    Returns:
        DataFrame with tournament history metrics
    """
    # SAFEGUARD: Ensure we don't use future tournament results
    if current_season is not None:
        is_prediction_season = current_season in PREDICTION_SEASONS
        if is_prediction_season:
            print(f"SAFEGUARD: Filtering out all tournament data for prediction season {current_season}")
            # Remove ALL data from prediction seasons
            tourney_df = tourney_df[~tourney_df['Season'].isin(PREDICTION_SEASONS)]
            print(f"After safeguard filtering, tourney_df contains seasons:", sorted(tourney_df['Season'].unique()))

    # Create records for winning teams
    w_records = tourney_df[['Season', 'WTeamID']].copy()
    w_records['TeamID'] = w_records['WTeamID']
    w_records['Won'] = 1
    w_records.drop('WTeamID', axis=1, inplace=True)

    # Create records for losing teams
    l_records = tourney_df[['Season', 'LTeamID']].copy()
    l_records['TeamID'] = l_records['LTeamID']
    l_records['Won'] = 0
    l_records.drop('LTeamID', axis=1, inplace=True)

    # Combine records
    all_records = pd.concat([w_records, l_records], ignore_index=True)

    # Define round mapping if not in data
    if 'DayNum' in tourney_df.columns and 'Round' not in tourney_df.columns:
        # Approximate rounds from DayNum
        round_mapping = {
            134: 'Round64', 135: 'Round64', 136: 'Round64', 137: 'Round64',
            138: 'Round32', 139: 'Round32',
            140: 'Sweet16', 141: 'Sweet16',
            142: 'Elite8', 143: 'Elite8',
            144: 'Final4',
            146: 'Championship'
        }
        tourney_df['Round'] = tourney_df['DayNum'].map(round_mapping)

        # Add round to all_records
        w_rounds = tourney_df[['Season', 'WTeamID', 'Round']].copy()
        w_rounds.columns = ['Season', 'TeamID', 'Round']

        l_rounds = tourney_df[['Season', 'LTeamID', 'Round']].copy()
        l_rounds.columns = ['Season', 'TeamID', 'Round']

        round_records = pd.concat([w_rounds, l_rounds], ignore_index=True)
        all_records = all_records.merge(round_records, on=['Season', 'TeamID'], how='left')

    # Calculate cumulative stats for each team by season
    team_history = []

    # Get all seasons
    seasons = tourney_df['Season'].unique()
    if current_season:
        if current_season not in seasons:
            seasons = np.append(seasons, [current_season])
        seasons = np.sort(seasons)

    for season in seasons:
        # Get past games before this season - THIS IS KEY TO PREVENT DATA LEAKAGE
        past_games = get_data_with_index(all_records, 'Season', lambda s: s < season)

        # Get unique teams that have been in the tournament before this season
        past_tourney = get_data_with_index(tourney_df, 'Season', lambda s: s < season, indexed_suffix='_by_season')
        tourney_teams = pd.concat([
            past_tourney['WTeamID'],
            past_tourney['LTeamID']
        ]).unique()

        for team_id in tourney_teams:
            team_games = past_games[past_games['TeamID'] == team_id]

            # Calculate tournament experience metrics
            appearances = team_games['Season'].nunique()
            wins = team_games['Won'].sum()
            games_played = len(team_games)
            win_rate = wins / games_played if games_played > 0 else 0

            # Calculate championship appearances and wins
            championships = 0

            if 'Round' in team_games.columns:
                championship_games = team_games[team_games['Round'] == 'Championship']
                championship_wins = championship_games[championship_games['Won'] == 1].shape[0]
                championships = championship_wins

            team_history.append({
                'Season': season,
                'TeamID': team_id,
                'TourneyAppearances': appearances,
                'TourneyWins': wins,
                'TourneyGames': games_played,
                'TourneyWinRate': win_rate,
                'Championships': championships
            })

    return pd.DataFrame(team_history)

def calculate_conference_strength(team_conferences, tourney_data, seed_data, current_season=None):
    """
    Calculate conference performance metrics in tournaments
    ensuring no data leakage from future tournaments

    Args:
        team_conferences: DataFrame with team conference assignments
        tourney_data: DataFrame with tournament results
        seed_data: DataFrame with tournament seeds
        current_season: Current season being analyzed (to filter data)

    Returns:
        DataFrame with conference strength metrics
    """
    # Filter tournament data to only include past seasons if current_season is provided
    # SAFEGUARD: Ensure we don't use future tournament results
    if current_season is not None:
        is_prediction_season = current_season in PREDICTION_SEASONS
        if is_prediction_season:
            print(f"SAFEGUARD: Filtering out all tournament data for prediction season {current_season}")
            # Remove ALL data from prediction seasons
            tourney_data = tourney_data[~tourney_data['Season'].isin(PREDICTION_SEASONS)]
            past_seed_data = seed_data[~seed_data['Season'].isin(PREDICTION_SEASONS)]
        else:
            past_seed_data = seed_data[seed_data['Season'] < current_season]
    else:
        past_seed_data = seed_data

    # Join tournament results with team conferences
    tourney_w = tourney_data[['Season', 'WTeamID']].merge(
        team_conferences,
        left_on=['Season', 'WTeamID'],
        right_on=['Season', 'TeamID'],
        how='left'
    ).drop('TeamID', axis=1)

    tourney_l = tourney_data[['Season', 'LTeamID']].merge(
        team_conferences,
        left_on=['Season', 'LTeamID'],
        right_on=['Season', 'TeamID'],
        how='left'
    ).drop('TeamID', axis=1)

    # Count tournament teams by conference and season
    tourney_teams = past_seed_data.merge(
        team_conferences,
        on=['Season', 'TeamID'],
        how='left'
    )

    conf_counts = tourney_teams.groupby(['Season', 'ConfAbbrev']).size().reset_index(name='TeamCount')

    # Count tournament wins by conference and season
    conf_wins = tourney_w.groupby(['Season', 'ConfAbbrev']).size().reset_index(name='Wins')

    # Count tournament losses by conference and season
    conf_losses = tourney_l.groupby(['Season', 'ConfAbbrev']).size().reset_index(name='Losses')

    # Combine metrics
    conf_metrics = conf_counts.merge(conf_wins, on=['Season', 'ConfAbbrev'], how='left')
    conf_metrics = conf_metrics.merge(conf_losses, on=['Season', 'ConfAbbrev'], how='left')

    # Fill NaNs with 0
    conf_metrics['Wins'] = conf_metrics['Wins'].fillna(0)
    conf_metrics['Losses'] = conf_metrics['Losses'].fillna(0)

    # Calculate win rate and efficiency
    conf_metrics['TourneyWinRate'] = conf_metrics['Wins'] / (conf_metrics['Wins'] + conf_metrics['Losses'])
    conf_metrics['WinsPerTeam'] = conf_metrics['Wins'] / conf_metrics['TeamCount']
    conf_metrics['TourneyWinRate'] = conf_metrics['TourneyWinRate'].fillna(0.5)
    conf_metrics['WinsPerTeam'] = conf_metrics['WinsPerTeam'].fillna(0)

    # Get unique seasons sorted
    seasons = sorted(conf_metrics['Season'].unique())

    # If we're analyzing a specific season, include it
    if current_season is not None and current_season not in seasons:
        seasons = sorted(list(seasons) + [current_season])

    # Calculate conference strength as a rolling average
    conf_strength = []

    for season in seasons:
        # Get all conferences for this season, including the current one
        current_confs = team_conferences[team_conferences['Season'] == season]['ConfAbbrev'].unique()

        for conf in current_confs:
            past_conf = get_data_with_index(conf_metrics, ('Season', 'ConfAbbrev'), (lambda s: s < season, conf))

            if past_conf.empty:
                # No past data for this conference
                conf_strength.append({
                    'Season': season,
                    'ConfAbbrev': conf,
                    'HistoricalWinRate': 0.5,
                    'HistoricalWinsPerTeam': 0.0
                })
            else:
                # Calculate exponentially weighted metrics with more weight on recent seasons
                past_seasons = past_conf['Season'].values
                current_idx = np.where(np.array(seasons) == season)[0][0]
                weights = []

                for past_season in past_seasons:
                    past_idx = np.where(np.array(seasons) == past_season)[0][0]
                    weight = 0.8 ** (current_idx - past_idx)  # Exponential decay
                    weights.append(weight)

                # Normalize weights
                weights = [w / sum(weights) for w in weights]

                win_rates = past_conf['TourneyWinRate'].values
                wins_per_team = past_conf['WinsPerTeam'].values

                # Calculate weighted averages
                weighted_win_rate = sum(w * wr for w, wr in zip(weights, win_rates))
                weighted_wins_per_team = sum(w * wpt for w, wpt in zip(weights, wins_per_team))

                conf_strength.append({
                    'Season': season,
                    'ConfAbbrev': conf,
                    'HistoricalWinRate': weighted_win_rate,
                    'HistoricalWinsPerTeam': weighted_wins_per_team
                })

    return pd.DataFrame(conf_strength)

def determine_expected_round(team1_seed, team2_seed):
    """
    Determine the expected tournament round for a matchup based on seeds
    This function relies only on static seed matchup patterns, not on actual tournament results

    Args:
        team1_seed: Seed number of team 1
        team2_seed: Seed number of team 2

    Returns:
        Expected round as string
    """
    # Handle non-tournament teams
    if team1_seed > 16 or team2_seed > 16:
        # If both are non-tournament teams, treat as "Unknown"
        if team1_seed > 16 and team2_seed > 16:
            return 'Unknown'
        # If one is a tournament team, treat as a first-round matchup
        return 'Round64'
    
    seed1 = min(team1_seed, team2_seed)
    seed2 = max(team1_seed, team2_seed)

    # First round matchups (1v16, 2v15, etc.)
    if (seed1 == 1 and seed2 == 16) or (seed1 == 2 and seed2 == 15) or \
       (seed1 == 3 and seed2 == 14) or (seed1 == 4 and seed2 == 13) or \
       (seed1 == 5 and seed2 == 12) or (seed1 == 6 and seed2 == 11) or \
       (seed1 == 7 and seed2 == 10) or (seed1 == 8 and seed2 == 9):
        return 'Round64'

    # Second round typical matchups
    elif (seed1 == 1 and seed2 in [8, 9]) or (seed1 == 2 and seed2 in [7, 10]) or \
         (seed1 == 3 and seed2 in [6, 11]) or (seed1 == 4 and seed2 in [5, 12]) or \
         (seed1 == 5 and seed2 in [4, 13]) or (seed1 == 6 and seed2 in [3, 14]) or \
         (seed1 == 7 and seed2 in [2, 15]) or (seed1 == 8 and seed2 in [1, 16]):
        return 'Round32'

    # Sweet 16 typical matchups
    elif (seed1 == 1 and seed2 in [4, 5, 12, 13]) or (seed1 == 2 and seed2 in [3, 6, 11, 14]) or \
         (seed1 == 3 and seed2 in [2, 7, 10, 15]) or (seed1 == 4 and seed2 in [1, 8, 9, 16]):
        return 'Sweet16'

    # Elite 8 typical matchups
    elif (seed1 == 1 and seed2 in [2, 3, 6, 7, 10, 11, 14, 15]) or \
         (seed1 == 2 and seed2 in [1, 4, 5, 8, 9, 12, 13, 16]):
        return 'Elite8'

    # Final Four - could be any combination, but typically higher seeds
    elif seed1 <= 4:
        return 'Final4'

    # Championship - usually top seeds
    elif seed1 <= 3:
        return 'Championship'

    # Default if we can't determine
    else:
        return 'Unknown'

def calculate_expected_round_features(seed_data, tourney_results, current_season=None):
    """
    Calculate team performance metrics by expected tournament round
    based on historical seed paths, ensuring no data leakage from future seasons

    Args:
        seed_data: DataFrame with seed information
        tourney_results: DataFrame with tournament results
        current_season: Current season being analyzed (to filter data)

    Returns:
        DataFrame with team performance metrics by likely rounds
    """
    # Filter tournament data to only include past seasons if current_season is provided
    if current_season is not None:
        # Check if there are any seasons before the current one
        if (tourney_results['Season'] < current_season).any():
            tourney_results = tourney_results[tourney_results['Season'] < current_season]
        else:
            # Handle the case where current_season is the earliest season
            # Return empty DataFrame with appropriate columns
            return pd.DataFrame(columns=['TeamID', 'WinRate_Round64', 'Games_Round64', 'Margin_Round64'])

    # Add seed numbers to tourney results
    tourney_with_seeds = tourney_results.merge(
        seed_data.rename(columns={'TeamID': 'WTeamID', 'Seed': 'WSeed'}),
        on=['Season', 'WTeamID'],
        how='left'
    )

    tourney_with_seeds = tourney_with_seeds.merge(
        seed_data.rename(columns={'TeamID': 'LTeamID', 'Seed': 'LSeed'}),
        on=['Season', 'LTeamID'],
        how='left'
    )

    # Extract seed numbers
    tourney_with_seeds['WSeedNum'] = tourney_with_seeds['WSeed'].apply(
        lambda x: int(x[1:3]) if isinstance(x, str) and len(x) >= 2 else None
    )

    tourney_with_seeds['LSeedNum'] = tourney_with_seeds['LSeed'].apply(
        lambda x: int(x[1:3]) if isinstance(x, str) and len(x) >= 2 else None
    )

    # Add expected round based on seed matchup
    tourney_with_seeds['ExpectedRound'] = tourney_with_seeds.apply(
        lambda row: determine_expected_round(row['WSeedNum'], row['LSeedNum']),
        axis=1
    )

    # Create records for winning teams
    w_records = tourney_with_seeds[['Season', 'WTeamID', 'WSeedNum', 'ExpectedRound', 'WScore', 'LScore']].copy()
    w_records.columns = ['Season', 'TeamID', 'SeedNum', 'ExpectedRound', 'TeamScore', 'OpponentScore']
    w_records['Win'] = 1

    # Create records for losing teams
    l_records = tourney_with_seeds[['Season', 'LTeamID', 'LSeedNum', 'ExpectedRound', 'LScore', 'WScore']].copy()
    l_records.columns = ['Season', 'TeamID', 'SeedNum', 'ExpectedRound', 'TeamScore', 'OpponentScore']
    l_records['Win'] = 0

    # Combine records
    all_records = pd.concat([w_records, l_records], ignore_index=True)

    # Calculate metrics by expected round for each team
    round_performance = all_records.groupby(['Season', 'TeamID', 'ExpectedRound']).agg({
        'Win': ['mean', 'count'],
        'TeamScore': 'mean',
        'OpponentScore': 'mean',
        'SeedNum': 'mean'
    })

    # Flatten column hierarchy
    round_performance.columns = ['_'.join(col).strip() for col in round_performance.columns.values]

    # Calculate margin
    round_performance['ScoreMargin_mean'] = round_performance['TeamScore_mean'] - round_performance['OpponentScore_mean']

    # Reshape the data to have one row per team with columns for each expected round
    round_performance = round_performance.reset_index()

    # Create pivot tables for each metric
    win_rate_by_round = round_performance.pivot_table(
        index=['Season', 'TeamID'],
        columns='ExpectedRound',
        values='Win_mean',
        fill_value=0
    ).add_prefix('WinRate_')

    games_by_round = round_performance.pivot_table(
        index=['Season', 'TeamID'],
        columns='ExpectedRound',
        values='Win_count',
        fill_value=0
    ).add_prefix('Games_')

    margin_by_round = round_performance.pivot_table(
        index=['Season', 'TeamID'],
        columns='ExpectedRound',
        values='ScoreMargin_mean',
        fill_value=0
    ).add_prefix('Margin_')

    # Combine all metrics and reset index
    round_stats = win_rate_by_round.join([games_by_round, margin_by_round]).reset_index()

     # If we still don't have a Season column for some reason (unlikely at this point)
    if 'Season' not in round_stats.columns:
        print("WARNING: Adding missing Season column to round_performance")
        if current_season is not None:
            round_stats['Season'] = current_season

    return round_stats

def calculate_pressure_performance(tourney_results, seed_data, current_season=None):
    """
    Calculate team performance metrics in high-pressure situations,
    ensuring no data leakage from future seasons

    Args:
        tourney_results: DataFrame with tournament results
        seed_data: DataFrame with seed information
        current_season: Current season being analyzed (to filter data)

    Returns:
        DataFrame with pressure performance metrics
    """


    # Filter tournament data to only include past seasons if current_season is provided
    # SAFEGUARD: Ensure we don't use future tournament results
    if current_season is not None:
        is_prediction_season = current_season in PREDICTION_SEASONS
        if is_prediction_season:
            print(f"SAFEGUARD: Filtering out all tournament data for prediction season {current_season}")
            # Remove ALL data from prediction seasons
            tourney_results = tourney_results[~tourney_results['Season'].isin(PREDICTION_SEASONS)]

    # Add seed information to tournament results
    tourney_with_seeds = tourney_results.merge(
        seed_data.rename(columns={'TeamID': 'WTeamID', 'Seed': 'WSeed'}),
        on=['Season', 'WTeamID'],
        how='left'
    )

    tourney_with_seeds = tourney_with_seeds.merge(
        seed_data.rename(columns={'TeamID': 'LTeamID', 'Seed': 'LSeed'}),
        on=['Season', 'LTeamID'],
        how='left'
    )

    # Extract seed numbers
    tourney_with_seeds['WSeedNum'] = tourney_with_seeds['WSeed'].apply(
        lambda x: int(x[1:3]) if isinstance(x, str) and len(x) >= 2 else None
    )

    tourney_with_seeds['LSeedNum'] = tourney_with_seeds['LSeed'].apply(
        lambda x: int(x[1:3]) if isinstance(x, str) and len(x) >= 2 else None
    )

    # Define pressure situations
    # 1. Close games (margin ≤ 5 points)
    close_games = tourney_with_seeds[abs(tourney_with_seeds['WScore'] - tourney_with_seeds['LScore']) <= 5].copy()

    # 2. Upset opportunities (lower seed vs higher seed, e.g. 8 vs 1)
    upset_opps = tourney_with_seeds[tourney_with_seeds['WSeedNum'] > tourney_with_seeds['LSeedNum']].copy()

    # 3. Games in Sweet 16 or later (approximated by expected round)
    tourney_with_seeds['ExpectedRound'] = tourney_with_seeds.apply(
        lambda row: determine_expected_round(row['WSeedNum'], row['LSeedNum']),
        axis=1
    )

    late_rounds = tourney_with_seeds[tourney_with_seeds['ExpectedRound'].isin(
        ['Sweet16', 'Elite8', 'Final4', 'Championship']
    )].copy()

    # Calculate performance for each team in pressure situations
    pressure_stats = {}

    # Process all teams that have appeared in tournaments
    all_tourney_teams = np.unique(np.concatenate([
        tourney_with_seeds['WTeamID'].unique(),
        tourney_with_seeds['LTeamID'].unique()
    ]))

    # If we're analyzing a specific season, also include teams from that season
    # who haven't been in previous tournaments
    if current_season is not None:
        current_teams = seed_data[seed_data['Season'] == current_season]['TeamID'].unique()
        all_teams = np.union1d(all_tourney_teams, current_teams)
    else:
        all_teams = all_tourney_teams

    for team_id in all_teams:
        # Close games
        close_wins = len(close_games[close_games['WTeamID'] == team_id])
        close_losses = len(close_games[close_games['LTeamID'] == team_id])
        close_total = close_wins + close_losses
        close_win_rate = close_wins / close_total if close_total > 0 else None

        # Upset opportunities
        upset_wins = len(upset_opps[upset_opps['WTeamID'] == team_id])
        upset_opps_total = len(tourney_with_seeds[
            ((tourney_with_seeds['WTeamID'] == team_id) & (tourney_with_seeds['WSeedNum'] > tourney_with_seeds['LSeedNum'])) |
            ((tourney_with_seeds['LTeamID'] == team_id) & (tourney_with_seeds['LSeedNum'] > tourney_with_seeds['WSeedNum']))
        ])
        upset_win_rate = upset_wins / upset_opps_total if upset_opps_total > 0 else None

        # Late rounds
        late_wins = len(late_rounds[late_rounds['WTeamID'] == team_id])
        late_losses = len(late_rounds[late_rounds['LTeamID'] == team_id])
        late_total = late_wins + late_losses
        late_win_rate = late_wins / late_total if late_total > 0 else None

        # Store metrics
        pressure_stats[team_id] = {
            'CloseGames_Count': close_total,
            'CloseGames_WinRate': close_win_rate if close_win_rate is not None else 0.5,
            'UpsetOpps_Count': upset_opps_total,
            'UpsetOpps_WinRate': upset_win_rate if upset_win_rate is not None else 0.5,
            'LateRounds_Count': late_total,
            'LateRounds_WinRate': late_win_rate if late_win_rate is not None else 0.5
        }

    # Convert to DataFrame
    pressure_df = pd.DataFrame.from_dict(pressure_stats, orient='index')
    pressure_df = pressure_df.reset_index().rename(columns={'index': 'TeamID'})

    # IMPORTANT: Add Season column
    if current_season is not None:
        pressure_df['Season'] = current_season
    else:
        # If no current_season specified, use the most recent one from seed_data
        pressure_df['Season'] = seed_data['Season'].max() if not seed_data.empty else None

    # Calculate composite pressure score
    pressure_df['PressureScore'] = (
        (0.4 * pressure_df['CloseGames_WinRate'] * np.minimum(pressure_df['CloseGames_Count'] / 5, 1)) +
        (0.3 * pressure_df['UpsetOpps_WinRate'] * np.minimum(pressure_df['UpsetOpps_Count'] / 3, 1)) +
        (0.3 * pressure_df['LateRounds_WinRate'] * np.minimum(pressure_df['LateRounds_Count'] / 3, 1))
    )

    return pressure_df

def calculate_conference_tournament_impact(conf_tourney_results, team_conferences, reg_season_results, current_season=None):
    """
    Calculate the impact of conference tournament performance on NCAA tournament
    ensuring no data leakage from future seasons

    Args:
        conf_tourney_results: DataFrame with conference tournament results
        team_conferences: DataFrame with team conference assignments
        reg_season_results: DataFrame with regular season results
        current_season: Current season being analyzed (to filter data)

    Returns:
        DataFrame with conference tournament impact metrics
    """
    # Filter to only include data up to the current season if specified
    if current_season is not None:
        # For conference tournament results, we can use the current season
        # since these occur before NCAA tournament
        conf_tourney = conf_tourney_results[conf_tourney_results['Season'] <= current_season]
        # For regular season, we can also use current season data
        reg_season = reg_season_results[reg_season_results['Season'] <= current_season]
    else:
        conf_tourney = conf_tourney_results.copy()
        reg_season = reg_season_results.copy()

    # Combine conference tournament results with team conferences
    conf_tourney = conf_tourney.merge(
        team_conferences.rename(columns={'TeamID': 'WTeamID'}),
        on=['Season', 'WTeamID', 'ConfAbbrev'],
        how='left'
    )

    conf_tourney = conf_tourney.merge(
        team_conferences.rename(columns={'TeamID': 'LTeamID'}),
        on=['Season', 'LTeamID', 'ConfAbbrev'],
        how='left'
    )

    # Create records for winning teams
    w_records = conf_tourney[['Season', 'WTeamID', 'ConfAbbrev', 'DayNum']].copy()
    w_records.columns = ['Season', 'TeamID', 'ConfAbbrev', 'DayNum']
    w_records['Win'] = 1

    # Create records for losing teams
    l_records = conf_tourney[['Season', 'LTeamID', 'ConfAbbrev', 'DayNum']].copy()
    l_records.columns = ['Season', 'TeamID', 'ConfAbbrev', 'DayNum']
    l_records['Win'] = 0

    # Combine records
    all_conf_games = pd.concat([w_records, l_records], ignore_index=True)

    # Identify conference tournament champions
    # For each season and conference, find the team that won the last game
    last_day_games = conf_tourney.groupby(['Season', 'ConfAbbrev'])['DayNum'].max().reset_index()
    championship_games = conf_tourney.merge(
        last_day_games,
        on=['Season', 'ConfAbbrev', 'DayNum'],
        how='inner'
    )

    champions = championship_games[['Season', 'WTeamID', 'ConfAbbrev']].copy()
    champions.columns = ['Season', 'TeamID', 'ConfAbbrev']
    champions['ConfChampion'] = 1

    # Merge champion indicator with all games
    all_conf_games = all_conf_games.merge(
        champions,
        on=['Season', 'TeamID', 'ConfAbbrev'],
        how='left'
    )
    all_conf_games['ConfChampion'] = all_conf_games['ConfChampion'].fillna(0)

    # Calculate conference tournament stats by team and season
    conf_tourney_stats = all_conf_games.groupby(['Season', 'TeamID']).agg({
        'Win': ['sum', 'mean', 'count'],
        'ConfChampion': 'max'
    })

    # Flatten column hierarchy
    conf_tourney_stats.columns = ['_'.join(col).strip() for col in conf_tourney_stats.columns.values]
    conf_tourney_stats = conf_tourney_stats.rename(columns={
        'Win_sum': 'ConfTourney_Wins',
        'Win_mean': 'ConfTourney_WinRate',
        'Win_count': 'ConfTourney_Games',
        'ConfChampion_max': 'ConfChampion'
    })

    # Reset index for easier merging
    conf_tourney_stats = conf_tourney_stats.reset_index()

    # Calculate momentum difference between conference tournament and regular season
    team_reg_momentum = []

    # Process each season separately to ensure no data leakage
    for season in reg_season['Season'].unique():
        # Find the start day of conference tournaments for this season
        season_conf_tourney = conf_tourney[conf_tourney['Season'] == season]

        if len(season_conf_tourney) > 0:
            conf_start_day = season_conf_tourney['DayNum'].min()
        else:
            # If no conference tournament data, use a default
            conf_start_day = 132  # Approximate

        # Get regular season games before conference tournaments
        season_data = get_data_with_index(reg_season, season)
        season_reg = season_data[season_data['DayNum'] < conf_start_day]

        # For each team, get the last 5 games of regular season
        teams = np.union1d(season_reg['WTeamID'].unique(), season_reg['LTeamID'].unique())

        for team_id in teams:
            # Get team's last 5 games as winner
            last_w_games = season_reg[season_reg['WTeamID'] == team_id].sort_values('DayNum', ascending=False).head(5)
            w_count = len(last_w_games)

            # Get team's last 5 games as loser
            last_l_games = season_reg[season_reg['LTeamID'] == team_id].sort_values('DayNum', ascending=False).head(5)
            l_count = len(last_l_games)

            # Calculate regular season momentum (last 5 games win rate)
            reg_momentum = w_count / (w_count + l_count) if (w_count + l_count) > 0 else 0.5

            team_reg_momentum.append({
                'Season': season,
                'TeamID': team_id,
                'RegularSeason_LastGames_WinRate': reg_momentum
            })

    reg_momentum_df = pd.DataFrame(team_reg_momentum)

    # Merge with conference tournament stats
    conf_impact = conf_tourney_stats.merge(
        reg_momentum_df,
        on=['Season', 'TeamID'],
        how='left'
    )

    # Calculate momentum change
    conf_impact['MomentumChange'] = conf_impact['ConfTourney_WinRate'] - conf_impact['RegularSeason_LastGames_WinRate']
    conf_impact['RegularSeason_LastGames_WinRate'] = conf_impact['RegularSeason_LastGames_WinRate'].fillna(0.5)
    conf_impact['MomentumChange'] = conf_impact['MomentumChange'].fillna(0)

    return conf_impact

def create_seed_trend_features(seed_data, tourney_results, current_season=None):
    """
    Create features based on a team's seeding trend across years
    and how well they perform relative to seed expectations
    ensuring no data leakage from future tournaments

    Args:
        seed_data: DataFrame with seed information
        tourney_results: DataFrame with tournament results
        current_season: Current season being analyzed (to filter data)

    Returns:
        DataFrame with seed trend metrics
    """
    # Calculate historical seeding trends
    seed_history = []

    # Get all seasons in the data
    all_seasons = sorted(seed_data['Season'].unique())

    # If analyzing a specific season that's not in the data, include it
    if current_season is not None and current_season not in all_seasons:
        all_seasons = sorted(list(all_seasons) + [current_season])

    for season in all_seasons:
        # Skip seasons we don't have seed data for
        if season not in seed_data['Season'].unique():
            continue

        for team_id in seed_data[seed_data['Season'] == season]['TeamID'].unique():
            # Get team's tournament appearances before this season
            past_appearances = seed_data[
                (seed_data['Season'] < season) &
                (seed_data['TeamID'] == team_id)
            ].sort_values('Season')

            # If team has previous appearances, calculate trends
            if len(past_appearances) > 0:
                # Extract numerical seeds
                past_appearances['SeedNum'] = past_appearances['Seed'].apply(
                    lambda x: int(x[1:3]) if isinstance(x, str) and len(x) >= 2 else None
                )

                # Calculate metrics
                avg_seed = past_appearances['SeedNum'].mean()
                min_seed = past_appearances['SeedNum'].min()  # Best seed (lowest number)
                max_seed = past_appearances['SeedNum'].max()  # Worst seed (highest number)
                appearances = len(past_appearances)

                # Calculate trend (negative means improving seeds)
                if len(past_appearances) >= 2:
                    recent_seasons = past_appearances.tail(3)
                    if len(recent_seasons) >= 2:
                        seed_trend = np.polyfit(recent_seasons['Season'], recent_seasons['SeedNum'], 1)[0]
                    else:
                        seed_trend = 0
                else:
                    seed_trend = 0

                seed_history.append({
                    'Season': season,
                    'TeamID': team_id,
                    'HistoricalAvgSeed': avg_seed,
                    'BestHistoricalSeed': min_seed,
                    'WorstHistoricalSeed': max_seed,
                    'PriorAppearances': appearances,
                    'SeedTrend': seed_trend  # Negative means improving (better seeds)
                })

    seed_history_df = pd.DataFrame(seed_history)

    # Calculate performance relative to seed expectation
    seed_performance = []

    # Filter tournament results if we're only looking at data up to a specific season
    if current_season is not None:
        past_tourney = tourney_results[tourney_results['Season'] < current_season]
    else:
        past_tourney = tourney_results

    # Get all seasons for analysis
    seasons_to_analyze = all_seasons if current_season is None else [s for s in all_seasons if s <= current_season]

    for season in seasons_to_analyze:
        # Only use past seasons for analysis relative to the current season being analyzed
        season_past_tourney = get_data_with_index(past_tourney, 'Season', lambda s: s < season, indexed_suffix='_by_season')

        # Add expected rounds if not in the data
        if 'ExpectedRound' not in season_past_tourney.columns:
            # Add seed information to past tournament
            season_past_tourney = season_past_tourney.merge(
                seed_data.rename(columns={'TeamID': 'WTeamID', 'Seed': 'WSeed'}),
                on=['Season', 'WTeamID'],
                how='left'
            )

            season_past_tourney = season_past_tourney.merge(
                seed_data.rename(columns={'TeamID': 'LTeamID', 'Seed': 'LSeed'}),
                on=['Season', 'LTeamID'],
                how='left'
            )

            # Extract seed numbers
            season_past_tourney['WSeedNum'] = season_past_tourney['WSeed'].apply(
                lambda x: int(x[1:3]) if isinstance(x, str) and len(x) >= 2 else None
            )

            season_past_tourney['LSeedNum'] = season_past_tourney['LSeed'].apply(
                lambda x: int(x[1:3]) if isinstance(x, str) and len(x) >= 2 else None
            )

            # Determine expected round
            season_past_tourney['ExpectedRound'] = season_past_tourney.apply(
                lambda row: determine_expected_round(row['WSeedNum'], row['LSeedNum']),
                axis=1
            )

        # Define round values for calculating expected performance
        round_values = {
            'Round64': 1,
            'Round32': 2,
            'Sweet16': 3,
            'Elite8': 4,
            'Final4': 5,
            'Championship': 6,
            'Unknown': 0  # For games without round info
        }

        # Skip if we don't have seed data for this season
        if season not in seed_data['Season'].unique():
            continue

        # Get team's tournament history
        for team_id in seed_data[seed_data['Season'] == season]['TeamID'].unique():
            # Get the team's appearances and results
            team_appearances = get_data_with_index(seed_data, ('Season', 'TeamID'), 
            (lambda s: s < season, team_id), indexed_suffix='_by_team')

            if len(team_appearances) > 0:
                performance_by_appearance = []

                for _, appearance in team_appearances.iterrows():
                    app_season = appearance['Season']
                    app_seed_str = appearance['Seed']
                    app_seed = int(app_seed_str[1:3]) if isinstance(app_seed_str, str) and len(app_seed_str) >= 2 else None

                    # Find the team's games in this tournament
                    season_games = get_data_with_index(season_past_tourney, 'Season', app_season, indexed_suffix='_by_season')
                    team_games = season_games[(season_games['WTeamID'] == team_id) | (season_games['LTeamID'] == team_id)]

                    # Determine how far the team advanced
                    if len(team_games) > 0:
                        # Find the last game (highest round value)
                        if 'ExpectedRound' in team_games.columns:
                            team_rounds = [round_values.get(r, 0) for r in team_games['ExpectedRound']]
                            max_round = max(team_rounds) if team_rounds else 0
                        else:
                            max_round = 1  # Default if no round info

                        # For winner of championship, add one more round
                        championship_win = False
                        if 'ExpectedRound' in team_games.columns:
                            championship_win = any(
                                (team_games['ExpectedRound'] == 'Championship') &
                                (team_games['WTeamID'] == team_id)
                            )

                        if championship_win:
                            # Championship winner
                            furthest_round = 7
                        else:
                            furthest_round = max_round
                    else:
                        # Did not appear or no games recorded
                        furthest_round = 0

                    performance_by_appearance.append({
                        'Season': app_season,
                        'Seed': app_seed,
                        'FurthestRound': furthest_round
                    })

                performance_df = pd.DataFrame(performance_by_appearance)

                # Calculate expected performance by seed
                # Simplified seed expectations based on historical patterns
                seed_expectations = {
                    1: 4.5,  # 1 seeds typically reach Final Four
                    2: 3.5,  # 2 seeds typically reach Elite Eight
                    3: 3.0,  # 3 seeds typically reach Sweet 16
                    4: 2.5,  # 4 seeds between Sweet 16 and Round of 32
                    5: 2.0,  # 5 seeds typically reach Round of 32
                    6: 2.0,  # 6 seeds typically reach Round of 32
                    7: 1.5,  # 7 seeds between Round of 32 and Round of 64
                    8: 1.5,  # 8 seeds between Round of 32 and Round of 64
                    9: 1.5,  # 9 seeds between Round of 32 and Round of 64
                    10: 1.5, # 10 seeds between Round of 32 and Round of 64
                    11: 1.5, # 11 seeds can sometimes reach Sweet 16
                    12: 1.5, # 12 seeds have upset potential
                    13: 1.0, # 13 seeds typically exit in Round of 64
                    14: 1.0, # 14 seeds typically exit in Round of 64
                    15: 1.0, # 15 seeds typically exit in Round of 64
                    16: 1.0  # 16 seeds typically exit in Round of 64
                }

                # Calculate team's performance vs. expectations
                perf_vs_exp = []

                for _, perf in performance_df.iterrows():
                    seed_num = perf['Seed']
                    actual_round = perf['FurthestRound']

                    # Get expected performance for this seed
                    expected_round = seed_expectations.get(seed_num, 1)

                    # Calculate difference
                    round_diff = actual_round - expected_round

                    perf_vs_exp.append(round_diff)

                # Calculate average over-/under-performance
                avg_perf_diff = sum(perf_vs_exp) / len(perf_vs_exp) if perf_vs_exp else 0

                seed_performance.append({
                    'Season': season,
                    'TeamID': team_id,
                    'AvgRoundPerformance': avg_perf_diff
                })

    seed_performance_df = pd.DataFrame(seed_performance)

    # Check if DataFrames have the required columns before merging
    if 'Season' not in seed_history_df.columns:
        print("WARNING: 'Season' column missing in seed_history_df")
        if len(seed_history_df) > 0:
            seed_history_df['Season'] = current_season
    
    if seed_performance_df is not None and len(seed_performance_df) > 0:
        if 'Season' not in seed_performance_df.columns:
            print("WARNING: 'Season' column missing in seed_performance_df")
            seed_performance_df['Season'] = current_season
        
        # Now merge when both have the needed columns
        seed_features = seed_history_df.merge(
            seed_performance_df,
            on=['Season', 'TeamID'],
            how='left'
        )
    else:
        # Handle case where seed_performance_df is empty or None
        seed_features = seed_history_df.copy()

    # Fill missing values
    seed_features['AvgRoundPerformance'] = seed_features['AvgRoundPerformance'].fillna(0)

    return seed_features

def calculate_coach_tournament_metrics(tourney_results, coaches_data, current_season=None):
    """
    Calculate detailed coach performance metrics in tournaments
    ensuring no data leakage from future tournaments

    Args:
        tourney_results: DataFrame with tournament results
        coaches_data: DataFrame with coach information
        current_season: Current season being analyzed (to filter data)

    Returns:
        DataFrame with coach tournament metrics
    """
    import pandas as pd
    import numpy as np

    # Filter tournament data if we're only looking at data up to a specific season
    if current_season is not None:
        tourney_results = tourney_results[tourney_results['Season'] < current_season]

    # Add coach information to tournament results
    # Create combined dataframe with coach for each team in each game
    rename_columns_w = {'TeamID': 'WTeamID', 'CoachName': 'WCoach'}
    rename_columns_l = {'TeamID': 'LTeamID', 'CoachName': 'LCoach'}

    # Add FirstDayNum and LastDayNum to the columns being renamed if they exist
    if 'FirstDayNum' in coaches_data.columns:
        rename_columns_w['FirstDayNum'] = 'WFirstDayNum'
        rename_columns_l['FirstDayNum'] = 'LFirstDayNum'

    if 'LastDayNum' in coaches_data.columns:
        rename_columns_w['LastDayNum'] = 'WLastDayNum'
        rename_columns_l['LastDayNum'] = 'LLastDayNum'

    # Perform merges with properly renamed columns
    tourney_coaches = tourney_results.merge(
        coaches_data.rename(columns=rename_columns_w),
        on=['Season', 'WTeamID'],
        how='left'
    )

    tourney_coaches = tourney_coaches.merge(
        coaches_data.rename(columns=rename_columns_l),
        on=['Season', 'LTeamID'],
        how='left'
    )

    # Filter to ensure the coaches were active during the game
    if 'FirstDayNum' in coaches_data.columns and 'LastDayNum' in coaches_data.columns:
        tourney_coaches = tourney_coaches[
            (tourney_coaches['DayNum'] >= tourney_coaches['WFirstDayNum']) &
            (tourney_coaches['DayNum'] <= tourney_coaches['WLastDayNum']) &
            (tourney_coaches['DayNum'] >= tourney_coaches['LFirstDayNum']) &
            (tourney_coaches['DayNum'] <= tourney_coaches['LLastDayNum'])
        ]

    # Add round information if not present
    if 'Round' not in tourney_coaches.columns:
        # Approximate rounds from DayNum
        round_mapping = {
            134: 'Round64', 135: 'Round64', 136: 'Round64', 137: 'Round64',
            138: 'Round32', 139: 'Round32',
            140: 'Sweet16', 141: 'Sweet16',
            142: 'Elite8', 143: 'Elite8',
            144: 'Final4',
            146: 'Championship'
        }
        tourney_coaches['Round'] = tourney_coaches['DayNum'].map(round_mapping)

    # Create a flattened DataFrame with season-specific metrics
    coach_tourney_metrics = []

    # For each season, calculate coach's metrics up to that point
    all_seasons = sorted(coaches_data['Season'].unique())

    # If we're analyzing a specific season that's not in the data, include it
    if current_season is not None and current_season not in all_seasons:
        all_seasons = sorted(list(all_seasons) + [current_season])

    for season in all_seasons:
        # Skip seasons we don't have coach data for
        if season not in coaches_data['Season'].unique():
            continue

        # Get all coaches active in this season
        season_coaches = coaches_data[coaches_data['Season'] == season]

        for _, coach_row in season_coaches.iterrows():
            coach_name = coach_row['CoachName']
            team_id = coach_row['TeamID']

            # Skip if coach name is missing
            if pd.isna(coach_name):
                continue

            # Get coach's history prior to this season
            prior_w_games = tourney_coaches[
                (tourney_coaches['WCoach'] == coach_name) &
                (tourney_coaches['Season'] < season)
            ]

            prior_l_games = tourney_coaches[
                (tourney_coaches['LCoach'] == coach_name) &
                (tourney_coaches['Season'] < season)
            ]

            # Calculate metrics from prior seasons
            prior_total = len(prior_w_games) + len(prior_l_games)

            if prior_total > 0:
                prior_win_rate = len(prior_w_games) / prior_total

                # Round-specific win rates
                round_rates = {}

                if 'Round' in tourney_coaches.columns:
                    for round_name in ['Round64', 'Round32', 'Sweet16', 'Elite8', 'Final4', 'Championship']:
                        round_w = prior_w_games[prior_w_games['Round'] == round_name]
                        round_l = prior_l_games[prior_l_games['Round'] == round_name]

                        round_total = len(round_w) + len(round_l)
                        round_win_rate = len(round_w) / round_total if round_total > 0 else 0.5

                        round_rates[f'{round_name}_WinRate'] = round_win_rate

                # Calculate upset metrics - use WSeedNum/LSeedNum if available or use a placeholder
                if 'WSeedNum' in tourney_coaches.columns and 'LSeedNum' in tourney_coaches.columns:
                    upsets_achieved = sum(prior_w_games['WSeedNum'] > prior_w_games['LSeedNum'])
                    upsets_suffered = sum(prior_l_games['LSeedNum'] < prior_l_games['WSeedNum'])

                    # Calculate upset rates
                    upset_opportunities = sum(prior_w_games['WSeedNum'] > prior_w_games['LSeedNum']) + \
                                         sum(prior_l_games['LSeedNum'] > prior_l_games['WSeedNum'])

                    upset_rate = upsets_achieved / upset_opportunities if upset_opportunities > 0 else 0.5

                    upset_defense_opportunities = sum(prior_w_games['WSeedNum'] < prior_w_games['LSeedNum']) + \
                                               sum(prior_l_games['LSeedNum'] < prior_l_games['WSeedNum'])

                    upset_defense_rate = 1 - (upsets_suffered / upset_defense_opportunities) if upset_defense_opportunities > 0 else 0.5
                else:
                    upset_rate = 0.5
                    upset_defense_rate = 0.5

                # Championships won
                if 'Round' in tourney_coaches.columns:
                    championships = len(prior_w_games[prior_w_games['Round'] == 'Championship'])
                else:
                    championships = 0

                metrics = {
                    'Season': season,  # CRITICAL: Include Season column
                    'TeamID': team_id,
                    'CoachName': coach_name,
                    'PriorTourneyGames': prior_total,
                    'PriorTourneyWinRate': prior_win_rate,
                    'PriorChampionships': championships,
                    'UpsetRate': upset_rate,
                    'UpsetDefenseRate': upset_defense_rate
                }

                # Add round-specific rates
                metrics.update(round_rates)

                coach_tourney_metrics.append(metrics)
            else:
                # No prior tournament experience
                metrics = {
                    'Season': season,  # CRITICAL: Include Season column
                    'TeamID': team_id,
                    'CoachName': coach_name,
                    'PriorTourneyGames': 0,
                    'PriorTourneyWinRate': 0.5,  # Neutral default
                    'PriorChampionships': 0,
                    'UpsetRate': 0.5,
                    'UpsetDefenseRate': 0.5
                }

                coach_tourney_metrics.append(metrics)

    # Create DataFrame from metrics
    coach_metrics_df = pd.DataFrame(coach_tourney_metrics)

    # SAFEGUARD: Check if Season column exists (it should at this point)
    if 'Season' not in coach_metrics_df.columns and len(coach_metrics_df) > 0:
        print("WARNING: Adding missing Season column to coach_metrics")
        if current_season is not None:
            coach_metrics_df['Season'] = current_season

    # Verify that required columns exist before returning
    if len(coach_metrics_df) > 0:
        required_columns = ['Season', 'TeamID', 'PriorTourneyWinRate', 'UpsetRate', 'UpsetDefenseRate']
        for col in required_columns:
            if col not in coach_metrics_df.columns:
                print(f"WARNING: Missing required column {col} in coach_metrics_df")
                # Add missing column with default values
                if col == 'Season' and current_season is not None:
                    coach_metrics_df[col] = current_season
                elif col == 'TeamID':
                    # This is critical and should not be missing, but provide default
                    coach_metrics_df[col] = 0
                else:
                    coach_metrics_df[col] = 0.5

    # If no metrics were created, return an empty DataFrame with required columns
    if len(coach_metrics_df) == 0:
        print("WARNING: No coach tournament metrics found, returning empty DataFrame with required columns")
        coach_metrics_df = pd.DataFrame(columns=['Season', 'TeamID', 'CoachName', 'PriorTourneyWinRate',
                                              'UpsetRate', 'UpsetDefenseRate'])
        if current_season is not None:
            # Add a placeholder row to avoid completely empty DataFrame
            coach_metrics_df = pd.DataFrame([{
                'Season': current_season,
                'TeamID': 0,
                'CoachName': 'Unknown',
                'PriorTourneyWinRate': 0.5,
                'UpsetRate': 0.5,
                'UpsetDefenseRate': 0.5
            }])

    return coach_metrics_df

def calculate_tournament_readiness(team_profile, momentum_data, sos_data, team_id, season):
    """
    Calculate a team's tournament readiness score based on:
    - Late season performance
    - Performance against tournament-level competition
    - Experience in high-pressure games

    Args:
        team_profile: Series with team season profile
        momentum_data: DataFrame with momentum metrics
        sos_data: DataFrame with strength of schedule metrics
        team_id: Team ID
        season: Season year

    Returns:
        Dictionary with tournament readiness metrics
    """
    readiness_metrics = {}

    # Extract team's end-of-season momentum
    team_momentum = momentum_data[(momentum_data['Season'] == season) &
                                  (momentum_data['TeamID'] == team_id)]

    if len(team_momentum) > 0:
        # Recent win momentum (last 5 games)
        win_momentum = team_momentum['Win_Last5'].values[0] if 'Win_Last5' in team_momentum.columns else 0.5
        readiness_metrics['RecentWinMomentum'] = win_momentum

        # Recent scoring momentum
        score_momentum = team_momentum['ScoreMargin_Last5'].values[0] if 'ScoreMargin_Last5' in team_momentum.columns else 0
        readiness_metrics['RecentScoreMomentum'] = score_momentum

        # Recent exponential win weighting (more weight to most recent games)
        exp_win_momentum = team_momentum['Win_Exp'].values[0] if 'Win_Exp' in team_momentum.columns else 0.5
        readiness_metrics['ExpWinMomentum'] = exp_win_momentum
    else:
        # Default values if no momentum data
        readiness_metrics['RecentWinMomentum'] = 0.5
        readiness_metrics['RecentScoreMomentum'] = 0
        readiness_metrics['ExpWinMomentum'] = 0.5

    # Extract team's performance against quality competition
    team_sos = sos_data[(sos_data['Season'] == season) &
                       (sos_data['TeamID'] == team_id)]

    if len(team_sos) > 0:
        # Win rate against tournament-caliber teams
        win_vs_good = team_sos['WinRateVsWinningTeams'].values[0]
        readiness_metrics['WinVsQualityTeams'] = win_vs_good

        # Number of games against quality opponents (experience in tough games)
        quality_games = team_sos['GamesAgainstWinningTeams'].values[0]
        # Normalize by typical number of quality games
        readiness_metrics['QualityGameExperience'] = min(quality_games / 10, 1.0)
    else:
        # Default values if no SOS data
        readiness_metrics['WinVsQualityTeams'] = 0.5
        readiness_metrics['QualityGameExperience'] = 0.5

    # Add close game performance from team profile
    if 'CloseGameWinRate' in team_profile:
        readiness_metrics['CloseGameWinRate'] = team_profile['CloseGameWinRate']
    else:
        readiness_metrics['CloseGameWinRate'] = 0.5

    # Calculate the overall tournament readiness score (weighted average)
    readiness_metrics['TournamentReadiness'] = (
        0.3 * readiness_metrics['RecentWinMomentum'] +
        0.1 * (readiness_metrics['RecentScoreMomentum'] / 10) +  # Normalize score margin
        0.2 * readiness_metrics['ExpWinMomentum'] +
        0.3 * readiness_metrics['WinVsQualityTeams'] +
        0.1 * readiness_metrics['QualityGameExperience']
    )

    return readiness_metrics
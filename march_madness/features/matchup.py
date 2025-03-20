import pandas as pd
import numpy as np
from ..data.loaders import extract_seed_number
from ..features.tournament import determine_expected_round, calculate_tournament_readiness
from ..utils.data_access import get_data_with_index

def calculate_matchup_style_compatibility(team1_profile, team2_profile):
    """
    Calculate style compatibility metrics between two teams, with proper defensive metrics

    Args:
        team1_profile: Series with team 1 profile metrics
        team2_profile: Series with team 2 profile metrics

    Returns:
        Dictionary with style matchup metrics
    """
    # Define team styles based on metrics
    # Ensure we're looking at the right column names for defensive metrics
    team1_defensive = team1_profile.get('Season_DefEfficiency', team1_profile.get('DefEfficiency', 1.0)) < 0.95
    team2_defensive = team2_profile.get('Season_DefEfficiency', team2_profile.get('DefEfficiency', 1.0)) < 0.95

    # Faster pace teams
    team1_fast_paced = team1_profile.get('FastPaced', False) or team1_profile.get('Tempo', 70) > 70
    team2_fast_paced = team2_profile.get('FastPaced', False) or team2_profile.get('Tempo', 70) > 70

    # Three-point reliant teams
    team1_three_reliant = team1_profile.get('OutsideScoring', False) or team1_profile.get('Pct_Points_From_3', 0.3) > 0.3
    team2_three_reliant = team2_profile.get('OutsideScoring', False) or team2_profile.get('Pct_Points_From_3', 0.3) > 0.3

    # Inside scoring teams
    team1_inside_scoring = team1_profile.get('InsideScoring', False) or team1_profile.get('Pct_Points_From_2', 0.5) > 0.5
    team2_inside_scoring = team2_profile.get('InsideScoring', False) or team2_profile.get('Pct_Points_From_2', 0.5) > 0.5

    # Determine style compatibility
    # Same pace matchup
    same_pace = (team1_fast_paced and team2_fast_paced) or (not team1_fast_paced and not team2_fast_paced)

    # Shooting style matchup
    complementary_shooting = (team1_three_reliant and team2_inside_scoring) or (team1_inside_scoring and team2_three_reliant)

    # Defensive vs offensive matchup
    defense_vs_offense = (team1_defensive and not team2_defensive) or (not team1_defensive and team2_defensive)

    # Calculate style advantage metrics
    matchup_style = {
        'PaceAdvantage': 0.1 if (team1_fast_paced and not team2_fast_paced) else -0.1 if (not team1_fast_paced and team2_fast_paced) else 0,
        'ShootingStyleAdvantage': 0.05 if complementary_shooting else 0,
        'DefensiveEdge': 0.1 if (team1_defensive and not team2_defensive) else -0.1 if (not team1_defensive and team2_defensive) else 0,
    }

    return matchup_style

def calculate_seed_based_probability(seed1, seed2):
    """
    Calculate win probability based on historical seed performance

    Args:
        seed1: Seed number of team 1
        seed2: Seed number of team 2

    Returns:
        Probability of team 1 winning
    """
    # Historical performance data for seed matchups
    # These values represent the approximate historical win rates
    # when lower seed number (stronger team) plays higher seed number (weaker team)
    # Handle cases with non-tournament teams
    if seed1 > 16 and seed2 > 16:
        # If both are non-tournament teams, slight advantage to the "better" team
        return 0.52 if seed1 < seed2 else 0.48
    elif seed1 > 16:
        # Tournament team (seed2) vs non-tournament team (seed1)
        return max(0.05, 0.35 - (0.02 * min(seed2, 16)))
    elif seed2 > 16:
        # Tournament team (seed1) vs non-tournament team (seed2)
        return min(0.95, 0.65 + (0.02 * min(seed1, 16)))

    # 1 vs 16 matchups: ~99% win rate for 1 seeds
    if (seed1 == 1 and seed2 == 16) or (seed1 == 16 and seed2 == 1):
        return 0.99 if seed1 < seed2 else 0.01

    # 2 vs 15 matchups: ~94% win rate for 2 seeds
    if (seed1 == 2 and seed2 == 15) or (seed1 == 15 and seed2 == 2):
        return 0.94 if seed1 < seed2 else 0.06

    # 3 vs 14 matchups: ~85% win rate for 3 seeds
    if (seed1 == 3 and seed2 == 14) or (seed1 == 14 and seed2 == 3):
        return 0.85 if seed1 < seed2 else 0.15

    # 4 vs 13 matchups: ~80% win rate for 4 seeds
    if (seed1 == 4 and seed2 == 13) or (seed1 == 13 and seed2 == 4):
        return 0.80 if seed1 < seed2 else 0.20

    # 5 vs 12 matchups: ~67% win rate for 5 seeds (12 seeds often upset)
    if (seed1 == 5 and seed2 == 12) or (seed1 == 12 and seed2 == 5):
        return 0.67 if seed1 < seed2 else 0.33

    # 6 vs 11 matchups: ~65% win rate for 6 seeds
    if (seed1 == 6 and seed2 == 11) or (seed1 == 11 and seed2 == 6):
        return 0.65 if seed1 < seed2 else 0.35

    # 7 vs 10 matchups: ~60% win rate for 7 seeds
    if (seed1 == 7 and seed2 == 10) or (seed1 == 10 and seed2 == 7):
        return 0.60 if seed1 < seed2 else 0.40

    # 8 vs 9 matchups: ~50% win rate (very even)
    if (seed1 == 8 and seed2 == 9) or (seed1 == 9 and seed2 == 8):
        return 0.51 if seed1 < seed2 else 0.49

    # For other matchups, use a seed difference based approach
    seed_diff = abs(seed1 - seed2)

    if seed_diff <= 2:
        # Close seeds
        base_prob = 0.55  # Slight edge to better seed
    elif seed_diff <= 4:
        # Moderate difference
        base_prob = 0.65
    elif seed_diff <= 8:
        # Large difference
        base_prob = 0.75
    else:
        # Very large difference
        base_prob = 0.85

    # Adjust for which team has the better seed
    return base_prob if seed1 < seed2 else (1 - base_prob)

def create_matchup_features_pre_tournament(team1_id, team2_id, season, team_profiles, seed_data,
                                        momentum_data, sos_data, coach_features, tourney_history,
                                        conf_strength, team_conferences, team_consistency=None,
                                        team_playstyle=None, round_performance=None, pressure_metrics=None,
                                        conf_impact=None, seed_features=None, coach_metrics=None, team_to_seed=None):
    """
    Create features for tournament matchups before the tournament starts

    Args:
        team1_id: Team 1 ID
        team2_id: Team 2 ID
        season: Season year
        team_profiles: DataFrame with team season profiles
        seed_data: DataFrame with seed information
        momentum_data: DataFrame with momentum features
        sos_data: DataFrame with strength of schedule metrics
        coach_features: DataFrame with coach features
        tourney_history: DataFrame with tournament history
        conf_strength: DataFrame with conference strength
        team_conferences: DataFrame with team conference assignments
        team_consistency: DataFrame with team consistency metrics (optional)
        team_playstyle: DataFrame with team play style metrics (optional)
        round_performance: DataFrame with round performance metrics (optional)
        pressure_metrics: DataFrame with pressure performance metrics (optional)
        conf_impact: DataFrame with conference tournament impact (optional)
        seed_features: DataFrame with seed trend features (optional)
        coach_metrics: DataFrame with coach tournament metrics (optional)

    Returns:
        Dictionary with matchup features
    """
    try:
        # Convert inputs to appropriate types
        season_value = int(season)
        team1_id_value = int(team1_id)
        team2_id_value = int(team2_id)

        # Check if teams exist in profiles
        team1_in_profiles = team1_id_value in team_profiles[team_profiles['Season'] == season_value]['TeamID'].values
        team2_in_profiles = team2_id_value in team_profiles[team_profiles['Season'] == season_value]['TeamID'].values

        if not team1_in_profiles or not team2_in_profiles:
            missing = []
            if not team1_in_profiles: missing.append(f"Team1 ({team1_id_value}) not in profiles")
            if not team2_in_profiles: missing.append(f"Team2 ({team2_id_value}) not in profiles")
            return None  # Early exit if any team is missing

        # Get team profiles
        # Assuming team_profiles is indexed by ['Season', 'TeamID']
        team1_profile_df = get_data_with_index(team_profiles, (season_value, team1_id_value))
        team2_profile_df = get_data_with_index(team_profiles, (season_value, team2_id_value))

        if len(team1_profile_df) == 0 or len(team2_profile_df) == 0:
            return None

        # Get seeds from provided mapping or fallback to seed data
        if team_to_seed is not None:
            team1_seed_num = team_to_seed.get(team1_id_value, 17)
            team2_seed_num = team_to_seed.get(team2_id_value, 17)
        else:
            # Try to get seeds from seed_data (only for tournament teams)
            team1_seed_row = get_data_with_index(seed_data, (season_value, team1_id_value))
            team2_seed_row = get_data_with_index(seed_data, (season_value, team2_id_value))

            team1_seed_num = extract_seed_number(team1_seed_row['Seed'].values[0]) if len(team1_seed_row) > 0 else 17
            team2_seed_num = extract_seed_number(team2_seed_row['Seed'].values[0]) if len(team2_seed_row) > 0 else 17

        # Determine expected round for this matchup
        expected_round = determine_expected_round(team1_seed_num, team2_seed_num)

        # Convert profile DataFrames to Series
        team1_profile = team1_profile_df.iloc[0]
        team2_profile = team2_profile_df.iloc[0]

        # Get conference information - using stored primitive values
        team1_conf = get_data_with_index(team_conferences, (season_value, team1_id_value), indexed_suffix='_by_team')
        team2_conf = get_data_with_index(team_conferences, (season_value, team2_id_value), indexed_suffix='_by_team')

        # Extract conference information
        team1_conf_abbrev = team1_conf['ConfAbbrev'].values[0] if len(team1_conf) > 0 else 'Unknown'
        team2_conf_abbrev = team2_conf['ConfAbbrev'].values[0] if len(team2_conf) > 0 else 'Unknown'

        # Get conference strength - using stored primitive values
        team1_conf_strength = get_data_with_index(conf_strength, (season_value, team1_conf_abbrev), indexed_suffix='_by_conf')
        team2_conf_strength = get_data_with_index(conf_strength, (season_value, team2_conf_abbrev), indexed_suffix='_by_conf')

        # Extract conference strength metrics
        team1_conf_win_rate = team1_conf_strength['HistoricalWinRate'].values[0] if len(team1_conf_strength) > 0 else 0.5
        team2_conf_win_rate = team2_conf_strength['HistoricalWinRate'].values[0] if len(team2_conf_strength) > 0 else 0.5

        team1_conf_wins_per_team = team1_conf_strength['HistoricalWinsPerTeam'].values[0] if len(team1_conf_strength) > 0 else 0
        team2_conf_wins_per_team = team2_conf_strength['HistoricalWinsPerTeam'].values[0] if len(team2_conf_strength) > 0 else 0

        # Get coach features - using stored primitive values
        team1_coach = get_data_with_index(coach_features, (season_value, team1_id_value), indexed_suffix='_by_team')
        team2_coach = get_data_with_index(coach_features, (season_value, team2_id_value), indexed_suffix='_by_team')

        # Extract coach metrics
        team1_coach_exp = team1_coach['CoachYearsExp'].values[0] if len(team1_coach) > 0 else 0
        team2_coach_exp = team2_coach['CoachYearsExp'].values[0] if len(team2_coach) > 0 else 0

        team1_coach_tourney_exp = team1_coach['CoachTourneyExp'].values[0] if len(team1_coach) > 0 else 0
        team2_coach_tourney_exp = team2_coach['CoachTourneyExp'].values[0] if len(team2_coach) > 0 else 0

        team1_coach_champ = team1_coach['CoachChampionships'].values[0] if len(team1_coach) > 0 else 0
        team2_coach_champ = team2_coach['CoachChampionships'].values[0] if len(team2_coach) > 0 else 0

        # Initialize these vairables before the conditional check
        team1_history = pd.DataFrame()
        team2_history = pd.DataFrame()

        # Get tournament history with safeguard for empty dataframe
        if not tourney_history.empty:
            team1_history = get_data_with_index(tourney_history, (season_value, team1_id_value), indexed_suffix='_by_team')
            team2_history = get_data_with_index(tourney_history, (season_value, team2_id_value), indexed_suffix='_by_team')

            # Extract tournament history metrics
            team1_appearances = team1_history['TourneyAppearances'].values[0] if len(team1_history) > 0 else 0
            team2_appearances = team2_history['TourneyAppearances'].values[0] if len(team2_history) > 0 else 0

            team1_tourney_win_rate = team1_history['TourneyWinRate'].values[0] if len(team1_history) > 0 else 0.5
            team2_tourney_win_rate = team2_history['TourneyWinRate'].values[0] if len(team2_history) > 0 else 0.5

            team1_championships = team1_history['Championships'].values[0] if len(team1_history) > 0 else 0
            team2_championships = team2_history['Championships'].values[0] if len(team2_history) > 0 else 0
        else:
            # Default values if no tournament history exists
            team1_appearances = 0
            team2_appearances = 0
            team1_tourney_win_rate = 0.5
            team2_tourney_win_rate = 0.5
            team1_championships = 0
            team2_championships = 0

        # Extract tournament history metrics
        team1_appearances = team1_history['TourneyAppearances'].values[0] if len(team1_history) > 0 else 0
        team2_appearances = team2_history['TourneyAppearances'].values[0] if len(team2_history) > 0 else 0

        team1_tourney_win_rate = team1_history['TourneyWinRate'].values[0] if len(team1_history) > 0 else 0.5
        team2_tourney_win_rate = team2_history['TourneyWinRate'].values[0] if len(team2_history) > 0 else 0.5

        team1_championships = team1_history['Championships'].values[0] if len(team1_history) > 0 else 0
        team2_championships = team2_history['Championships'].values[0] if len(team2_history) > 0 else 0

        # Get momentum data - using stored primitive values
        team1_momentum = get_data_with_index(momentum_data, (season_value, team1_id_value), indexed_suffix='_by_team')
        team2_momentum = get_data_with_index(momentum_data, (season_value, team2_id_value), indexed_suffix='_by_team')

        # Get various momentum metrics if available, otherwise use defaults
        momentum_metrics = {
            'Win_Last3': ('Win_Last3', 0.5),
            'ScoreMargin_Last3': ('ScoreMargin_Last3', 0),
            'OffEfficiency_Last3': ('OffEfficiency_Last3', 1.0),
            'Win_Last5': ('Win_Last5', 0.5),
            'ScoreMargin_Last5': ('ScoreMargin_Last5', 0),
            'Win_Last10': ('Win_Last10', 0.5),
            'Win_Exp': ('Win_Exp', 0.5),
            'ScoreMargin_Exp': ('ScoreMargin_Exp', 0)
        }

        team1_momentum_vals = {}
        team2_momentum_vals = {}

        for metric, (col_name, default) in momentum_metrics.items():
            # Get momentum values, use team average if not available
            if len(team1_momentum) > 0 and col_name in team1_momentum.columns:
                team1_momentum_vals[metric] = team1_momentum[col_name].values[0]
            else:
                # Try to map to existing column
                team1_momentum_vals[metric] = default

            if len(team2_momentum) > 0 and col_name in team2_momentum.columns:
                team2_momentum_vals[metric] = team2_momentum[col_name].values[0]
            else:
                team2_momentum_vals[metric] = default

        # Get strength of schedule - using stored primitive values
        team1_sos = get_data_with_index(sos_data, (season_value, team1_id_value), indexed_suffix='_by_team')
        team2_sos = get_data_with_index(sos_data, (season_value, team2_id_value), indexed_suffix='_by_team')

        # Define default SOS values
        sos_defaults = {
            'AvgOpponentWinRate': 0.5,
            'AvgOpponentOffEff': 1.0,
            'AvgOpponentDefEff': 1.0,
            'WinRateVsWinningTeams': 0.5,
            'SOSPercentile': 0.5
        }

        team1_sos_vals = {}
        team2_sos_vals = {}

        for metric, default in sos_defaults.items():
            if len(team1_sos) > 0 and metric in team1_sos.columns:
                team1_sos_vals[metric] = team1_sos[metric].values[0]
            else:
                team1_sos_vals[metric] = default

            if len(team2_sos) > 0 and metric in team2_sos.columns:
                team2_sos_vals[metric] = team2_sos[metric].values[0]
            else:
                team2_sos_vals[metric] = default

        # Get new features

        # Team consistency metrics
        team1_consistency = {}
        team2_consistency = {}

        if team_consistency is not None:
            # Use stored primitive values
            team1_consistency = get_data_with_index(team_consistency, (season_value, team1_id_value), indexed_suffix='_by_team')
            team2_consistency = get_data_with_index(team_consistency, (season_value, team2_id_value), indexed_suffix='_by_team')

            if len(team1_consistency) > 0:
                for col in team1_consistency.columns:
                    if col not in ['Season', 'TeamID']:
                        team1_consistency[col] = team1_consistency[col].values[0]
            else:
                # Default values
                team1_consistency = {
                    'Score_CV': 0.1,
                    'Margin_CV': 0.2,
                    'CloseGame_WinRate': 0.5,
                    'Score_std': 5,
                    'ScoreMargin_std': 5
                }

            if len(team2_consistency) > 0:
                for col in team2_consistency.columns:
                    if col not in ['Season', 'TeamID']:
                        team2_consistency[col] = team2_consistency[col].values[0]
            else:
                # Default values
                team2_consistency = {
                    'Score_CV': 0.1,
                    'Margin_CV': 0.2,
                    'CloseGame_WinRate': 0.5,
                    'Score_std': 5,
                    'ScoreMargin_std': 5
                }
        else:
            # Default values
            team1_consistency = {'Score_CV': 0.1, 'Margin_CV': 0.2, 'CloseGame_WinRate': 0.5, 'Score_std': 5, 'ScoreMargin_std': 5}
            team2_consistency = {'Score_CV': 0.1, 'Margin_CV': 0.2, 'CloseGame_WinRate': 0.5, 'Score_std': 5, 'ScoreMargin_std': 5}

        # Team play style metrics
        if team_playstyle is not None:
            # Use stored primitive values
            team1_style_data = get_data_with_index(team_playstyle, (season_value, team1_id_value), indexed_suffix='_by_team')
            team2_style_data = get_data_with_index(team_playstyle, (season_value, team2_id_value), indexed_suffix='_by_team')

            # Calculate style matchup compatibility
            if len(team1_style_data) > 0 and len(team2_style_data) > 0:
                style_matchup = calculate_matchup_style_compatibility(
                    team1_style_data.iloc[0], team2_style_data.iloc[0]
                )
            else:
                style_matchup = {'PaceAdvantage': 0, 'ShootingStyleAdvantage': 0, 'DefensiveEdge': 0}
        else:
            style_matchup = {'PaceAdvantage': 0, 'ShootingStyleAdvantage': 0, 'DefensiveEdge': 0}

        # Tournament readiness metrics
        team1_readiness = calculate_tournament_readiness(
            team1_profile, momentum_data, sos_data, team1_id_value, season_value
        )

        team2_readiness = calculate_tournament_readiness(
            team2_profile, momentum_data, sos_data, team2_id_value, season_value
        )

        # Get tournament-specific metrics
        team1_pressure = {}
        team2_pressure = {}

        if pressure_metrics is not None and isinstance(pressure_metrics, pd.DataFrame) and 'TeamID' in pressure_metrics.columns:
            # Note: Not requiring 'Season' column
            team1_press = get_data_with_index(pressure_metrics, 'TeamID', team1_id_value)
            team2_press = get_data_with_index(pressure_metrics, 'TeamID', team2_id_value)

            if len(team1_press) > 0:
                team1_pressure = {
                    'PressureScore': team1_press['PressureScore'].values[0],
                    'CloseGames_WinRate': team1_press['CloseGames_WinRate'].values[0]
                }
            else:
                team1_pressure = {'PressureScore': 0.5, 'CloseGames_WinRate': 0.5}

            if len(team2_press) > 0:
                team2_pressure = {
                    'PressureScore': team2_press['PressureScore'].values[0],
                    'CloseGames_WinRate': team2_press['CloseGames_WinRate'].values[0]
                }
            else:
                team2_pressure = {'PressureScore': 0.5, 'CloseGames_WinRate': 0.5}
        else:
            team1_pressure = {'PressureScore': 0.5, 'CloseGames_WinRate': 0.5}
            team2_pressure = {'PressureScore': 0.5, 'CloseGames_WinRate': 0.5}

        # Get performance in expected round (if available)
        team1_round_perf = {}
        team2_round_perf = {}

        if round_performance is not None and isinstance(round_performance, pd.DataFrame) and 'TeamID' in round_performance.columns:
            # Note: We're not requiring 'Season' column since your debug shows it doesn't have one
            team1_round = get_data_with_index(round_performance, 'TeamID', team1_id_value)
            team2_round = get_data_with_index(round_performance, 'TeamID', team2_id_value)

            if len(team1_round) > 0:
                # Extract round-specific metrics for the expected round
                round_prefix = f"WinRate_{expected_round}"
                if round_prefix in team1_round.columns:
                    team1_round_perf['RoundWinRate'] = team1_round[round_prefix].values[0]
                else:
                    team1_round_perf['RoundWinRate'] = 0.5
            else:
                team1_round_perf['RoundWinRate'] = 0.5

            if len(team2_round) > 0:
                # Extract round-specific metrics for the expected round
                round_prefix = f"WinRate_{expected_round}"
                if round_prefix in team2_round.columns:
                    team2_round_perf['RoundWinRate'] = team2_round[round_prefix].values[0]
                else:
                    team2_round_perf['RoundWinRate'] = 0.5
            else:
                team2_round_perf['RoundWinRate'] = 0.5
        else:
            team1_round_perf['RoundWinRate'] = 0.5
            team2_round_perf['RoundWinRate'] = 0.5

        # Conference tournament impact (if available)
        team1_conf_impact = {}
        team2_conf_impact = {}

        if conf_impact is not None:
            # Use stored primitive values
            team1_conf = get_data_with_index(conf_impact, (season_value, team1_id_value), indexed_suffix='_by_team')
            team2_conf = get_data_with_index(conf_impact, (season_value, team2_id_value), indexed_suffix='_by_team')

            if len(team1_conf) > 0:
                team1_conf_impact = {
                    'ConfTourney_WinRate': team1_conf['ConfTourney_WinRate'].values[0],
                    'ConfChampion': team1_conf['ConfChampion'].values[0],
                    'MomentumChange': team1_conf['MomentumChange'].values[0]
                }
            else:
                team1_conf_impact = {'ConfTourney_WinRate': 0.5, 'ConfChampion': 0, 'MomentumChange': 0}

            if len(team2_conf) > 0:
                team2_conf_impact = {
                    'ConfTourney_WinRate': team2_conf['ConfTourney_WinRate'].values[0],
                    'ConfChampion': team2_conf['ConfChampion'].values[0],
                    'MomentumChange': team2_conf['MomentumChange'].values[0]
                }
            else:
                team2_conf_impact = {'ConfTourney_WinRate': 0.5, 'ConfChampion': 0, 'MomentumChange': 0}
        else:
            team1_conf_impact = {'ConfTourney_WinRate': 0.5, 'ConfChampion': 0, 'MomentumChange': 0}
            team2_conf_impact = {'ConfTourney_WinRate': 0.5, 'ConfChampion': 0, 'MomentumChange': 0}

        # Seed trend features (if available)
        team1_seed_trends = {}
        team2_seed_trends = {}

        if seed_features is not None:
            # Use stored primitive values
            team1_seed = get_data_with_index(seed_features, (season_value, team1_id_value), indexed_suffix='_by_team')
            team2_seed = get_data_with_index(seed_features, (season_value, team2_id_value), indexed_suffix='_by_team')

            if len(team1_seed) > 0:
                team1_seed_trends = {
                    'HistoricalAvgSeed': team1_seed['HistoricalAvgSeed'].values[0],
                    'SeedTrend': team1_seed['SeedTrend'].values[0],
                    'AvgRoundPerformance': team1_seed['AvgRoundPerformance'].values[0]
                }
            else:
                team1_seed_trends = {'HistoricalAvgSeed': team1_seed_num, 'SeedTrend': 0, 'AvgRoundPerformance': 0}

            if len(team2_seed) > 0:
                team2_seed_trends = {
                    'HistoricalAvgSeed': team2_seed['HistoricalAvgSeed'].values[0],
                    'SeedTrend': team2_seed['SeedTrend'].values[0],
                    'AvgRoundPerformance': team2_seed['AvgRoundPerformance'].values[0]
                }
            else:
                team2_seed_trends = {'HistoricalAvgSeed': team2_seed_num, 'SeedTrend': 0, 'AvgRoundPerformance': 0}
        else:
            team1_seed_trends = {'HistoricalAvgSeed': team1_seed_num, 'SeedTrend': 0, 'AvgRoundPerformance': 0}
            team2_seed_trends = {'HistoricalAvgSeed': team2_seed_num, 'SeedTrend': 0, 'AvgRoundPerformance': 0}

        # Advanced coach metrics (if available)
        team1_coach_tourney = {}
        team2_coach_tourney = {}

        if coach_metrics is not None and isinstance(coach_metrics, pd.DataFrame) and len(coach_metrics.columns) > 0:
            # Check if required columns exist before trying to use them
            if 'Season' in coach_metrics.columns and 'TeamID' in coach_metrics.columns:
                # Only filter if both required columns exist
                team1_coach = get_data_with_index(coach_metrics, (season_value, team1_id_value), indexed_suffix='_by_team')
                team2_coach = get_data_with_index(coach_metrics, (season_value, team2_id_value), indexed_suffix='_by_team')

                if len(team1_coach) > 0:
                    team1_coach_tourney = {
                        'PriorTourneyWinRate': team1_coach['PriorTourneyWinRate'].values[0],
                        'UpsetRate': team1_coach['UpsetRate'].values[0],
                        'UpsetDefenseRate': team1_coach_metrics['UpsetDefenseRate'].values[0]
                    }

                    # Add round-specific performance if available
                    round_col = f"{expected_round}_WinRate"
                    if round_col in team1_coach_metrics.columns:
                        team1_coach_tourney['RoundWinRate'] = team1_coach_metrics[round_col].values[0]
                    else:
                        team1_coach_tourney['RoundWinRate'] = 0.5
                else:
                    team1_coach_tourney = {
                        'PriorTourneyWinRate': 0.5,
                        'UpsetRate': 0.5,
                        'UpsetDefenseRate': 0.5,
                        'RoundWinRate': 0.5
                    }

                if len(team2_coach) > 0:
                    team2_coach_tourney = {
                        'PriorTourneyWinRate': team2_coach['PriorTourneyWinRate'].values[0],
                        'UpsetRate': team2_coach['UpsetRate'].values[0],
                        'UpsetDefenseRate': team2_coach['UpsetDefenseRate'].values[0]
                    }

                    # Add round-specific performance if available
                    round_col = f"{expected_round}_WinRate"
                    if round_col in team2_coach.columns:
                        team2_coach_tourney['RoundWinRate'] = team2_coach[round_col].values[0]
                    else:
                        team2_coach_tourney['RoundWinRate'] = 0.5
                else:
                    team2_coach_tourney = {
                        'PriorTourneyWinRate': 0.5,
                        'UpsetRate': 0.5,
                        'UpsetDefenseRate': 0.5,
                        'RoundWinRate': 0.5
                    }
            else:
                # Missing required columns, use default values
                print(f"Coach metrics is missing required columns")
                team1_coach_tourney = {'PriorTourneyWinRate': 0.5, 'UpsetRate': 0.5, 'UpsetDefenseRate': 0.5, 'RoundWinRate': 0.5}
                team2_coach_tourney = {'PriorTourneyWinRate': 0.5, 'UpsetRate': 0.5, 'UpsetDefenseRate': 0.5, 'RoundWinRate': 0.5}
        else:
            # Default values for missing coach metrics
            team1_coach_tourney = {'PriorTourneyWinRate': 0.5, 'UpsetRate': 0.5, 'UpsetDefenseRate': 0.5, 'RoundWinRate': 0.5}
            team2_coach_tourney = {'PriorTourneyWinRate': 0.5, 'UpsetRate': 0.5, 'UpsetDefenseRate': 0.5, 'RoundWinRate': 0.5}

        # Create feature dictionary with all our predictors
        feature_dict = {
            'Season': season_value,
            'Team1ID': team1_id_value,
            'Team2ID': team2_id_value,
            'ExpectedRound': expected_round,

            # Seed information
            'Team1Seed': team1_seed_num,
            'Team2Seed': team2_seed_num,
            'SeedDiff': team1_seed_num - team2_seed_num,
            'SeedSum': team1_seed_num + team2_seed_num,

            # Basic team performance
            'Team1WinRate': team1_profile['Season_Win'] if 'Season_Win' in team1_profile else 0.5,
            'Team2WinRate': team2_profile['Season_Win'] if 'Season_Win' in team2_profile else 0.5,
            'WinRateDiff': (team1_profile['Season_Win'] if 'Season_Win' in team1_profile else 0.5) -
                          (team2_profile['Season_Win'] if 'Season_Win' in team2_profile else 0.5),

            # Offensive and defensive efficiency
            'Team1OffEfficiency': team1_profile['Season_OffEfficiency'] if 'Season_OffEfficiency' in team1_profile else 0,
            'Team2OffEfficiency': team2_profile['Season_OffEfficiency'] if 'Season_OffEfficiency' in team2_profile else 0,
            'OffEfficiencyDiff': (team1_profile['Season_OffEfficiency'] if 'Season_OffEfficiency' in team1_profile else 0) -
                               (team2_profile['Season_OffEfficiency'] if 'Season_OffEfficiency' in team2_profile else 0),

            'Team1DefEfficiency': team1_profile['Season_DefEfficiency'] if 'Season_DefEfficiency' in team1_profile else 0,
            'Team2DefEfficiency': team2_profile['Season_DefEfficiency'] if 'Season_DefEfficiency' in team2_profile else 0,
            'DefEfficiencyDiff': (team1_profile['Season_DefEfficiency'] if 'Season_DefEfficiency' in team1_profile else 0) -
                               (team2_profile['Season_DefEfficiency'] if 'Season_DefEfficiency' in team2_profile else 0),

            'Team1NetEfficiency': team1_profile['Season_NetEfficiency'] if 'Season_NetEfficiency' in team1_profile else 0,
            'Team2NetEfficiency': team2_profile['Season_NetEfficiency'] if 'Season_NetEfficiency' in team2_profile else 0,
            'NetEfficiencyDiff': (team1_profile['Season_NetEfficiency'] if 'Season_NetEfficiency' in team1_profile else 0) -
                               (team2_profile['Season_NetEfficiency'] if 'Season_NetEfficiency' in team2_profile else 0),

            # Momentum features
            'Team1WinRate_Last3': team1_momentum_vals['Win_Last3'],
            'Team2WinRate_Last3': team2_momentum_vals['Win_Last3'],
            'WinRateLast3_Diff': team1_momentum_vals['Win_Last3'] - team2_momentum_vals['Win_Last3'],

            'Team1ScoreMargin_Last5': team1_momentum_vals['ScoreMargin_Last5'],
            'Team2ScoreMargin_Last5': team2_momentum_vals['ScoreMargin_Last5'],
            'ScoreMarginLast5_Diff': team1_momentum_vals['ScoreMargin_Last5'] - team2_momentum_vals['ScoreMargin_Last5'],

            'Team1WinRate_Exp': team1_momentum_vals['Win_Exp'],
            'Team2WinRate_Exp': team2_momentum_vals['Win_Exp'],
            'WinRateExp_Diff': team1_momentum_vals['Win_Exp'] - team2_momentum_vals['Win_Exp'],

            # Strength of schedule
            'Team1SOS': team1_sos_vals['AvgOpponentWinRate'],
            'Team2SOS': team2_sos_vals['AvgOpponentWinRate'],
            'SOSDiff': team1_sos_vals['AvgOpponentWinRate'] - team2_sos_vals['AvgOpponentWinRate'],

            'Team1SOSPercentile': team1_sos_vals['SOSPercentile'],
            'Team2SOSPercentile': team2_sos_vals['SOSPercentile'],
            'SOSPercentileDiff': team1_sos_vals['SOSPercentile'] - team2_sos_vals['SOSPercentile'],

            'Team1WinVsGoodTeams': team1_sos_vals['WinRateVsWinningTeams'],
            'Team2WinVsGoodTeams': team2_sos_vals['WinRateVsWinningTeams'],
            'WinVsGoodTeamsDiff': team1_sos_vals['WinRateVsWinningTeams'] - team2_sos_vals['WinRateVsWinningTeams'],

            # Conference strength
            'Team1ConfWinRate': team1_conf_win_rate,
            'Team2ConfWinRate': team2_conf_win_rate,
            'ConfWinRateDiff': team1_conf_win_rate - team2_conf_win_rate,

            'Team1ConfWinsPerTeam': team1_conf_wins_per_team,
            'Team2ConfWinsPerTeam': team2_conf_wins_per_team,
            'ConfWinsPerTeamDiff': team1_conf_wins_per_team - team2_conf_wins_per_team,

            # Coach experience
            'Team1CoachExp': team1_coach_exp,
            'Team2CoachExp': team2_coach_exp,
            'CoachExpDiff': team1_coach_exp - team2_coach_exp,

            'Team1CoachTourneyExp': team1_coach_tourney_exp,
            'Team2CoachTourneyExp': team2_coach_tourney_exp,
            'CoachTourneyExpDiff': team1_coach_tourney_exp - team2_coach_tourney_exp,

            'Team1CoachChampionships': team1_coach_champ,
            'Team2CoachChampionships': team2_coach_champ,
            'CoachChampionshipsDiff': team1_coach_champ - team2_coach_champ,

            # Tournament history
            'Team1TourneyAppearances': team1_appearances,
            'Team2TourneyAppearances': team2_appearances,
            'TourneyAppearancesDiff': team1_appearances - team2_appearances,

            'Team1TourneyWinRate': team1_tourney_win_rate,
            'Team2TourneyWinRate': team2_tourney_win_rate,
            'TourneyWinRateDiff': team1_tourney_win_rate - team2_tourney_win_rate,

            'Team1Championships': team1_championships,
            'Team2Championships': team2_championships,
            'ChampionshipsDiff': team1_championships - team2_championships,

            # Close game performance
            'Team1CloseGameWinRate': team1_consistency.get('CloseGame_WinRate', 0.5),
            'Team2CloseGameWinRate': team2_consistency.get('CloseGame_WinRate', 0.5),
            'CloseGameWinRateDiff': team1_consistency.get('CloseGame_WinRate', 0.5) - team2_consistency.get('CloseGame_WinRate', 0.5),

            # Team consistency metrics
            'Team1ScoreConsistency': team1_consistency.get('Score_CV', 0.1),
            'Team2ScoreConsistency': team2_consistency.get('Score_CV', 0.1),
            'ScoreConsistencyDiff': team1_consistency.get('Score_CV', 0.1) - team2_consistency.get('Score_CV', 0.1),

            'Team1MarginConsistency': team1_consistency.get('Margin_CV', 0.2),
            'Team2MarginConsistency': team2_consistency.get('Margin_CV', 0.2),
            'MarginConsistencyDiff': team1_consistency.get('Margin_CV', 0.2) - team2_consistency.get('Margin_CV', 0.2),

            # Style matchup metrics
            'Team1PaceAdvantage': style_matchup['PaceAdvantage'],
            'Team1ShootingStyleAdvantage': style_matchup['ShootingStyleAdvantage'],
            'Team1DefensiveEdge': style_matchup['DefensiveEdge'],

            # Pressure performance
            'Team1PressureScore': team1_pressure.get('PressureScore', 0.5),
            'Team2PressureScore': team2_pressure.get('PressureScore', 0.5),
            'PressureScoreDiff': team1_pressure.get('PressureScore', 0.5) - team2_pressure.get('PressureScore', 0.5),

            # Tournament readiness
            'Team1TournamentReadiness': team1_readiness['TournamentReadiness'],
            'Team2TournamentReadiness': team2_readiness['TournamentReadiness'],
            'TournamentReadinessDiff': team1_readiness['TournamentReadiness'] - team2_readiness['TournamentReadiness'],

            # Round-specific performance
            'Team1RoundWinRate': team1_round_perf['RoundWinRate'],
            'Team2RoundWinRate': team2_round_perf['RoundWinRate'],
            'RoundWinRateDiff': team1_round_perf['RoundWinRate'] - team2_round_perf['RoundWinRate'],

            # Conference tournament impact
            'Team1ConfTourneyWinRate': team1_conf_impact.get('ConfTourney_WinRate', 0.5),
            'Team2ConfTourneyWinRate': team2_conf_impact.get('ConfTourney_WinRate', 0.5),
            'ConfTourneyWinRateDiff': team1_conf_impact.get('ConfTourney_WinRate', 0.5) - team2_conf_impact.get('ConfTourney_WinRate', 0.5),

            'Team1ConfChampion': team1_conf_impact.get('ConfChampion', 0),
            'Team2ConfChampion': team2_conf_impact.get('ConfChampion', 0),
            'ConfChampionDiff': team1_conf_impact.get('ConfChampion', 0) - team2_conf_impact.get('ConfChampion', 0),

            'Team1MomentumChange': team1_conf_impact.get('MomentumChange', 0),
            'Team2MomentumChange': team2_conf_impact.get('MomentumChange', 0),
            'MomentumChangeDiff': team1_conf_impact.get('MomentumChange', 0) - team2_conf_impact.get('MomentumChange', 0),

            # Seed trend features
            'Team1HistoricalAvgSeed': team1_seed_trends.get('HistoricalAvgSeed', team1_seed_num),
            'Team2HistoricalAvgSeed': team2_seed_trends.get('HistoricalAvgSeed', team2_seed_num),
            'HistoricalAvgSeedDiff': team1_seed_trends.get('HistoricalAvgSeed', team1_seed_num) - team2_seed_trends.get('HistoricalAvgSeed', team2_seed_num),

            'Team1SeedTrend': team1_seed_trends.get('SeedTrend', 0),
            'Team2SeedTrend': team2_seed_trends.get('SeedTrend', 0),
            'SeedTrendDiff': team1_seed_trends.get('SeedTrend', 0) - team2_seed_trends.get('SeedTrend', 0),

            'Team1AvgRoundPerformance': team1_seed_trends.get('AvgRoundPerformance', 0),
            'Team2AvgRoundPerformance': team2_seed_trends.get('AvgRoundPerformance', 0),
            'AvgRoundPerformanceDiff': team1_seed_trends.get('AvgRoundPerformance', 0) - team2_seed_trends.get('AvgRoundPerformance', 0),

            # Coach tournament metrics
            'Team1CoachTourneyWinRate': team1_coach_tourney.get('PriorTourneyWinRate', 0.5),
            'Team2CoachTourneyWinRate': team2_coach_tourney.get('PriorTourneyWinRate', 0.5),
            'CoachTourneyWinRateDiff': team1_coach_tourney.get('PriorTourneyWinRate', 0.5) - team2_coach_tourney.get('PriorTourneyWinRate', 0.5),

            'Team1CoachUpsetRate': team1_coach_tourney.get('UpsetRate', 0.5),
            'Team2CoachUpsetRate': team2_coach_tourney.get('UpsetRate', 0.5),
            'CoachUpsetRateDiff': team1_coach_tourney.get('UpsetRate', 0.5) - team2_coach_tourney.get('UpsetRate', 0.5),

            'Team1CoachUpsetDefenseRate': team1_coach_tourney.get('UpsetDefenseRate', 0.5),
            'Team2CoachUpsetDefenseRate': team2_coach_tourney.get('UpsetDefenseRate', 0.5),
            'CoachUpsetDefenseRateDiff': team1_coach_tourney.get('UpsetDefenseRate', 0.5) - team2_coach_tourney.get('UpsetDefenseRate', 0.5),

            'Team1CoachRoundWinRate': team1_coach_tourney.get('RoundWinRate', 0.5),
            'Team2CoachRoundWinRate': team2_coach_tourney.get('RoundWinRate', 0.5),
            'CoachRoundWinRateDiff': team1_coach_tourney.get('RoundWinRate', 0.5) - team2_coach_tourney.get('RoundWinRate', 0.5)
        }

        # Add core team metrics
        for stat in ['Season_FGPct', 'Season_FG3Pct', 'Season_FTPct', 'Season_eFGPct',
                    'Season_ASTtoTOV', 'Season_OR', 'Season_DR',
                    'Season_ScoreMargin', 'WinStreak']:
            if stat in team1_profile and stat in team2_profile:
                feature_name = stat.replace('Season_', '')
                feature_dict[f'Team1{feature_name}'] = team1_profile[stat]
                feature_dict[f'Team2{feature_name}'] = team2_profile[stat]
                feature_dict[f'{feature_name}Diff'] = team1_profile[stat] - team2_profile[stat]

        # Add additional advanced metrics if available
        for stat in ['Season_FG2Pct', 'Season_TOV_Rate', 'Season_FT_Rate', 'Season_Tempo',
                    'Season_Assist_Rate', 'Season_Steal_Rate', 'Season_Block_Rate']:
            if stat in team1_profile and stat in team2_profile:
                feature_name = stat.replace('Season_', '')
                feature_dict[f'Team1{feature_name}'] = team1_profile[stat]
                feature_dict[f'Team2{feature_name}'] = team2_profile[stat]
                feature_dict[f'{feature_name}Diff'] = team1_profile[stat] - team2_profile[stat]

        # Add percentile ranks if available
        for stat in ['Season_OffEfficiency_Percentile', 'Season_DefEfficiency_Percentile', 'Season_eFGPct_Percentile',
                    'Season_Win_Percentile', 'Season_NetEfficiency_Percentile']:
            if stat in team1_profile and stat in team2_profile:
                feature_name = stat.replace('Season_', '')
                feature_dict[f'Team1{feature_name}'] = team1_profile[stat]
                feature_dict[f'Team2{feature_name}'] = team2_profile[stat]
                feature_dict[f'{feature_name}Diff'] = team1_profile[stat] - team2_profile[stat]

        # Add seed matchup indicators
        if team1_seed_num == 1 and team2_seed_num == 16:
            feature_dict['Seed1v16Matchup'] = 1
        else:
            feature_dict['Seed1v16Matchup'] = 0

        if team1_seed_num == 2 and team2_seed_num == 15:
            feature_dict['Seed2v15Matchup'] = 1
        else:
            feature_dict['Seed2v15Matchup'] = 0

        if team1_seed_num == 8 and team2_seed_num == 9:
            feature_dict['Seed8v9Matchup'] = 1
        else:
            feature_dict['Seed8v9Matchup'] = 0
            
        # Add matchup ID
        feature_dict['MatchupID'] = f"{season_value}_{min(team1_id_value, team2_id_value)}_{max(team1_id_value, team2_id_value)}"

        return feature_dict

    except Exception as e:
        # Enhanced error handling with detailed context
        print(f"Error creating features for matchup {team1_id} vs {team2_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_matchup_features_with_seed_handling(team1_id, team2_id, season, season_profiles, seed_data,
                                          momentum_data, sos_data, coach_features, tourney_history,
                                          conf_strength, team_conferences, team_consistency=None,
                                          team_playstyle=None, round_performance=None, pressure_metrics=None,
                                          conf_impact=None, seed_features=None, coach_metrics=None,
                                          team_to_seed=None):
    """
    Modified version of create_matchup_features_pre_tournament that handles non-tournament teams
    by assigning them default seeds
    """
    # Convert inputs to appropriate types
    season_value = int(season)
    team1_id_value = int(team1_id)
    team2_id_value = int(team2_id)
    
    # Get team profiles from the season-specific profiles
    team1_profile_df = get_data_with_index(season_profiles, 'TeamID', team1_id_value)
    team2_profile_df = get_data_with_index(season_profiles, 'TeamID', team2_id_value)
    
    if len(team1_profile_df) == 0 or len(team2_profile_df) == 0:
        return None  # Skip if either team doesn't have a profile

    # In create_matchup_features_pre_tournament 
    # Initialize team_to_seed as an empty dict if not provided
    team_to_seed = {} if team_to_seed is None else team_to_seed
    
    # Get seed numbers from the provided mapping
    team1_seed_num = team_to_seed.get(team1_id_value, 17) if team_to_seed else 17
    team2_seed_num = team_to_seed.get(team2_id_value, 17) if team_to_seed else 17
    
    # Determine expected round
    expected_round = determine_expected_round(team1_seed_num, team2_seed_num)
    
    # Convert to Series for easier access
    team1_profile = team1_profile_df.iloc[0]
    team2_profile = team2_profile_df.iloc[0]
    
    # Now call the original function with the seed information or build the feature dictionary directly
    # Here I'll build a simplified feature dictionary as an example
    
    # Create a basic feature dictionary
    feature_dict = {
        'Season': season_value,
        'Team1ID': team1_id_value,
        'Team2ID': team2_id_value,
        'ExpectedRound': expected_round,
        'Team1Seed': team1_seed_num,
        'Team2Seed': team2_seed_num,
        'SeedDiff': team1_seed_num - team2_seed_num,
        'SeedSum': team1_seed_num + team2_seed_num,
        
        # Team performance metrics
        'Team1WinRate': team1_profile.get('Season_Win', 0.5),
        'Team2WinRate': team2_profile.get('Season_Win', 0.5),
        'WinRateDiff': team1_profile.get('Season_Win', 0.5) - team2_profile.get('Season_Win', 0.5),
        
        # Offensive and defensive efficiency
        'Team1OffEfficiency': team1_profile.get('Season_OffEfficiency', 100),
        'Team2OffEfficiency': team2_profile.get('Season_OffEfficiency', 100),
        'OffEfficiencyDiff': team1_profile.get('Season_OffEfficiency', 100) - team2_profile.get('Season_OffEfficiency', 100),
        
        'Team1DefEfficiency': team1_profile.get('Season_DefEfficiency', 100),
        'Team2DefEfficiency': team2_profile.get('Season_DefEfficiency', 100),
        'DefEfficiencyDiff': team1_profile.get('Season_DefEfficiency', 100) - team2_profile.get('Season_DefEfficiency', 100),
        
        'Team1NetEfficiency': team1_profile.get('Season_NetEfficiency', 0),
        'Team2NetEfficiency': team2_profile.get('Season_NetEfficiency', 0),
        'NetEfficiencyDiff': team1_profile.get('Season_NetEfficiency', 0) - team2_profile.get('Season_NetEfficiency', 0)
    }
    
    # Add shooting metrics if available
    for stat in ['FGPct', 'FG3Pct', 'FTPct', 'eFGPct']:
        col = f'Season_{stat}'
        if col in team1_profile and col in team2_profile:
            feature_dict[f'Team1{stat}'] = team1_profile[col]
            feature_dict[f'Team2{stat}'] = team2_profile[col]
            feature_dict[f'{stat}Diff'] = team1_profile[col] - team2_profile[col]
    
    # Add matchup ID
    feature_dict['MatchupID'] = f"{season_value}_{min(team1_id_value, team2_id_value)}_{max(team1_id_value, team2_id_value)}"
    
    return feature_dict

def create_tournament_prediction_dataset(seasons, team_profiles, seed_data, momentum_data,
                                     sos_data, coach_features, tourney_history, conf_strength,
                                     team_conferences, team_consistency=None, team_playstyle=None,
                                     round_performance=None, pressure_metrics=None,
                                     conf_impact=None, seed_features=None, coach_metrics=None):
    """
    Create prediction dataset for ALL possible team matchups in given seasons
    """
    all_matchup_features = []

    # Debug information about input data
    print(f"DEBUG: Team profiles shape: {team_profiles.shape} with seasons {sorted(team_profiles['Season'].unique())}")
    print(f"DEBUG: Seed data shape: {seed_data.shape} with seasons {sorted(seed_data['Season'].unique())}")

    for season in seasons:
        print(f"Creating prediction dataset for season {season}")

        # Get teams from seed data for this season
        season_seed_data = seed_data[seed_data['Season'] == season]
        tournament_teams = season_seed_data['TeamID'].unique() if not season_seed_data.empty else []
        print(f"Found {len(tournament_teams)} tournament teams in seed data for season {season}")

        # Check if we have team profiles for this season
        season_profiles = team_profiles[team_profiles['Season'] == season]
        if len(season_profiles) == 0:
            print(f"No team profiles found for season {season} - creating profiles from seed data")
            
            # If we have regular season data for these teams, we could create proper profiles
            # For now, we'll create minimal placeholder profiles for tournament teams
            placeholder_profiles = []
            
            for team_id in tournament_teams:
                # Create a basic placeholder profile with default values
                placeholder_profile = {
                    'Season': season,
                    'TeamID': team_id,
                    'Season_Win': 0.65,  # Tournament team should have decent win rate
                    'Season_OffEfficiency': 100.0,  # Average offensive efficiency
                    'Season_DefEfficiency': 95.0,   # Average defensive efficiency
                    'Season_NetEfficiency': 5.0,    # Net positive efficiency
                    'Season_FGPct': 0.45,           # Average shooting
                    'Season_FG3Pct': 0.35,          # Average 3pt shooting
                    'Season_eFGPct': 0.50,          # Average effective FG%
                    'Season_FTPct': 0.72,           # Average free throw
                    'GamesPlayed': 30,              # Default games played
                    'WinStreak': 1,                 # Neutral win streak
                    'CloseGameWinRate': 0.5,        # Average in close games
                }
                
                # Add team seed as a feature
                team_seed_row = season_seed_data[season_seed_data['TeamID'] == team_id]
                if len(team_seed_row) > 0:
                    seed_str = team_seed_row['Seed'].values[0]
                    seed_num = extract_seed_number(seed_str)
                    
                    # Better seeds get better baseline stats
                    seed_factor = max(0.4, 1.0 - (seed_num/20))
                    placeholder_profile['Season_Win'] = 0.5 + (0.3 * seed_factor)
                    placeholder_profile['Season_OffEfficiency'] = 95 + (15 * seed_factor)
                    placeholder_profile['Season_DefEfficiency'] = 100 - (10 * seed_factor)
                    placeholder_profile['Season_NetEfficiency'] = placeholder_profile['Season_OffEfficiency'] - placeholder_profile['Season_DefEfficiency']
                
                placeholder_profiles.append(placeholder_profile)
            
            # Convert placeholder profiles to DataFrame and use it for this season
            if placeholder_profiles:
                season_profiles = pd.DataFrame(placeholder_profiles)
                print(f"Created {len(season_profiles)} placeholder profiles for tournament teams")
            else:
                print(f"ERROR: No tournament teams found for season {season} - skipping")
                continue
        
        # Get ALL teams, including non-tournament teams if wanted
        all_teams = season_profiles['TeamID'].unique()
        print(f"Found {len(all_teams)} total teams in profiles")
        
        # Determine tournament vs non-tournament teams
        tourney_teams = set(tournament_teams)
        non_tourney_teams = set(all_teams) - tourney_teams
        print(f"Tournament teams: {len(tourney_teams)}, Non-tournament teams: {len(non_tourney_teams)}")
        
        # Create seed mapping for all teams
        team_to_seed = {}
        for team_id in all_teams:
            if team_id in tourney_teams:
                team_seed_row = season_seed_data[season_seed_data['TeamID'] == team_id]
                if not team_seed_row.empty:
                    seed_str = team_seed_row['Seed'].values[0]
                    seed_num = extract_seed_number(seed_str)
                    team_to_seed[team_id] = seed_num
                else:
                    team_to_seed[team_id] = 17
            else:
                team_to_seed[team_id] = 17  # Default for non-tournament teams

        # Generate all possible matchups
        matchups = []
        for team1_id in all_teams:
            for team2_id in all_teams:
                if team1_id < team2_id:
                    matchups.append((team1_id, team2_id))

        print(f"Generated {len(matchups)} possible matchups between all teams")

        # Process matchups with error handling
        matchup_count = 0
        error_count = 0

        # Add placeholder data for tournament metrics if needed
        if round_performance is None or len(round_performance[round_performance['Season'] == season]) == 0:
            print(f"Creating placeholder round performance data for season {season}")
            season_round_perf = create_seed_based_features(all_teams, season_seed_data, season)
            if round_performance is None:
                round_performance = season_round_perf
            else:
                round_performance = pd.concat([round_performance, season_round_perf], ignore_index=True)
            
        if pressure_metrics is None or len(pressure_metrics[pressure_metrics['Season'] == season]) == 0:
            print(f"Creating placeholder pressure metrics for season {season}")
            season_pressure = create_seed_based_pressure_metrics(all_teams, season_seed_data, season)
            if pressure_metrics is None:
                pressure_metrics = season_pressure
            else:
                pressure_metrics = pd.concat([pressure_metrics, season_pressure], ignore_index=True)
            
        if seed_features is None or len(seed_features[seed_features['Season'] == season]) == 0:
            print(f"Creating placeholder seed features for season {season}")
            season_seed_features = create_seed_based_trend_features(all_teams, season_seed_data, season)
            if seed_features is None:
                seed_features = season_seed_features
            else:
                seed_features = pd.concat([seed_features, season_seed_features], ignore_index=True)

        for team1_id, team2_id in matchups:
            try:
                feature_dict = create_matchup_features_with_seed_handling(
                    team1_id, team2_id, season, season_profiles, seed_data,
                    momentum_data, sos_data, coach_features, tourney_history,
                    conf_strength, team_conferences, team_consistency, team_playstyle,
                    round_performance, pressure_metrics, conf_impact, seed_features, coach_metrics,
                    team_to_seed=team_to_seed
                )

                if feature_dict:
                    all_matchup_features.append(feature_dict)
                    matchup_count += 1
                    if matchup_count % 5000 == 0:
                        print(f"Successfully created {matchup_count} matchups so far...")
                else:
                    error_count += 1
                    if error_count < 10:
                        print(f"Warning: Empty feature dictionary returned for matchup {team1_id} vs {team2_id}")

            except Exception as e:
                error_count += 1
                if error_count < 20:
                    print(f"Error creating features for matchup {team1_id} vs {team2_id}: {str(e)}")
                elif error_count == 20:
                    print("Too many errors - suppressing further error messages...")

        print(f"Successfully created {matchup_count} matchups, encountered {error_count} errors")

        if matchup_count == 0:
            print(f"ERROR: Failed to create any valid matchups for season {season}")

    if not all_matchup_features:
        print("WARNING: No matchup features were created for any season!")
        empty_df = pd.DataFrame(columns=[
            'Season', 'Team1ID', 'Team2ID', 'ExpectedRound', 'Team1Seed', 'Team2Seed',
            'SeedDiff', 'Team1WinRate', 'Team2WinRate', 'WinRateDiff'
        ])
        return empty_df

    result_df = pd.DataFrame(all_matchup_features)
    print(f"SUCCESS: Created a total of {len(result_df)} matchup features across all seasons")
    return result_df

def create_seed_based_features(all_teams, seed_data, season):
    """
    Create round performance features for all teams
    """
    seed_nums = []
    tournament_teams = set(seed_data[seed_data['Season'] == season]['TeamID'].unique())
    
    for team in all_teams:
        if team in tournament_teams:
            team_seed_row = seed_data[(seed_data['Season'] == season) & (seed_data['TeamID'] == team)]
            if not team_seed_row.empty:
                seed_num = extract_seed_number(team_seed_row['Seed'].iloc[0])
            else:
                seed_num = 17
        else:
            seed_num = 17  # Default for non-tournament teams
            
        seed_nums.append(seed_num)

    # Calculate performance metrics based on seeds
    win_rates = []
    margins = []
    
    for seed_num in seed_nums:
        if seed_num <= 16:  # Tournament team
            win_rate = max(0.95 - (seed_num-1)*0.05, 0.3)
            margin = 10 - (seed_num-1)*0.6
        else:  # Non-tournament team
            win_rate = 0.25  # Below average
            margin = 0      # Even margin
            
        win_rates.append(win_rate)
        margins.append(margin)

    return pd.DataFrame({
        'Season': [season] * len(all_teams),
        'TeamID': all_teams,
        'WinRate_Round64': win_rates,
        'Games_Round64': [5] * len(all_teams),  # Placeholder value
        'Margin_Round64': margins
    })

def create_seed_based_pressure_metrics(all_teams, seed_data, season):
    """
    Create pressure performance metrics for all teams
    Non-tournament teams get lower pressure metrics
    """
    results = []
    tournament_teams = set(seed_data[seed_data['Season'] == season]['TeamID'].unique())

    for team in all_teams:
        is_tournament_team = team in tournament_teams
        
        if is_tournament_team:
            team_seed_row = seed_data[(seed_data['Season'] == season) & (seed_data['TeamID'] == team)]
            if len(team_seed_row) > 0:
                seed_num = extract_seed_number(team_seed_row['Seed'].iloc[0])
            else:
                seed_num = 17
        else:
            seed_num = 17  # Default for non-tournament teams

        # Different metrics for tournament vs. non-tournament teams
        if not is_tournament_team:
            close_win_rate = 0.38  # Below average
            upset_win_rate = 0.4   # Below average
            upset_defense_rate = 0.4  # Below average
            late_rounds_rate = 0.2  # Very low
        else:
            close_win_rate = max(0.75 - (seed_num-1)*0.025, 0.4)
            upset_win_rate = 0.5  # Default for all seeds
            upset_defense_rate = max(0.85 - (seed_num-1)*0.03, 0.5)
            late_rounds_rate = max(0.8 - (seed_num-1)*0.05, 0.3)

        pressure_score = (0.4 * close_win_rate +
                          0.3 * upset_defense_rate +
                          0.3 * late_rounds_rate)

        results.append({
            'Season': season,
            'TeamID': team,
            'CloseGames_Count': 3,  # Default placeholder
            'CloseGames_WinRate': close_win_rate,
            'UpsetOpps_Count': 2,  # Default placeholder
            'UpsetOpps_WinRate': upset_win_rate,
            'LateRounds_Count': 2,  # Default placeholder
            'LateRounds_WinRate': late_rounds_rate,
            'PressureScore': pressure_score
        })

    return pd.DataFrame(results)

def create_seed_based_trend_features(all_teams, seed_data, season):
    """
    Create seed trend features for all teams
    """
    results = []
    tournament_teams = set(seed_data[seed_data['Season'] == season]['TeamID'].unique())

    for team in all_teams:
        if team in tournament_teams:
            team_seed_row = seed_data[(seed_data['Season'] == season) & (seed_data['TeamID'] == team)]
            if not team_seed_row.empty:
                seed_num = extract_seed_number(team_seed_row['Seed'].iloc[0])
            else:
                seed_num = 17
        else:
            seed_num = 17  # Default for non-tournament teams

        # Different metrics based on tournament status
        if seed_num > 16:
            avg_round_perf = -0.5  # Slightly negative performance
            prior_appearances = 0
        else:
            avg_round_perf = 0  # Neutral for first-time teams
            prior_appearances = 1

        results.append({
            'Season': season,
            'TeamID': team,
            'HistoricalAvgSeed': seed_num,
            'BestHistoricalSeed': seed_num,
            'WorstHistoricalSeed': seed_num,
            'PriorAppearances': prior_appearances,
            'SeedTrend': 0,  # No trend for first appearance
            'AvgRoundPerformance': avg_round_perf
        })

    return pd.DataFrame(results)

def create_upset_specific_features(X, gender="men's"):
    """Enhanced version of create_upset_specific_features with more predictive features."""
    X_enhanced = X.copy()
    
    # Core seed difference transformations
    if 'SeedDiff' in X.columns:
        # Transform seed difference to capture non-linear upset potential
        X_enhanced['SeedDiffLog'] = np.log(abs(X['SeedDiff']) + 1) * np.sign(X['SeedDiff'])
        X_enhanced['SeedDiffSquared'] = X['SeedDiff'] ** 2 * np.sign(X['SeedDiff'])
        
        # Specific upset zone markers for men's tournament
        X_enhanced['UpsetZone_MajorUpsetsV1'] = (
            ((X['Team1Seed'] == 12) & (X['Team2Seed'] == 5)) |  # 12 vs 5 upset
            ((X['Team1Seed'] == 11) & (X['Team2Seed'] == 6)) |  # 11 vs 6 upset
            ((X['Team1Seed'] == 10) & (X['Team2Seed'] == 7))    # 10 vs 7 upset
        ).astype(int)
        
        X_enhanced['UpsetZone_MinorUpsetsV1'] = (
            ((X['Team1Seed'] == 9) & (X['Team2Seed'] == 8))    # 9 vs 8 upset
        ).astype(int)
    
    # Momentum interactions specific to men's upset potential
    momentum_col = next((col for col in X.columns if col.startswith('Team1Win') and 'Last' in col), None)
    if momentum_col and 'SeedDiff' in X.columns:
        # Create an interaction term that boosts underdog momentum
        X_enhanced['UnderdogMomentumBoost'] = np.where(
            X['SeedDiff'] > 0,  # Team1 is underdog
            X[momentum_col] * np.sqrt(abs(X['SeedDiff'])),
            X[momentum_col.replace('Team1', 'Team2')] * np.sqrt(abs(X['SeedDiff']))
        )
    
    # 3-point shooting upset factor (critical in men's basketball)
    if all(col in X.columns for col in ['Team1FG3Pct', 'Team2FG3Pct', 'SeedDiff']):
        X_enhanced['ThreePointUpsetFactor'] = (
            (X['Team1FG3Pct'] - X['Team2FG3Pct']) * 
            np.sign(X['SeedDiff']) * 
            np.log(abs(X['SeedDiff']) + 1)
        )
    
    # Defensive performance in upset potential
    if all(col in X.columns for col in ['Team1DefEfficiency', 'Team2DefEfficiency']):
        X_enhanced['DefensiveUpsetFactor'] = np.where(
            X['SeedDiff'] > 0,  # Team1 is underdog
            X['Team2DefEfficiency'] - X['Team1DefEfficiency'],
            X['Team1DefEfficiency'] - X['Team2DefEfficiency']
        )
    
    # Add men's specific features
    if gender == "men's":
        # More conservative approach that checks for columns before using them
        required_cols = ['Team1Seed', 'Team2Seed', 'Team1WinRate', 'Team2WinRate',
                        'Team1DefEfficiency', 'Team2DefEfficiency', 'Team1FG3Pct']
        
        if all(col in X.columns for col in required_cols):
            # Basic conditions for upsets that don't require missing columns
            X_enhanced['HighPrecisionUpset'] = (
                # 11-seed over 6-seed with good 3-point shooting
                ((X['Team1Seed'] == 11) & (X['Team2Seed'] == 6) & 
                (X['Team1FG3Pct'] > 0.36) & (X['Team1WinRate'] > 0.6)) |
                # 12-seed over 5-seed with basic conditions
                ((X['Team1Seed'] == 12) & (X['Team2Seed'] == 5) & 
                (X['Team1DefEfficiency'] < X['Team2DefEfficiency'])) |
                # 10-seed over 7-seed with defensive edge
                ((X['Team1Seed'] == 10) & (X['Team2Seed'] == 7) & 
                (X['Team1DefEfficiency'] < X['Team2DefEfficiency'] - 2) & (X['Team1WinRate'] > 0.65))
            ).astype(int)
        else:
            # Fallback with simpler conditions if missing required columns
            X_enhanced['HighPrecisionUpset'] = X_enhanced.get('UpsetZone_MajorUpsetsV1', 0)
    
    # Add women's specific features
    if gender == "women's":
        if 'Team1Seed' in X.columns and 'Team2Seed' in X.columns:
            # Women's tournament specific upset zones
            X_enhanced['WomensUpsetZone_Major'] = (
                ((X['Team1Seed'] == 10) & (X['Team2Seed'] == 7)) |
                ((X['Team1Seed'] == 11) & (X['Team2Seed'] == 6)) |
                ((X['Team1Seed'] == 12) & (X['Team2Seed'] == 5))
            ).astype(int)
    
    # Composite upset likelihood score
    upset_indicators = []
    
    # Add upset indicators with weights
    if 'UnderdogMomentumBoost' in X_enhanced.columns:
        upset_indicators.append(
            X_enhanced['UnderdogMomentumBoost'] * 0.35
        )
    
    if 'ThreePointUpsetFactor' in X_enhanced.columns:
        upset_indicators.append(
            np.where(X_enhanced['ThreePointUpsetFactor'] > 0, 
                    X_enhanced['ThreePointUpsetFactor'] * 0.3, 0)
        )
    
    if 'DefensiveUpsetFactor' in X_enhanced.columns:
        upset_indicators.append(
            np.where(X_enhanced['DefensiveUpsetFactor'] > 0, 
                    X_enhanced['DefensiveUpsetFactor'] * 0.2, 0)
        )
    
    if 'UpsetZone_MajorUpsetsV1' in X_enhanced.columns:
        upset_indicators.append(
            X_enhanced['UpsetZone_MajorUpsetsV1'] * 0.15
        )
    
    if 'HighPrecisionUpset' in X_enhanced.columns:
        upset_indicators.append(
            X_enhanced['HighPrecisionUpset'] * 0.35  # Higher weight for high-precision predictions
        )
    
    if gender == "women's" and 'WomensUpsetZone_Major' in X_enhanced.columns:
        upset_indicators.append(
            X_enhanced['WomensUpsetZone_Major'] * 0.25
        )
    
    # Combine indicators
    if upset_indicators:
        X_enhanced['CompositeUpsetScore'] = sum(upset_indicators)
    
    return X_enhanced
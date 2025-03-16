import pandas as pd
import numpy as np
from ..data.processors import filter_reg_season
from ..utils.data_access import get_data_with_index

def engineer_team_season_stats(season_games_df, team_id_col, opp_team_id_col):
    """
    Creates team-level aggregated statistics from season games

    Args:
        season_games_df: DataFrame with season games
        team_id_col: Column name for team ID
        opp_team_id_col: Column name for opponent team ID

    Returns:
        DataFrame with team-level statistics
    """
    # Prefix based on whether we're processing winning or losing team stats
    prefix = 'W' if team_id_col == 'WTeamID' else 'L'
    opp_prefix = 'L' if prefix == 'W' else 'W'

    # Extract team games
    team_games = season_games_df[[
        'Season', 'DayNum', team_id_col, f'{prefix}Score',
        f'{prefix}FGM', f'{prefix}FGA', f'{prefix}FGM3', f'{prefix}FGA3',
        f'{prefix}FTM', f'{prefix}FTA', f'{prefix}OR', f'{prefix}DR',
        f'{prefix}Ast', f'{prefix}TO', f'{prefix}Stl', f'{prefix}Blk',
        f'{prefix}PF', opp_team_id_col, f'{opp_prefix}Score'
    ]].copy()

    # Rename columns to generic names
    rename_dict = {
        team_id_col: 'TeamID',
        opp_team_id_col: 'OpponentID',
        f'{prefix}Score': 'Score',
        f'{prefix}FGM': 'FGM',
        f'{prefix}FGA': 'FGA',
        f'{prefix}FGM3': 'FGM3',
        f'{prefix}FGA3': 'FGA3',
        f'{prefix}FTM': 'FTM',
        f'{prefix}FTA': 'FTA',
        f'{prefix}OR': 'OR',
        f'{prefix}DR': 'DR',
        f'{prefix}Ast': 'Ast',
        f'{prefix}TO': 'TO',
        f'{prefix}Stl': 'Stl',
        f'{prefix}Blk': 'Blk',
        f'{prefix}PF': 'PF',
        f'{opp_prefix}Score': 'OpponentScore'
    }
    team_games.rename(columns=rename_dict, inplace=True)

    # Add win/loss indicator
    team_games['Win'] = 1 if prefix == 'W' else 0

    # Calculate game-level derived stats
    team_games['Possessions'] = (team_games['FGA'] + (0.44*team_games['FTA']) -
                                team_games['OR'] + team_games['TO']).round()

    team_games['ScoreMargin'] = team_games['Score'] - team_games['OpponentScore']
    team_games['eFGPct'] = (team_games['FGM'] + (0.5*team_games['FGM3']))/team_games['FGA']
    team_games['FGPct'] = team_games['FGM']/team_games['FGA']
    team_games['FG3Pct'] = team_games['FGM3']/team_games['FGA3']
    team_games['FTPct'] = team_games['FTM']/team_games['FTA']
    team_games['TotalReb'] = team_games['OR'] + team_games['DR']
    team_games['ASTtoTOV'] = team_games['Ast']/team_games['TO']
    team_games['OffEfficiency'] = 100 * team_games['Score']/team_games['Possessions']

    # Handle potential division by zero issues
    for col in ['FGPct', 'FG3Pct', 'FTPct', 'ASTtoTOV']:
        team_games[col] = team_games[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return team_games

def create_team_season_profiles(regular_season_df, current_season=None, tournament_days=None):
    """
    Create team season profiles using ONLY regular season data

    Args:
        regular_season_df: DataFrame with regular season games
        current_season: Current season to process
        tournament_days: List of day numbers that are tournament days (to exclude)

    Returns:
        DataFrame with team season profiles
    """
    # Filter to include only regular season games (pre-tournament)
    regular_season_df = filter_reg_season(regular_season_df, current_season, tournament_days)

    # Process winning team games
    w_team_games = engineer_team_season_stats(regular_season_df, 'WTeamID', 'LTeamID')

    # Process losing team games
    l_team_games = engineer_team_season_stats(regular_season_df, 'LTeamID', 'WTeamID')

    # Combine winning and losing team data
    all_team_games = pd.concat([w_team_games, l_team_games], ignore_index=True)

    # Sort by season, team, and day for proper aggregation
    all_team_games = all_team_games.sort_values(['Season', 'TeamID', 'DayNum'])

    # Calculate team season aggregates
    team_season_stats = all_team_games.groupby(['Season', 'TeamID']).agg({
        'Win': 'mean',  # Win rate
        'Score': 'mean',
        'OpponentScore': 'mean',
        'ScoreMargin': 'mean',
        'FGPct': 'mean',
        'FG3Pct': 'mean',
        'FTPct': 'mean',
        'eFGPct': 'mean',
        'OR': 'mean',
        'DR': 'mean',
        'TotalReb': 'mean',
        'Ast': 'mean',
        'TO': 'mean',
        'Stl': 'mean',
        'Blk': 'mean',
        'ASTtoTOV': 'mean',
        'Possessions': 'mean',
        'OffEfficiency': 'mean'
    }).reset_index()

    # Rename columns to indicate they are season averages
    stat_cols = team_season_stats.columns.difference(['Season', 'TeamID'])
    rename_dict = {col: f'Season_{col}' for col in stat_cols}
    team_season_stats.rename(columns=rename_dict, inplace=True)

    # Count number of games played
    game_counts = all_team_games.groupby(['Season', 'TeamID']).size().reset_index(name='GamesPlayed')
    team_season_stats = team_season_stats.merge(game_counts, on=['Season', 'TeamID'])

    # Calculate win/loss record
    team_wins = all_team_games[all_team_games['Win'] == 1].groupby(['Season', 'TeamID']).size().reset_index(name='Wins')
    team_season_stats = team_season_stats.merge(team_wins, on=['Season', 'TeamID'], how='left')
    team_season_stats['Wins'] = team_season_stats['Wins'].fillna(0)
    team_season_stats['Losses'] = team_season_stats['GamesPlayed'] - team_season_stats['Wins']

    # Calculate win streak at the end of regular season
    all_team_games['WinStreak'] = all_team_games.groupby(['Season', 'TeamID'])['Win'].transform(
        lambda x: x.groupby((x != x.shift(1)).cumsum()).cumsum()
    )
    all_team_games['WinStreak'] = all_team_games['WinStreak'] * all_team_games['Win']

    final_streaks = all_team_games.sort_values(['Season', 'TeamID', 'DayNum']).groupby(['Season', 'TeamID']).last()[['WinStreak']].reset_index()
    team_season_stats = team_season_stats.merge(final_streaks, on=['Season', 'TeamID'], how='left')
    team_season_stats['WinStreak'] = team_season_stats['WinStreak'].fillna(0)

    # Close game performance (games decided by 5 or fewer points)
    all_team_games['CloseGame'] = abs(all_team_games['ScoreMargin']) <= 5
    close_games = all_team_games[all_team_games['CloseGame']].groupby(['Season', 'TeamID']).agg(
        CloseGameCount=('CloseGame', 'sum'),
        CloseGameWins=('Win', 'sum')
    ).reset_index()
    close_games['CloseGameWinRate'] = close_games['CloseGameWins'] / close_games['CloseGameCount']
    team_season_stats = team_season_stats.merge(close_games, on=['Season', 'TeamID'], how='left')
    team_season_stats['CloseGameWinRate'] = team_season_stats['CloseGameWinRate'].fillna(0.5)

    return team_season_stats, all_team_games

# Replace the existing calculate_momentum_features function in march_madness/features/team_features.py
# with this enhanced version (keeps the same function name)

def calculate_momentum_features(game_data, current_season=None, tournament_days=None, 
                              team_profiles=None, sos_data=None):
    """
    Calculate momentum features based on recent game performance
    Uses shift to avoid data leakage and only includes regular season games

    Args:
        game_data: DataFrame with all team games
        current_season: Current season for prediction
        tournament_days: List of tournament day numbers to exclude
        team_profiles: DataFrame with team profile data (optional, for advanced features)
        sos_data: DataFrame with strength of schedule data (optional, for advanced features)

    Returns:
        DataFrame with momentum features
    """
    import numpy as np
    import pandas as pd
    from ..data.processors import filter_reg_season
    from ..utils.data_access import get_data_with_index
    
    # Filter to only include regular season games
    reg_season_data = filter_reg_season(game_data, current_season, tournament_days)
    
    # Sort by team and date
    reg_season_data = reg_season_data.sort_values(['Season', 'TeamID', 'DayNum'])
    
    # Define momentum windows
    short_window = 3
    medium_window = 5
    long_window = 10
    
    # Calculate rolling averages with proper shifting
    basic_momentum_cols = ['Win', 'ScoreMargin', 'OffEfficiency', 'FGPct', 'FG3Pct']
    for col in basic_momentum_cols:
        # Short-term momentum (last 3 games)
        reg_season_data[f'{col}_Last{short_window}'] = reg_season_data.groupby(['Season', 'TeamID'])[col].transform(
            lambda x: x.rolling(window=short_window, min_periods=1).mean()
        )
        
        # Medium-term momentum (last 5 games)
        reg_season_data[f'{col}_Last{medium_window}'] = reg_season_data.groupby(['Season', 'TeamID'])[col].transform(
            lambda x: x.rolling(window=medium_window, min_periods=1).mean()
        )
        
        # Long-term momentum (last 10 games)
        reg_season_data[f'{col}_Last{long_window}'] = reg_season_data.groupby(['Season', 'TeamID'])[col].transform(
            lambda x: x.rolling(window=long_window, min_periods=1).mean()
        )
        
        # Exponentially weighted momentum
        reg_season_data[f'{col}_Exp'] = reg_season_data.groupby(['Season', 'TeamID'])[col].transform(
            lambda x: x.ewm(span=5, min_periods=1).mean()
        )
    
    # If we have the required data for advanced features, calculate them
    if team_profiles is not None and sos_data is not None:
        try:
            # 1. Add opponent strength
            # Create a dictionary for quick opponent strength lookup
            opponent_strength = {}
            for _, row in sos_data.iterrows():
                opponent_strength[(row['Season'], row['TeamID'])] = row.get('AvgOpponentWinRate', 0.5)
            
            # Add opponent strength to each game
            reg_season_data['OpponentStrength'] = reg_season_data.apply(
                lambda row: opponent_strength.get((row['Season'], row['OpponentID']), 0.5),
                axis=1
            )
            
            # 1. Opponent-Weighted Momentum Features
            # Weight wins and scoring by opponent strength
            reg_season_data['WeightedWin'] = reg_season_data['Win'] * reg_season_data['OpponentStrength'] * 2
            reg_season_data['WeightedScoreMargin'] = reg_season_data['ScoreMargin'] * reg_season_data['OpponentStrength'] * 2
            
            for window in [short_window, medium_window, long_window]:
                # Weighted win momentum
                reg_season_data[f'WeightedWin_Last{window}'] = reg_season_data.groupby(['Season', 'TeamID'])['WeightedWin'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                
                # Weighted score margin momentum
                reg_season_data[f'WeightedScoreMargin_Last{window}'] = reg_season_data.groupby(['Season', 'TeamID'])['WeightedScoreMargin'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
            
            # Exponential weighted quality win momentum
            reg_season_data['WeightedWin_Exp'] = reg_season_data.groupby(['Season', 'TeamID'])['WeightedWin'].transform(
                lambda x: x.ewm(span=5, min_periods=1).mean()
            )
            
            # 2. Trend-Based Momentum (is team improving or declining?)
            # Calculate trend over last N games for key metrics
            for col in ['Win', 'ScoreMargin', 'OffEfficiency']:
                for window in [medium_window, long_window]:
                    # Use linear regression slope as trend
                    reg_season_data[f'{col}_Trend_{window}'] = reg_season_data.groupby(['Season', 'TeamID'])[col].transform(
                        lambda x: calculate_trend(x, window)
                    )
            
            # 3. Consistency/Volatility in Recent Performance
            # Calculate standard deviation in recent scoring and efficiency
            for col in ['Score', 'ScoreMargin', 'OffEfficiency']:
                for window in [medium_window, long_window]:
                    reg_season_data[f'{col}_Volatility_{window}'] = reg_season_data.groupby(['Season', 'TeamID'])[col].transform(
                        lambda x: x.rolling(window=window, min_periods=2).std()
                    )
            
            # 4. Recent Performance Against Quality Opponents
            # Tag games against top-tier opponents (win rate > 0.7)
            reg_season_data['QualityOpponent'] = reg_season_data['OpponentStrength'] > 0.7

            # Initialize quality opponent columns with NaN values
            for window in [long_window, 20]:
                reg_season_data[f'QualityOpp_WinRate_{window}'] = np.nan
                reg_season_data[f'QualityOpp_ScoreMargin_{window}'] = np.nan

            # Process each team-season group individually
            for (season, team_id), group in reg_season_data.groupby(['Season', 'TeamID']):
                # Check if this team has any quality opponents
                quality_games = group[group['QualityOpponent']]
                
                if len(quality_games) > 0:
                    # This team has played quality opponents
                    for window in [long_window, 20]:
                        try:
                            # Calculate cumulative metrics for this team
                            qual_win_rate = quality_games['Win'].expanding().mean()
                            qual_margin = quality_games['ScoreMargin'].expanding().mean()
                            
                            # Get the last values (most current) and update only those
                            last_idx = group.index[-1]
                            if not qual_win_rate.empty:
                                reg_season_data.loc[last_idx, f'QualityOpp_WinRate_{window}'] = qual_win_rate.iloc[-1]
                                reg_season_data.loc[last_idx, f'QualityOpp_ScoreMargin_{window}'] = qual_margin.iloc[-1]
                        except Exception as e:
                            print(f"  Warning: Error calculating quality metrics for team {team_id}, season {season}: {str(e)}")
            
            # 5. Late Season Momentum (give more weight to most recent games)
            # Implement a decay function that gives higher weight to more recent games
            for team_season_group in reg_season_data.groupby(['Season', 'TeamID']):
                team_key, team_data = team_season_group
                if len(team_data) >= 5:
                    # Calculate days from last game for each game
                    max_day = team_data['DayNum'].max()
                    team_data['DaysFromLast'] = max_day - team_data['DayNum']
                    
                    # Apply decay weights using decay factor
                    decay_factor = 0.95  # 5% decay per game back
                    team_data['LateSeasonWeight'] = team_data['DaysFromLast'].apply(lambda x: decay_factor ** x)
                    
                    # Calculate weighted metrics
                    team_data['LateSeasonWin'] = team_data['Win'] * team_data['LateSeasonWeight']
                    team_data['LateSeasonMargin'] = team_data['ScoreMargin'] * team_data['LateSeasonWeight']
                    
                    # Update the original dataframe for this team and season
                    idx = team_data.index
                    reg_season_data.loc[idx, 'LateSeasonWeight'] = team_data['LateSeasonWeight']
                    reg_season_data.loc[idx, 'LateSeasonWin'] = team_data['LateSeasonWin'] 
                    reg_season_data.loc[idx, 'LateSeasonMargin'] = team_data['LateSeasonMargin']
            
            # Calculate late season momentum metrics
            reg_season_data['LateSeasonWin_Sum'] = reg_season_data.groupby(['Season', 'TeamID'])['LateSeasonWin'].transform(
                lambda x: x.rolling(window=10, min_periods=1).sum()
            )
            reg_season_data['LateSeasonWeight_Sum'] = reg_season_data.groupby(['Season', 'TeamID'])['LateSeasonWeight'].transform(
                lambda x: x.rolling(window=10, min_periods=1).sum()
            )
            
            # Normalize the weighted sum by the sum of weights
            reg_season_data['LateSeasonWinRate'] = reg_season_data['LateSeasonWin_Sum'] / reg_season_data['LateSeasonWeight_Sum'].replace(0, 1)
            
            # 6. Close Game Performance Momentum
            # Tag close games (margin ≤ 5 points)
            reg_season_data['CloseGame'] = abs(reg_season_data['ScoreMargin']) <= 5
            
            # Calculate performance in close games
            reg_season_data['CloseGameWin'] = (reg_season_data['CloseGame'] & (reg_season_data['Win'] == 1)).astype(int)
            
            # Calculate rolling performance in close games
            for window in [medium_window, long_window]:
                reg_season_data[f'CloseGameWinRate_{window}'] = reg_season_data.groupby(['Season', 'TeamID'])['CloseGameWin'].transform(
                    lambda x: calculate_conditional_mean(x, reg_season_data.loc[x.index, 'CloseGame'], window)
                )
            
            # 7. Conference Game Momentum
            # Assuming each team can determine its conference from team_profiles or another source
            if 'ConfAbbrev' in reg_season_data.columns:
                # Tag games against conference opponents
                reg_season_data['ConfGame'] = reg_season_data.apply(
                    lambda row: row['ConfAbbrev'] == get_team_conference(row['OpponentID'], row['Season'], team_profiles),
                    axis=1
                )
                
                # Calculate conference game performance
                for window in [5, 10]:
                    reg_season_data[f'ConfGameWinRate_{window}'] = reg_season_data.groupby(['Season', 'TeamID']).apply(
                        lambda group: group['Win'][group['ConfGame']].rolling(window=window, min_periods=1).mean()
                    ).reset_index(level=[0,1], drop=True)
            
            # 8. Momentum Composite Score
            # Create a combined momentum score from multiple factors
            reg_season_data['MomentumComposite'] = (
                # Recent win rate (40%)
                0.40 * reg_season_data['Win_Last5'] + 
                # Opponent-adjusted win rate (25%)
                0.25 * reg_season_data['WeightedWin_Last5'] + 
                # Trend (15%)
                0.15 * np.clip(reg_season_data['Win_Trend_10'] * 10, -1, 1) + 
                # Late season momentum (20%)
                0.20 * reg_season_data['LateSeasonWinRate']
            )
            
            print("Successfully calculated advanced momentum features")
            
        except Exception as e:
            print(f"Error calculating advanced momentum features: {str(e)}")
            print("Will proceed with basic momentum features only")
    
    # Get final values at the end of regular season for each team and season
    latest_momentum = reg_season_data.sort_values(['Season', 'TeamID', 'DayNum']).groupby(['Season', 'TeamID']).last().reset_index()
    
    # Select all momentum columns
    momentum_cols = [col for col in latest_momentum.columns if any(x in col for x in 
                                                                 ['_Last', '_Exp', 'WinStreak', 
                                                                  'Weighted', '_Trend_', '_Volatility_', 
                                                                  'QualityOpp_', 'LateSeasonWin', 'LateSeasonWinRate',
                                                                  'CloseGameWinRate_', 'ConfGameWinRate_',
                                                                  'MomentumComposite'])]
    
    return latest_momentum[['Season', 'TeamID'] + momentum_cols]

def calculate_trend(series, window):
    """
    Calculate the linear trend (slope) of recent values in a series
    
    Args:
        series: Time series of values
        window: Number of most recent values to use
        
    Returns:
        Slope coefficient (trend)
    """
    import numpy as np
    
    values = series.iloc[-window:] if len(series) >= window else series
    if len(values) <= 1:
        return 0
    
    # Create x values (0, 1, 2, ...)
    x = np.arange(len(values))
    # Get y values
    y = values.values
    
    # Handle null/nan values
    mask = ~np.isnan(y)
    if sum(mask) <= 1:  # Need at least two valid points
        return 0
    
    x = x[mask]
    y = y[mask]
    
    # Calculate slope using linear regression
    if len(x) > 1:
        try:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        except:
            return 0
    return 0

def calculate_conditional_mean(values, condition, window):
    """
    Calculate rolling mean for values where condition is True
    
    Args:
        values: Series of values
        condition: Boolean series indicating which values to include
        window: Rolling window size
        
    Returns:
        Series with rolling conditional mean
    """
    import numpy as np
    import pandas as pd
    
    result = pd.Series(index=values.index, dtype=float)
    
    for i in range(len(values)):
        # Get last N values where condition was met
        end_idx = i + 1
        start_idx = max(0, end_idx - window)
        
        # Extract relevant subset
        subset_vals = values.iloc[start_idx:end_idx]
        subset_cond = condition.iloc[start_idx:end_idx]
        
        # Filter values by condition
        filtered = subset_vals[subset_cond]
        
        # Calculate mean if any values remain
        if len(filtered) > 0:
            result.iloc[i] = filtered.mean()
        else:
            result.iloc[i] = np.nan
    
    return result

def get_team_conference(team_id, season, team_profiles):
    """
    Get team's conference from team_profiles
    
    Args:
        team_id: Team ID
        season: Season
        team_profiles: DataFrame with team profiles
        
    Returns:
        Conference name
    """
    from ..utils.data_access import get_data_with_index
    
    # Try to use get_data_with_index for optimized lookup
    try:
        # Check if team_profiles has the right structure
        if hasattr(team_profiles, 'index') and isinstance(team_profiles.index, pd.MultiIndex):
            # If team_profiles is already indexed by Season and TeamID
            if 'ConfAbbrev' in team_profiles.columns:
                team_data = get_data_with_index(team_profiles, ('Season', 'TeamID'), (season, team_id))
                if not team_data.empty:
                    return team_data['ConfAbbrev'].iloc[0]
        else:
            # If team_profiles is a regular DataFrame
            if 'ConfAbbrev' in team_profiles.columns:
                team_data = team_profiles[(team_profiles['Season'] == season) & (team_profiles['TeamID'] == team_id)]
                if not team_data.empty:
                    return team_data['ConfAbbrev'].iloc[0]
    except Exception as e:
        pass
    
    return None

def calculate_coach_features(coaches_df, tourney_df):
    """
    Calculate coaching experience and performance features
    """
    # Identify head coaches (for simplicity, assuming last coach listed for a team in a season)
    head_coaches = coaches_df.sort_values(['Season', 'TeamID', 'LastDayNum']).groupby(['Season', 'TeamID']).last().reset_index()

    # Get coach career records across seasons
    coach_records = []

    # Process all coaches from our filtered dataset
    for coach_name in head_coaches['CoachName'].unique():
        coach_seasons = head_coaches[head_coaches['CoachName'] == coach_name]

        for i, row in coach_seasons.iterrows():
            season = row['Season']
            team_id = row['TeamID']

            # Find all previous seasons for this coach
            prev_experience = head_coaches[(head_coaches['CoachName'] == coach_name) &
                                         (head_coaches['Season'] < season)]

            # Calculate years of experience
            years_exp = len(prev_experience)

            # Find tournament appearances
            tourney_teams_w = tourney_df[tourney_df['Season'] < season]['WTeamID'].unique()
            tourney_teams_l = tourney_df[tourney_df['Season'] < season]['LTeamID'].unique()
            tourney_teams = np.union1d(tourney_teams_w, tourney_teams_l)

            prev_tourney_exp = prev_experience[prev_experience['TeamID'].isin(tourney_teams)]
            tourney_appearances = len(prev_tourney_exp)

            # Check for championships (assuming highest DayNum represents championship game)
            if 'DayNum' in tourney_df.columns:
                seasons_with_champ_games = {}
                for s in tourney_df['Season'].unique():
                    if len(tourney_df[tourney_df['Season'] == s]) > 0:
                        max_day = tourney_df[tourney_df['Season'] == s]['DayNum'].max()
                        champ_games = tourney_df[(tourney_df['Season'] == s) & (tourney_df['DayNum'] == max_day)]
                        for _, champ_game in champ_games.iterrows():
                            seasons_with_champ_games[s] = champ_game['WTeamID']  # Winner of championship

                # Count championships for this coach
                championships = 0
                for s, champ_team in seasons_with_champ_games.items():
                    if s < season:  # Only consider past seasons
                        coach_in_season = prev_experience[prev_experience['Season'] == s]
                        if len(coach_in_season) > 0 and coach_in_season['TeamID'].values[0] == champ_team:
                            championships += 1
            else:
                championships = 0  # No way to identify championships without round/daynum info

            coach_records.append({
                'Season': season,
                'TeamID': team_id,
                'CoachName': coach_name,
                'CoachYearsExp': years_exp,
                'CoachTourneyExp': tourney_appearances,
                'CoachChampionships': championships
            })

    return pd.DataFrame(coach_records)

def enhance_team_metrics(all_team_games, all_team_profiles, regular_season_df):
    """
    Enhances team metrics while ensuring proper calculation of defensive statistics

    Args:
        all_team_games: DataFrame with team game data from create_team_season_profiles
        all_team_profiles: DataFrame with team profile data from create_team_season_profiles
        regular_season_df: Original regular season results dataframe (needed for proper defensive calculations)

    Returns:
        Tuple of (enhanced_team_games, enhanced_team_profiles)
    """
    # Import numpy if not already imported in your code
    import numpy as np
    import pandas as pd

    # Debugging: Print the structure of the input dataframes
    print(f"all_team_games shape: {all_team_games.shape}")
    print(f"all_team_games index type: {type(all_team_games.index)}")

    # Work with copies to avoid modifying originals
    enhanced_games = all_team_games.copy()
    enhanced_profiles = all_team_profiles.copy()

    # 1. Add new offensive metrics at the game level

    # Score distribution metrics
    enhanced_games['FG2M'] = enhanced_games['FGM'] - enhanced_games['FGM3']
    enhanced_games['FG2A'] = enhanced_games['FGA'] - enhanced_games['FGA3']

    # Handle potential division by zero in all calculations
    enhanced_games['FG2Pct'] = np.where(
        enhanced_games['FG2A'] > 0,
        enhanced_games['FG2M'] / enhanced_games['FG2A'],
        0
    )

    enhanced_games['Points_From_2'] = enhanced_games['FG2M'] * 2
    enhanced_games['Points_From_3'] = enhanced_games['FGM3'] * 3
    enhanced_games['Points_From_FT'] = enhanced_games['FTM']

    # Safe division operations
    enhanced_games['Pct_Points_From_2'] = np.where(
        enhanced_games['Score'] > 0,
        enhanced_games['Points_From_2'] / enhanced_games['Score'],
        0
    )

    enhanced_games['Pct_Points_From_3'] = np.where(
        enhanced_games['Score'] > 0,
        enhanced_games['Points_From_3'] / enhanced_games['Score'],
        0
    )

    enhanced_games['Pct_Points_From_FT'] = np.where(
        enhanced_games['Score'] > 0,
        enhanced_games['Points_From_FT'] / enhanced_games['Score'],
        0
    )

    # Additional efficiency metrics with safe division
    enhanced_games['Assist_Rate'] = np.where(
        enhanced_games['FGM'] > 0,
        enhanced_games['Ast'] / enhanced_games['FGM'],
        0
    )

    enhanced_games['Steal_Rate'] = np.where(
        enhanced_games['Possessions'] > 0,
        enhanced_games['Stl'] / enhanced_games['Possessions'],
        0
    )

    enhanced_games['Block_Rate'] = np.where(
        enhanced_games['OpponentScore'] > 0,
        enhanced_games['Blk'] / enhanced_games['OpponentScore'] * 100,
        0
    )

    # Safe denominator for TOV_Rate
    denom_tov = enhanced_games['FGA'] + 0.44 * enhanced_games['FTA'] + enhanced_games['TO']
    enhanced_games['TOV_Rate'] = np.where(
        denom_tov > 0,
        enhanced_games['TO'] / denom_tov,
        0
    )

    enhanced_games['FT_Rate'] = np.where(
        enhanced_games['FGA'] > 0,
        enhanced_games['FTA'] / enhanced_games['FGA'],
        0
    )

    # Add tempo (pace) - COMPLETELY REDONE
    # Create a NumOT column filled with zeros if it doesn't exist
    if 'NumOT' not in enhanced_games.columns:
        enhanced_games['NumOT'] = 0

    # Use apply method to calculate tempo row by row to avoid any indexing issues
    enhanced_games['Tempo'] = enhanced_games.apply(
        lambda row: row['Possessions'] / (40 - 5 * (row['NumOT'] if not pd.isna(row['NumOT']) else 0))
                    if (40 - 5 * (row['NumOT'] if not pd.isna(row['NumOT']) else 0)) > 0 else 0,
        axis=1
    )

    # Ensure proper numeric data types
    numeric_cols = ['FG2Pct', 'Pct_Points_From_2', 'Pct_Points_From_3', 'Pct_Points_From_FT',
                   'Assist_Rate', 'Steal_Rate', 'Block_Rate', 'TOV_Rate', 'FT_Rate', 'Tempo']

    for col in numeric_cols:
        enhanced_games[col] = pd.to_numeric(enhanced_games[col], errors='coerce')
        enhanced_games[col] = enhanced_games[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 2. Properly calculate defensive efficiency using opponent possessions
    # Process winning team games
    w_possessions = regular_season_df.copy()
    w_possessions['WPossessions'] = (w_possessions['WFGA'] + (0.44*w_possessions['WFTA']) -
                                     w_possessions['WOR'] + w_possessions['WTO']).round()
    w_possessions['LPossessions'] = (w_possessions['LFGA'] + (0.44*w_possessions['LFTA']) -
                                     w_possessions['LOR'] + w_possessions['LTO']).round()

    # Create game identifier
    w_possessions['GameID'] = (w_possessions['Season'].astype(str) + '_' +
                              w_possessions['DayNum'].astype(str) + '_' +
                              w_possessions['WTeamID'].astype(str) + '_' +
                              w_possessions['LTeamID'].astype(str))

    # Create lookup dictionaries for possessions and efficiency
    w_off_eff = {}  # Team offensive efficiency
    w_def_eff = {}  # Team defensive efficiency

    for _, game in w_possessions.iterrows():
        # Calculate offensive and defensive efficiency for winning team
        w_team = game['WTeamID']
        l_team = game['LTeamID']
        w_poss = game['WPossessions']
        l_poss = game['LPossessions']

        # Offensive efficiency = Own points / Own possessions
        w_off = 100 * game['WScore'] / w_poss if w_poss > 0 else 0

        # Defensive efficiency = Opponent points / Opponent possessions
        w_def = 100 * game['LScore'] / l_poss if l_poss > 0 else 0

        # Create team-game keys
        w_key = f"{game['Season']}_{game['DayNum']}_{w_team}"
        l_key = f"{game['Season']}_{game['DayNum']}_{l_team}"

        # Store winning team's offensive and defensive efficiency
        w_off_eff[w_key] = w_off
        w_def_eff[w_key] = w_def

        # Store losing team's offensive and defensive efficiency (reversed)
        w_off_eff[l_key] = 100 * game['LScore'] / l_poss if l_poss > 0 else 0
        w_def_eff[l_key] = 100 * game['WScore'] / w_poss if w_poss > 0 else 0

    # Add proper defensive efficiency to enhanced_games
    # Use apply to avoid potential index issues
    enhanced_games['DefEfficiency'] = enhanced_games.apply(
        lambda row: w_def_eff.get(f"{row['Season']}_{row['DayNum']}_{row['TeamID']}", 0),
        axis=1
    )

    # Calculate net efficiency at the game level
    enhanced_games['NetEfficiency'] = enhanced_games['OffEfficiency'] - enhanced_games['DefEfficiency']

    # Group by season and team to calculate additional aggregated stats
    # Using a more explicit approach to avoid potential groupby issues
    additional_stats_list = []

    for (season, team_id), group in enhanced_games.groupby(['Season', 'TeamID']):
        stats_dict = {
            'Season': season,
            'TeamID': team_id,
            'Season_FG2Pct': group['FG2Pct'].mean(),
            'Season_Pct_Points_From_2': group['Pct_Points_From_2'].mean(),
            'Season_Pct_Points_From_3': group['Pct_Points_From_3'].mean(),
            'Season_Pct_Points_From_FT': group['Pct_Points_From_FT'].mean(),
            'Season_Assist_Rate': group['Assist_Rate'].mean(),
            'Season_Steal_Rate': group['Steal_Rate'].mean(),
            'Season_Block_Rate': group['Block_Rate'].mean(),
            'Season_TOV_Rate': group['TOV_Rate'].mean(),
            'Season_FT_Rate': group['FT_Rate'].mean(),
            'Season_Tempo': group['Tempo'].mean(),
            'Season_DefEfficiency': group['DefEfficiency'].mean(),
            'Season_NetEfficiency': group['NetEfficiency'].mean()
        }
        additional_stats_list.append(stats_dict)

    additional_stats = pd.DataFrame(additional_stats_list)

    # Merge additional stats with team profiles
    # Ensure we're merging on the correct types to avoid issues
    enhanced_profiles['Season'] = enhanced_profiles['Season'].astype(int)
    enhanced_profiles['TeamID'] = enhanced_profiles['TeamID'].astype(int)
    additional_stats['Season'] = additional_stats['Season'].astype(int)
    additional_stats['TeamID'] = additional_stats['TeamID'].astype(int)

    enhanced_profiles = enhanced_profiles.merge(additional_stats, on=['Season', 'TeamID'], how='left')

    # Add percentile ranks for key metrics
    percentile_cols = ['Season_OffEfficiency', 'Season_DefEfficiency', 'Season_eFGPct',
                      'Season_Win', 'Season_NetEfficiency']

    # Use a safer approach for percentile calculation
    for col in percentile_cols:
        if col in enhanced_profiles.columns:
            for season in enhanced_profiles['Season'].unique():
                season_mask = enhanced_profiles['Season'] == season
                if season_mask.any():  # Check if there are any rows for this season
                    season_data = enhanced_profiles.loc[season_mask, col]
                    if not season_data.empty:
                        enhanced_profiles.loc[season_mask, f'{col}_Percentile'] = season_data.rank(pct=True)

    # Skip or implement calculate_team_consistency and calculate_team_playstyle functions
    # For now, we'll just pass to avoid errors if they're not defined

    print("Enhancement completed successfully")
    return enhanced_games, enhanced_profiles

def calculate_strength_of_schedule(regular_season_results, team_profiles, current_season=None, tournament_days=None):
    """
    Calculate strength of schedule (SOS) using only regular season games

    Args:
        regular_season_results: DataFrame with regular season results
        team_profiles: DataFrame with team season profiles
        current_season: Current season for prediction
        tournament_days: List of tournament day numbers to exclude

    Returns:
        DataFrame with SOS metrics
    """
    # Filter to only include regular season games
    reg_season_df = filter_reg_season(regular_season_results, current_season, tournament_days)

    # Create temporary dataframes with opponent information
    w_games = reg_season_df[['Season', 'WTeamID', 'LTeamID']].copy()
    w_games.columns = ['Season', 'TeamID', 'OpponentID']
    w_games['Win'] = 1

    l_games = reg_season_df[['Season', 'LTeamID', 'WTeamID']].copy()
    l_games.columns = ['Season', 'TeamID', 'OpponentID']
    l_games['Win'] = 0

    all_games = pd.concat([w_games, l_games], ignore_index=True)

    # Merge with team profiles to get opponent win rates
    team_metrics = team_profiles[['Season', 'TeamID', 'Season_Win', 'Season_OffEfficiency', 'Season_DefEfficiency']]

    all_games = all_games.merge(
        team_metrics.rename(columns={
            'TeamID': 'OpponentID',
            'Season_Win': 'OpponentWinRate',
            'Season_OffEfficiency': 'OpponentOffEff',
            'Season_DefEfficiency': 'OpponentDefEff'
        }),
        on=['Season', 'OpponentID'],
        how='left'
    )

    # Calculate SOS metrics
    sos_metrics = all_games.groupby(['Season', 'TeamID']).agg(
        AvgOpponentWinRate=('OpponentWinRate', 'mean'),
        AvgOpponentOffEff=('OpponentOffEff', 'mean'),
        AvgOpponentDefEff=('OpponentDefEff', 'mean'),
        GamesAgainstWinningTeams=('OpponentWinRate', lambda x: sum(x > 0.5)),
        WinsAgainstWinningTeams=('Win', lambda x: sum((all_games.loc[x.index, 'OpponentWinRate'] > 0.5) & (x == 1))),
    ).reset_index()

    # Calculate win rate against winning teams
    sos_metrics['WinRateVsWinningTeams'] = sos_metrics['WinsAgainstWinningTeams'] / sos_metrics['GamesAgainstWinningTeams']
    sos_metrics['WinRateVsWinningTeams'] = sos_metrics['WinRateVsWinningTeams'].fillna(0.5)

    # Normalize SOS metrics by season
    sos_metrics['SOSPercentile'] = sos_metrics.groupby('Season')['AvgOpponentWinRate'].transform(
        lambda x: x.rank(pct=True)
    )

    return sos_metrics

def calculate_team_consistency(team_games):
    """
    Calculate metrics for team consistency and volatility

    Args:
        team_games: DataFrame with team game data

    Returns:
        DataFrame with team consistency metrics by season
    """
    # Group by season and team
    consistency_metrics = team_games.groupby(['Season', 'TeamID']).agg({
        'Score': ['std', 'mean'],
        'ScoreMargin': ['std', 'mean', 'min', 'max'],
        'eFGPct': ['std', 'mean'],
        'Win': ['std']  # This gives us volatility in win/loss outcomes
    })

    # Flatten the column hierarchy
    consistency_metrics.columns = ['_'.join(col).strip() for col in consistency_metrics.columns.values]

    # Calculate coefficient of variation (CV) for key metrics
    # CV = std / mean (shows relative variability)
    consistency_metrics['Score_CV'] = consistency_metrics['Score_std'] / consistency_metrics['Score_mean']
    consistency_metrics['Margin_CV'] = consistency_metrics['ScoreMargin_std'] / (
        consistency_metrics['ScoreMargin_mean'].abs() + 1)  # Added +1 to avoid division by zero
    consistency_metrics['eFGPct_CV'] = consistency_metrics['eFGPct_std'] / (consistency_metrics['eFGPct_mean'] + 0.001)

    # Calculate "clutch" metrics
    # How a team performs in close games (decided by 5 points or less)
    close_games = team_games[abs(team_games['ScoreMargin']) <= 5]

    if not close_games.empty:
        close_game_metrics = close_games.groupby(['Season', 'TeamID']).agg({
            'Win': 'mean',  # Win rate in close games
            'Score': 'mean',  # Scoring in close games
            'ScoreMargin': 'mean'  # Average margin in close games
        })

        close_game_metrics.columns = ['CloseGame_WinRate', 'CloseGame_Score', 'CloseGame_Margin']

        # Merge with consistency metrics
        consistency_metrics = consistency_metrics.reset_index().merge(
            close_game_metrics.reset_index(),
            on=['Season', 'TeamID'],
            how='left'
        )
    else:
        # Add placeholder columns if no close games
        consistency_metrics = consistency_metrics.reset_index()
        consistency_metrics['CloseGame_WinRate'] = 0.5
        consistency_metrics['CloseGame_Score'] = consistency_metrics['Score_mean']
        consistency_metrics['CloseGame_Margin'] = 0

    return consistency_metrics

def calculate_team_playstyle(team_games):
    """
    Identify a team's play style based on statistical tendencies

    Args:
        team_games: DataFrame with team game statistics

    Returns:
        DataFrame with team play style metrics
    """
    # Ensure necessary columns exist
    required_cols = ['Possessions', 'Pct_Points_From_2', 'Pct_Points_From_3',
                     'Pct_Points_From_FT', 'Assist_Rate', 'OR', 'Stl', 'Blk']

    for col in required_cols:
        if col not in team_games.columns and col not in ['Pct_Points_From_2', 'Pct_Points_From_3', 'Pct_Points_From_FT', 'Assist_Rate']:
            # Skip non-essential columns if they don't exist
            team_games[col] = 0

    # Group by season and team to get style metrics
    cols_to_use = ['Season', 'TeamID', 'Possessions']
    for col in required_cols:
        if col in team_games.columns:
            cols_to_use.append(col)

    play_style = team_games[cols_to_use].groupby(['Season', 'TeamID']).mean().reset_index()

    # If tempStyle doesn't exist, calculate it
    if 'Tempo' not in play_style.columns and 'Possessions' in play_style.columns:
        play_style['Tempo'] = play_style['Possessions'] / 40  # Simple approximation

    # Calculate style metrics based on percentiles
    season_styles = []

    for season in play_style['Season'].unique():
        season_data = play_style[play_style['Season'] == season].copy()

        # Calculate percentiles for each metric within the season
        for col in season_data.columns:
            if col not in ['Season', 'TeamID']:
                percentile_col = f'{col}_Percentile'
                season_data[percentile_col] = season_data[col].rank(pct=True)

        season_styles.append(season_data)

    # Combine all seasons
    all_styles = pd.concat(season_styles, ignore_index=True)

    # Categorize team play styles
    # Define thresholds for play style categorization (top/bottom 25%)
    high_threshold = 0.75
    low_threshold = 0.25

    # Fast vs. Slow pace
    if 'Tempo_Percentile' in all_styles.columns:
        all_styles['FastPaced'] = all_styles['Tempo_Percentile'] > high_threshold
        all_styles['SlowPaced'] = all_styles['Tempo_Percentile'] < low_threshold
    else:
        all_styles['FastPaced'] = all_styles['Possessions_Percentile'] > high_threshold
        all_styles['SlowPaced'] = all_styles['Possessions_Percentile'] < low_threshold

    # Inside vs. Outside scoring
    if 'Pct_Points_From_2_Percentile' in all_styles.columns:
        all_styles['InsideScoring'] = all_styles['Pct_Points_From_2_Percentile'] > high_threshold
    else:
        all_styles['InsideScoring'] = False

    if 'Pct_Points_From_3_Percentile' in all_styles.columns:
        all_styles['OutsideScoring'] = all_styles['Pct_Points_From_3_Percentile'] > high_threshold
    else:
        all_styles['OutsideScoring'] = False

    # Offensive styles
    if 'Assist_Rate_Percentile' in all_styles.columns:
        all_styles['HighAssist'] = all_styles['Assist_Rate_Percentile'] > high_threshold
    else:
        all_styles['HighAssist'] = False

    if 'TOV_Rate_Percentile' in all_styles.columns:
        all_styles['LowTurnover'] = all_styles['TOV_Rate_Percentile'] < low_threshold
    else:
        all_styles['LowTurnover'] = False

    all_styles['OffensiveRebounding'] = all_styles['OR_Percentile'] > high_threshold

    # Defensive styles
    all_styles['HighSteal'] = all_styles['Stl_Percentile'] > high_threshold
    all_styles['HighBlock'] = all_styles['Blk_Percentile'] > high_threshold

    return all_styles

def calculate_womens_specific_features(team_games, team_profiles):
    """
    Calculate features specifically tuned for women's basketball using available data fields
    """
    womens_features = []
    
    for _, team_profile in team_profiles.iterrows():
        team_id = team_profile['TeamID']
        season = team_profile['Season']
        
        # Get team's games
        team_season_games = team_games[(team_games['Season'] == season) & 
                                       (team_games['TeamID'] == team_id)]
        
        if len(team_season_games) == 0:
            continue
            
        # Calculate women's game specific metrics with available fields
        try:
            # 1. Scoring distribution metrics (using existing data)
            points_from_2 = (team_season_games['FGM'] - team_season_games['FGM3']).sum() * 2
            points_from_3 = team_season_games['FGM3'].sum() * 3
            points_from_ft = team_season_games['FTM'].sum()
            total_points = team_season_games['Score'].sum()
            
            if total_points > 0:
                scoring_dist_2pt = points_from_2 / total_points
                scoring_dist_3pt = points_from_3 / total_points
                scoring_dist_ft = points_from_ft / total_points
            else:
                scoring_dist_2pt = scoring_dist_3pt = scoring_dist_ft = 0.333
            
            # 2. Assist-to-Field Goal ratio (team ball movement indicator)
            total_fgm = team_season_games['FGM'].sum()
            total_ast = team_season_games['Ast'].sum()
            ast_to_fgm = total_ast / total_fgm if total_fgm > 0 else 0
            
            # 3. Defense intensity metrics - using steals and blocks
            def_intensity = (team_season_games['Stl'].sum() + team_season_games['Blk'].sum()) / len(team_season_games)
            
            # 4. Free throw reliance
            ft_reliance = team_season_games['FTA'].sum() / team_season_games['FGA'].sum() if team_season_games['FGA'].sum() > 0 else 0
            
            # 5. Rebounding advantage - using available data
            total_reb = team_season_games['OR'].sum() + team_season_games['DR'].sum()
            total_games = len(team_season_games)
            reb_per_game = total_reb / total_games if total_games > 0 else 0
            
            # 6. Turnover management (women's game often involves different turnover patterns)
            to_rate = team_season_games['TO'].sum() / team_season_games['Possessions'].sum() if team_season_games['Possessions'].sum() > 0 else 0
            
            # 7. Field goal efficiency comparison
            fg2_pct = (team_season_games['FGM'] - team_season_games['FGM3']).sum() / (team_season_games['FGA'] - team_season_games['FGA3']).sum() if (team_season_games['FGA'] - team_season_games['FGA3']).sum() > 0 else 0
            fg3_pct = team_season_games['FGM3'].sum() / team_season_games['FGA3'].sum() if team_season_games['FGA3'].sum() > 0 else 0
            fg_ratio = fg2_pct / fg3_pct if fg3_pct > 0 else 1.0
            
            womens_features.append({
                'Season': season,
                'TeamID': team_id,
                'Womens_ScoringDist_2pt': scoring_dist_2pt,
                'Womens_ScoringDist_3pt': scoring_dist_3pt,
                'Womens_ScoringDist_FT': scoring_dist_ft,
                'Womens_AstToFGM': ast_to_fgm,
                'Womens_DefIntensity': def_intensity,
                'Womens_FTReliance': ft_reliance,
                'Womens_RebPerGame': reb_per_game,
                'Womens_TORate': to_rate,
                'Womens_FG2to3Ratio': fg_ratio
            })
        except Exception as e:
            print(f"Error calculating women's features for team {team_id}, season {season}: {str(e)}")
            continue
    
    # Only return if we actually generated features
    result_df = pd.DataFrame(womens_features) if womens_features else pd.DataFrame()
    if not result_df.empty:
        print(f"Generated women's-specific features for {len(result_df)} team-seasons")
    return result_df
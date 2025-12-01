from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and data
data = joblib.load('nba_model_data.joblib')
models = data['models']
x_features = data['x_features']
df_model = data['df_model']
team_defense_ratings = data['team_defense_ratings']

# Load Last 7 Games Data for Visualization
try:
    df_last_7 = pd.read_csv('nba_player_last_7_games.csv')
except:
    df_last_7 = pd.DataFrame()

# Calculate League Average DRtg for adjustment
league_avg_drtg = np.mean(list(team_defense_ratings.values()))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    selected_player = None
    selected_opponent = None
    last_7_data = None
    
    # Get lists for dropdowns
    players = sorted(df_model['NAME'].unique())
    teams = sorted(team_defense_ratings.keys())
    
    if request.method == 'POST':
        selected_player = request.form.get('player_name')
        selected_opponent = request.form.get('opponent_team')
        
        if selected_player and selected_opponent:
            # Get the player's data
            player_data = df_model[df_model['NAME'] == selected_player].iloc[0]
            input_data = player_data[x_features].to_dict()
            input_df = pd.DataFrame([input_data])
            input_df = input_df[x_features]
            preds = {} # Store predictions
            targets = ['PpG', 'RpG', 'ApG'] # Points, Rebounds, Assists
            
            # Get Opponent Defense Rating
            opponent_drtg = team_defense_ratings.get(selected_opponent, league_avg_drtg) # Default to league avg if not found
            defense_factor = opponent_drtg / league_avg_drtg # >1 means tougher defense(worse stats) and <1 means easier defense(better stats)
            
            for target in targets:
                base_pred = models[target].predict(input_df)[0]
                final_pred = base_pred * defense_factor
                preds[target] = round(final_pred, 1)           
            prediction = { # Store all prediction info
                'name': selected_player,
                'opponent': selected_opponent,
                'stats': preds,
                'actual_avg': {
                    'PpG': player_data['PpG'],
                    'RpG': player_data['RpG'],
                    'ApG': player_data['ApG']
                },
                'context': { # Defensive context
                'opp_drtg': round(opponent_drtg, 1), # Opponent Defensive Rating
                    'league_avg': round(league_avg_drtg, 1), # League Average Defensive Rating
                    'factor': round(defense_factor, 2) # Adjustment Factor
                }
            }

            # Get Last 7 Games for the player
            if not df_last_7.empty:
                player_games = df_last_7[df_last_7['PlayerName'] == selected_player].copy() 
                if not player_games.empty:
                    player_games = player_games.iloc[::-1] # Reverse to chronological order
                    
                    last_7_data = { # Prepare data for visualization
                        'dates': player_games['GAME_DATE'].tolist(),
                        'pts': player_games['PTS'].tolist(),
                        'reb': player_games['REB'].tolist(),
                        'ast': player_games['AST'].tolist()
                    }
            
    return render_template('index.html', players=players, teams=teams, prediction=prediction, selected_player=selected_player, selected_opponent=selected_opponent, last_7_data=last_7_data)

if __name__ == '__main__':
    app.run()

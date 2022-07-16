

import Metrica_IO as mio
import Metrica_Viz as mviz
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import Metrica_EPV as mepv


import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pdb
import warnings
import re
import argparse

warnings.simplefilter('ignore')

import third_party as thp
import obso_player as obs


# create parser
parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, default=2, help='game id')
parser.add_argument('--data', type=str, default='metrica', help='dataset')
args = parser.parse_args()

# select game number
game_id = args.id

if args.data == 'metrica':
    # set up initial path to data
    DATADIR = './metrica-sample-data/data/'

    # read in the event data
    events = mio.read_event_data(DATADIR,game_id)

    # read in tracking data
    tracking_home = mio.tracking_data(DATADIR,game_id,'Home')
    tracking_away = mio.tracking_data(DATADIR,game_id,'Away')

    # Convert positions from metrica units to meters (note change in Metrica's coordinate system since the last lesson)
    tracking_home = mio.to_metric_coordinates(tracking_home)
    tracking_away = mio.to_metric_coordinates(tracking_away)
    events = mio.to_metric_coordinates(events)

    # reverse direction of play in the second half so that home team is always attacking from right->left
    tracking_home,tracking_away,events = mio.to_single_playing_direction(tracking_home,tracking_away,events)

    # Calculate player velocities
    tracking_home = mvel.calc_player_velocities(tracking_home,smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away,smoothing=True)

    # event data convert spadl to Metrica
    Metrica_df = obs.convert_Metrica_for_event(sample_spadl)
    # check 'Home' team in tracking and event data
    Metrica_df = obs.check_home_away_event(Metrica_df, tracking_home, tracking_away)
    # delete last event because this event is 'time up' event
    Metrica_df = Metrica_df[:-1]

elif args.data == 'jleague':
    # set folder and file name
    Jdatafolder = "../JLeagueData"
    FMfolder = "/Data_2019FM/"
    Jdata_FM = Jdatafolder + FMfolder
    event_data_name = "/play.csv"
    player_data_name = "/player.csv"
    game_date =  os.listdir(path=Jdata_FM)

    # set event data
    sample_game_data = pd.read_csv(Jdata_FM+game_date[game_id]+event_data_name, encoding="shift_jis")
    sample_spadl = thp.convert_J2spadl(sample_game_data)

    # set tracking data
    tracking_home = pd.read_csv(Jdata_FM+game_date[game_id]+'/home_tracking.csv')
    tracking_away = pd.read_csv(Jdata_FM+game_date[game_id]+'/away_tracking.csv')
    tracking_home = tracking_home.drop(columns='Unnamed: 0')
    tracking_away = tracking_away.drop(columns='Unnamed: 0')

    # preprocessing player position 
    entry_home_df = tracking_home.loc[0].isnull()
    entry_away_df = tracking_away.loc[0].isnull()
    home_column = tracking_home.columns
    away_column = tracking_away.columns
    home_player_num = [s[:-2] for s in home_column if re.match('Home_\d*_x', s)]
    away_player_num = [s[:-2] for s in away_column if re.match('Away_\d*_x', s)]

    # replace nan 
    for player in home_player_num:
        if entry_home_df[player+'_x']:
            tracking_home[player+'_x'] = tracking_home[player+'_x'].fillna(method='ffill')
            tracking_home[player+'_y'] = tracking_home[player+'_y'].fillna(method='ffill')
        else:
            tracking_home[player+'_x'] = tracking_home[player+'_x'].fillna(method='bfill')
            tracking_home[player+'_y'] = tracking_home[player+'_y'].fillna(method='bfill')

    for player in away_player_num:
        if entry_away_df[player+'_x']:
            tracking_away[player+'_x'] = tracking_away[player+'_x'].fillna(method='ffill')
            tracking_away[player+'_y'] = tracking_away[player+'_y'].fillna(method='ffill')
        else:
            tracking_away[player+'_x'] = tracking_away[player+'_x'].fillna(method='bfill')
            tracking_away[player+'_y'] = tracking_away[player+'_y'].fillna(method='bfill')

    # data interpolation in ball position in tracking data
    tracking_home['ball_x'] = tracking_home['ball_x'].interpolate()
    tracking_home['ball_y'] = tracking_home['ball_y'].interpolate()
    tracking_away['ball_x'] = tracking_away['ball_x'].interpolate()
    tracking_away['ball_y'] = tracking_away['ball_y'].interpolate()

    # check nan ball position x and y in tracking data
    tracking_home['ball_x'] = tracking_home['ball_x'].fillna(method='bfill')
    tracking_home['ball_y'] = tracking_home['ball_y'].fillna(method='bfill')
    tracking_away['ball_x'] = tracking_away['ball_x'].fillna(method='bfill')
    tracking_away['ball_y'] = tracking_away['ball_y'].fillna(method='bfill')

    # event data convert spadl to Metrica
    Metrica_df = obs.convert_Metrica_for_event(sample_spadl)
    # check 'Home' team in tracking and event data
    Metrica_df = obs.check_home_away_event(Metrica_df, tracking_home, tracking_away)
    # delete last event because this event is 'time up' event
    Metrica_df = Metrica_df[:-1]

# filter:Savitzky-Golay
tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True) 
tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)

# set parameter
params = mpc.default_model_params()
GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]

# load control and transition model
EPV = mepv.load_EPV_grid('EPV_grid.csv')
EPV = EPV / np.max(EPV)
Trans_df = pd.read_csv('Transition_gauss.csv', header=None)
Trans = np.array((Trans_df))
Trans = Trans / np.max(Trans)

# set OBSO data
obso = np.zeros((len(Metrica_df), 32, 50))
for event_num, frame in tqdm(enumerate(Metrica_df['Start Frame'])):
    
    if Metrica_df['Team'].loc[event_num]=='Home':
        # check attack direction 1st half or 2nd half
        if Metrica_df.loc[event_num]['Period']==1:
            direction = mio.find_playing_direction(tracking_home[tracking_home['Period']==1], 'Home')
        elif Metrica_df.loc[event_num]['Period']==2:
            direction = mio.find_playing_direction(tracking_home[tracking_home['Period']==2], 'Home')
        PPCF, _, _, _ = mpc.generate_pitch_control_for_event(event_num, Metrica_df, tracking_home, tracking_away, params, GK_numbers, offsides=True)

    elif Metrica_df['Team'].loc[event_num]=='Away': 
        # check attack direction 1st half or 2nd half
        if Metrica_df.loc[event_num]['Period']==1:
            direction = mio.find_playing_direction(tracking_away[tracking_away['Period']==1], 'Away')
        elif Metrica_df.loc[event_num]['Period']==2:
            direction = mio.find_playing_direction(tracking_away[tracking_away['Period']==2], 'Away')
        PPCF, _, _, _ = mpc.generate_pitch_control_for_event(event_num, Metrica_df, tracking_home, tracking_away, params, GK_numbers, offsides=True)
    
    else:
        obso[event_num] = np.zeros((32, 50))
        continue
    obso[event_num], _ = obs.calc_obso(PPCF, Trans, EPV, tracking_home.loc[frame], attack_direction=direction)

home_obso, away_obso = obs.calc_player_evaluate_match(obso, Metrica_df, tracking_home, tracking_away)

# calculate onball obso
home_onball_obso, away_onball_obso = obs.calc_onball_obso(Metrica_df, tracking_home, tracking_away, home_obso, away_obso)
# remove offside player
home_obso, away_obso = obs.remove_offside_obso(Metrica_df, tracking_home, tracking_away, home_obso, away_obso)

# save obso in home and away
home_obso.to_pickle(Jdata_FM+game_date[game_id]+'/home_obso.pkl')
away_obso.to_pickle(Jdata_FM+game_date[game_id]+'/away_obso.pkl')
home_onball_obso.to_pickle(Jdata_FM+game_date[game_id]+'/home_onball_obso.pkl')
away_onball_obso.to_pickle(Jdata_FM+game_date[game_id]+'/away_onball_obso.pkl')

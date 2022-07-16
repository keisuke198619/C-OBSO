# features.py
import glob, os, sys, math, warnings, copy, time, glob
import numpy as np
import pandas as pd

# modifying the code https://github.com/samshipengs/Coordinated-Multi-Agent-Imitation-Learning

# =================================================================
# flatten_moments =================================================
# =================================================================
def flatten_moments(events_df,normalize):
    ''' This changes the nested list that represents single frame 
        to a 1-D array.
     '''
    df = events_df.copy()
    def flatten_moment(moment,normalize):
        m = np.array(moment[5])
        feet_m = 0.3048
        features = np.concatenate((m[1:11, 2:4].reshape(-1)*feet_m,    # x,y of all 10 players 
                                   m[0][2:5]*feet_m,                   # basketball x,y,z 
                                   np.array([moment[0]]),       # quarter number 
                                   np.array([moment[2]]),       # time in seconds left to the end of the period
                                   np.array([moment[3]])))      # shot clock 


        if normalize:                 
            LENGTH = 94*feet_m
            WIDTH = 50*feet_m
            SHIFT0 = [47*feet_m,25*feet_m]
            SEQUENCE_DIMENSION = 22 # features.shape[0]

            NORMALIZE = [LENGTH, WIDTH] * int(SEQUENCE_DIMENSION/2)
            SHIFT = SHIFT0 * int(SEQUENCE_DIMENSION/2)

            feat0 = features[:SEQUENCE_DIMENSION].copy() # for check
            features[:SEQUENCE_DIMENSION] = np.divide(features[:SEQUENCE_DIMENSION]-SHIFT, NORMALIZE)    

            # for check
            feat1 = np.multiply(features[:SEQUENCE_DIMENSION], NORMALIZE) + SHIFT 
            err = np.mean(np.abs(feat0[:SEQUENCE_DIMENSION]-feat1))
            if err > 1e-3:
                print('err ='+str(err))

        return features
    
    def get_team_ids(moment):
        m = np.array(moment[5])
        team_id1 = set(m[1:6, 0])
        team_id2 = set(m[6:11, 0])
        assert len(team_id1) == len(team_id2) == 1
        assert team_id1 != team_id2
        return [list(team_id1)[0], list(team_id2)[0]]
        
       
    df['flattened'] = df.moments.apply(lambda ms: [flatten_moment(m,normalize) for m in ms])
    df['team_ids'] = df.moments.apply(lambda ms: get_team_ids(ms[0])) # just use the first one to determine        
    return df['flattened'].values, df['team_ids'].values

def flatten_moments_soccer(events_df,normalize):
    ''' This changes the nested list that represents single frame 
        to a 1-D array.
     '''
    df = events_df.copy()
    team_id = []
    def flatten_moment(moment,normalize):
        m = np.array(moment[0])
        if len(m) == 92:
            features = m
        elif len(m) == 46:
            # the defending team is always the first 
            # goalkeeper is the first player but for processing, is moved to the last
            features = np.concatenate((m[2:22],m[0:2], # x,y of 11 defenders
                                    m[24:44],m[22:24],    # x,y of 11 attackers
                                    m[44:46]))             # ball x,y
            '''features = np.concatenate((m[26:46],m[24:26], # x,y of 11 defenders
                                    m[4:24],m[2:4],    # x,y of 11 attackers
                                    m[0:2]))             # ball x,y'''
            if normalize:                 
                LENGTH = 52.5
                WIDTH = 34
                SHIFT0 = [0,0]
                SEQUENCE_DIMENSION = features.shape[0]

                NORMALIZE = [LENGTH, WIDTH] * int(SEQUENCE_DIMENSION/2)
                SHIFT = SHIFT0 * int(SEQUENCE_DIMENSION/2)

                
                features = np.divide(features-SHIFT, NORMALIZE)

                # for check
                feat0 = features.copy() # for check
                feat1 = np.multiply(features, NORMALIZE) + SHIFT 
                err = np.mean(np.abs(feat0-feat1))
                if err > 1e-3:
                    print('err ='+str(err))
                    # import pdb; pdb.set_trace()

        try: features = features
        except: import pdb; pdb.set_trace()
        return features
        
    df['flattened'] = df.moments.apply(lambda ms: [flatten_moment(m,normalize) for m in ms])                                     
    return df['flattened'].values, team_id

# =================================================================
# create_static_features ==========================================
# =================================================================
def create_static_features(events_df,n_pl):
    ''' Provide some static features:
            displacement, cos, sin and theta from each player to the ball, hoop 
    ''' 
    df = events_df.copy()
    def create_static_features_(moment,n_pl):
        ''' moment: flatten moment i.e. (25=10*2+3+2,)'''
        # distance of each players to the ball
        player_xy = moment[:n_pl*4]
        b_xy = moment[n_pl*4:n_pl*4+2]
        if n_pl == 5:
            hoop_xy = np.array([3.917, 25])
        elif n_pl == 11:
            hoop_xy = np.array([52.5,0])
        
        def disp_(pxy, target, n_pl):
            # dispacement to ball or goal: -pi:piz
            disp = pxy.reshape(-1, 2) - np.tile(target, (n_pl*2, 1))
            assert disp.shape[0] == n_pl*2
            r = np.sqrt(disp[:,0]**2 + disp[:, 1]**2)  
            cos_theta = np.zeros(disp.shape[0])
            sin_theta = np.zeros(disp.shape[0])
            theta = np.zeros(disp.shape[0])
            
            for i in range(disp.shape[0]):
                if r[i] != 0:
                    cos_theta[i] = disp[i, 0]/r[i] # costheta
                    sin_theta[i] = disp[i, 1]/r[i] # sintheta
                    theta[i] = np.arccos(cos_theta[i]) # theta
            return np.concatenate((r, cos_theta, sin_theta, theta))

        moment = np.concatenate((moment, disp_(player_xy, b_xy, n_pl), disp_(player_xy, hoop_xy, n_pl)))
        for pl in range(n_pl*2): # relationship between all players and defenders => all players
            player2_xy = moment[pl*2:pl*2+2]
            moment = np.concatenate((moment, disp_(player_xy, player2_xy, n_pl)))
        return moment
    # vertical stack s.t. now each event i.e. a list of moments becomes an array
    # where each row is a frame (moment)
    df['enriched'] = df.moments.apply(lambda ms: np.vstack([create_static_features_(m,n_pl) for m in ms]))
    return df['enriched'].values


# =================================================================
# create_dynamic_features =========================================
# =================================================================
def create_dynamic_features(events_df, fs, n_pl, velocity):
    ''' Add velocity for players x, y direction and bball's x,y,z direction 
    '''
    df = events_df.copy()
    def create_dynamic_features_(moments, fs, n_pl, velocity):
        ''' moments: (moments length, n existing features)'''
        ball_dim = 3 if n_pl == 5 else 2
        pxy = moments[:, :n_pl*4+ball_dim] # get the players x,y and basketball x,y,z coordinates
        next_pxy = np.roll(pxy, -1, axis=0) # get next frame value
        vel = ((next_pxy - pxy)/fs)[:-1, :] # the last velocity is not meaningful
        # when we combine this back to the original features, we shift one done,
        # i.e. [p1, p2, ..., pT] combine [_, p2-p1, ...., pT-pT_1]
        # the reason why we shift is that we don't want to leak next position info
        
        if velocity == 2:
            acc = (vel[1:,:] - vel[:-1,:])/fs
            out = np.column_stack([moments[2:, :], vel[1:, :], acc])
        else: 
            out = np.column_stack([moments[1:, :], vel])
        return out
    df['enriched'] = df.moments.apply(lambda ms: create_dynamic_features_(ms, fs, n_pl, velocity))
    return df['enriched'].values 


# =================================================================
# OneHotEncoding ==================================================
# =================================================================
class OneHotEncoding:
    '''
        Perform one hot encoding on the team id, use mapping 
        from the id_team.csv file (or you an pass your own)
    '''
    def __init__(self, cat=None):
        cat = pd.read_csv('./meta_data/id_team.csv')
        # binary encode
        # ensure uniqueness
        assert sum(cat.team_id.duplicated()) == 0
        self.mapping = dict(zip(cat.team_id, range(0, len(cat)))) # temporarily just one hot encode two teams
        # self.mapping = {1610612741:0, 1610612761:1}

    def encode(self, teams):
        nb_classes = len(self.mapping)
        targets = np.array([self.mapping[int(i)] for i in teams])
        one_hot_targets = np.eye(nb_classes)[targets]
        # print(one_hot_targets)
        return one_hot_targets.reshape(-1)

    def add_ohs(self, events, team_ids):
        return [np.column_stack((events[i], np.tile(self.encode(team_ids[i]), (len(events[i]), 1)))) for i in range(len(events))]
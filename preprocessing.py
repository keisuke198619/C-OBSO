# preprocessing.py
import glob, os, sys, math, warnings, copy, time
import numpy as np
import pandas as pd
from scipy import signal

from features import OneHotEncoding, flatten_moments, create_static_features, create_dynamic_features, flatten_moments_soccer
from hidden_role_learning import HiddenStructureLearning
from sequencing import subsample_sequence
import collections

# modifying the code https://github.com/samshipengs/Coordinated-Multi-Agent-Imitation-Learning

# ================================================================================================
# remove_non_eleven ==============================================================================
# ================================================================================================
def remove_non_eleven(events_df, event_length_th,n_pl,dataset, verbose=False):
    df = events_df.copy() # shape [frames x 8 columns] 
    # playbyplay moments  visitor orig_events  start_time_left home  quarter  end_time_left  
    if n_pl == 5:
        home_id = df.loc[0]['home']['teamid']
        away_id = df.loc[0]['visitor']['teamid']
    else:
        home_id = []
        away_id = []

    def remove_non_eleven_(moments, event_length_th,n_pl,dataset, verbose=False):
        ''' Go through each moment, when encounters balls not present on court,
            or less than 10 players, discard these moments and then chunk the following moments 
            to as another event.

            Motivations: balls out of bound or throwing the ball at side line will
                probably create a lot noise for the defend trajectory learning model.
                We could add the case where players are less than 10 (it could happen),
                but this is not allowed in the model and it requres certain input dimension.

            moments: A list of moments
            event_length_th: The minimum length of an event

            segments: A list of events (or, list of moments) e.g. [ms1, ms2] where msi = [m1, m2]
        '''

        segments = []
        segment = []
        # looping through each moment
        for i in range(len(moments)):
            # get moment dimension
            if dataset == 'nba':
                moment_dim = len(moments[i][5]) # [player&ball][5-dims]
                accurate_dim = 11 # 1 bball + 10 players 
            elif dataset == 'soccer': 
                moment_dim = len(moments[i][0]) # 46-dims
                accurate_dim = 46
            elif 'jleague' in dataset:
                try: moment_dim = len(moments[i][0]) # 92-dims
                except: moment_dim = 0
                accurate_dim = 92
            
            
            if moment_dim == accurate_dim: 
                segment.append(moments[i]) # less than ten players or basketball is not on the court
            elif moment_dim > accurate_dim:
                segment.append([moments[i][0][:accurate_dim]])    
                '''# only grab these satisfy the length threshold
                if len(segment) >= event_length_th:
                    segments.append(segment)
                # reset the segment to empty list
                segment = []'''
            else:
                segment = []
                break

        # grab the last one
        if len(segment) >= event_length_th:
            segments.append(segment)
        if False: # len(segments) == 0:
            print('Warning: Zero length event returned')
        return segments
    # process for each event (row)
    df['chunked_moments'] = df.moments.apply(lambda m: remove_non_eleven_(m, event_length_th, n_pl, dataset, verbose))
    # in case there's zero length event
    df = df[df['chunked_moments'].apply(lambda e: len(e)) != 0]
    df['chunked_moments'] = df['chunked_moments'].apply(lambda e: e[0])
    return df['chunked_moments'].values, {'home_id': home_id, 'away_id': away_id}

# ================================================================================================
# remove_outlier ================================================================================
# ================================================================================================
def remove_outlier(events_df, n_pl, verbose=False):
    df = events_df.copy() 
    len_before = len(df['moments'])

    def remove_outlier_(moments, n_pl, verbose=False):
        segments = []
        # ballxyz= moments[i][5][0][2:4] # [0]:ball(2-4:3dim), [1-10]:ball(2-3:2dim)
        pl_vxy = moments[:,506:526]
        bl_vxy = moments[:,526:528]

        bl_v = np.sqrt((bl_vxy**2).sum(axis=1))
        pl_v = [[] for _ in range(n_pl)]
        for pl in range(n_pl):
            pl_v[pl] = np.sqrt((pl_vxy[:,pl*2:pl*2+2]**2).sum(axis=1))
        max_pl_v = np.max(np.vstack(pl_v))

        if np.max(bl_v) < 20 and max_pl_v < 13: 
            segments.append(moments) 
        return segments # , outlier
    
    df['chunked_moments'] = df.moments.apply(lambda m: remove_outlier_(m, n_pl, verbose))

    # in case there's zero length event
    df = df[df['chunked_moments'].apply(lambda e: len(e)) != 0]
    df['chunked_moments'] = df['chunked_moments'].apply(lambda e: e[0])
    len_after = len(df['chunked_moments'])
    outlier = len_before - len_after 
    
    return df['chunked_moments'].values, outlier

# ================================================================================================
# filters ================================================================================
# ================================================================================================
def filters(events_df,fs):
    order = 2 # order of the filter
    Nq = 1/(2*fs) # Nyquist frequency (Hz)  
    fp = 2 # low pass frequency (Hz)         
    b, a = signal.butter(order, fp/Nq, 'low', analog=False)

    df = events_df.copy()
    data_list = []

    for m in df.moments: # moments[seq][time][feature]
        data0 = np.zeros((len(m),len(m[0]))) # time, feature
        for i in range(len(m)): # time length
            data0[i,:] = m[i]
        
        data_filt = signal.filtfilt(b, a, data0, axis=0) 

        data_list0 = []
        for i in range(len(m)): # time length
            data_list0.append(data_filt[i,:])  
        data_list.append(data_list0)    

    return data_list 

# ================================================================================================
# chunk_shotclock ================================================================================
# ================================================================================================
def chunk_shotclock(events_df, event_length_th, verbose=False):
    df = events_df.copy()
    def chunk_shotclock_(moments, event_length_th, verbose):
        ''' When encounters ~24secs or game stops, chunk the moment to another event.
            shot clock test:
            1) c = [20.1, 20, 19, None,18, 12, 9, 7, 23.59, 23.59, 24, 12, 10, None, None, 10]
              result = [[20.1, 20, 19], [18, 12, 9, 7], [23.59], [23.59], [24, 12, 10]]
            2) c = [20.1, 20, 19, None, None,18, 12, 9, 7, 7, 7, 23.59, 23.59, 24, 12, 10, None, None, 10]
              result = [[20.1, 20, 19], [18, 12, 9, 7], [7], [7], [23.59], [23.59], [24, 12, 10]]

            Motivations: game flow would make sharp change when there's 24s or 
            something happened on the court s.t. the shot clock is stopped, thus discard
            these special moments and remake the following valid moments to be next event.

            moments: A list of moments
            event_length_th: The minimum length of an event
            verbose: print out exceptions or not

            segments: A list of events (or, list of moments) e.g. [ms1, ms2] where msi = [m1, m2] 
        '''

        segments = []
        segment = []
        # naturally we won't get the last moment, but it should be okay
        for i in range(len(moments)-1):
            current_shot_clock_i = moments[i][3]
            next_shot_clock_i = moments[i+1][3]
            # sometimes the shot clock value is None, thus cannot compare
            try:
                # if the game is still going i.e. sc is decreasing
                if next_shot_clock_i < current_shot_clock_i:
                    segment.append(moments[i])
                # for any reason the game is sstopped or reset
                else:
                    # not forget the last moment before game reset or stopped
                    if current_shot_clock_i < 24.:
                        segment.append(moments[i])
                    # add length condition
                    if len(segment) >= event_length_th:
                        segments.append(segment)
                    # reset the segment to empty list
                    segment = []
            # None value
            except Exception as e:
                # not forget the last valid moment before None value
                if current_shot_clock_i != None:
                    segment.append(moments[i])    
                if len(segment) >= event_length_th:
                    segments.append(segment)
                # reset the segment to empty list
                segment = []

        # grab the last one
        if len(segment) >= event_length_th:
            segments.append(segment)            
        if False: # len(segments) == 0:
            print('Warning: Zero length event returned')
        return segments
    
    # process for each event (row)
    df['chunked_moments'] = df.moments.apply(lambda m: chunk_shotclock_(m, event_length_th, verbose))
    # in case there's zero length event
    df = df[df['chunked_moments'].apply(lambda e: len(e)) != 0]
    df['chunked_moments'] = df['chunked_moments'].apply(lambda e: e[0])
    return df['chunked_moments'].values


# ================================================================================================
# chunk_halfcourt ================================================================================
# ================================================================================================
def chunk_halfcourt(events_df, event_length_th, n_pl, verbose=False):
    df = events_df.copy()
    def chunk_halfcourt_(moments, event_length_th, n_pl, verbose):
        ''' Discard any plays that are not single sided. When the play switches 
            court withhin one event, we chunk it to be as another event
        '''

        # NBA court size 94 by 50 feet
        if n_pl == 5:
            half_court = 94/2. #np.array([], dtype='float') # feet
        else:
            half_court = 105/2. # m
        cleaned = []

        # remove any moments where two teams are not playing at either side of the court
        for i in moments:
            # the x coordinates is on the 3rd or 2 ind of the matrix,
            # the first and second is team_id and player_id
            if n_pl == 5:
                a = 5 # index of data
                team1x = np.array(i[a])[1:6, :][:, 2]    # player data starts from 1, 0 ind is bball
                team2x = np.array(i[a])[6:11, :][:, 2]
            else: # soccer
                a = 0 # index of data
                team1x = np.array(i[a])[1:6, :][:, 2]    # player data starts from 1, 0 ind is bball
                team2x = np.array(i[a])[6:11, :][:, 2]

            # if both team are on the left court:
            if sum(team1x <= half_court)==5 and sum(team2x <= half_court)==5:
                cleaned.append(i)
            elif sum(team1x >= half_court)==5 and sum(team2x >= half_court)==5:
                cleaned.append(i)

        # if teamns playing court changed during same list of moments,
        # chunk it to another event
        segments = []
        segment = []
        for i in range(len(cleaned)-1):
            current_mean = np.mean(np.array(cleaned[i][5])[:, 2], axis=0)
            current_pos = 'R' if current_mean >= half_court else 'L'
            next_mean = np.mean(np.array(cleaned[i+1][5])[:, 2], axis=0)
            next_pos = 'R' if next_mean >= half_court else 'L'

            # the next moment both team are still on same side as current
            if next_pos == current_pos:
                segment.append(cleaned[i])
            else:
                if len(segment) >= event_length_th:
                    segments.append(segment)
                segment = []
        # grab the last one
        if len(segment) >= event_length_th:
            segments.append(segment)            
        if False: # len(segments) == 0:
            print('Warning: Zero length event returned')
        return segments
    
    # process for each event (row)
    df['chunked_moments'] = df.moments.apply(lambda m: chunk_halfcourt_(m, event_length_th, n_pl, verbose))
    # in case there's zero length event
    df = df[df['chunked_moments'].apply(lambda e: len(e)) != 0]
    df['chunked_moments'] = df['chunked_moments'].apply(lambda e: e[0])
    return df['chunked_moments'].values


# ================================================================================================
# reorder_teams ==================================================================================
# ================================================================================================
def reorder_teams(events_df, game_id,n_pl):
    df = events_df.copy()
    def reorder_teams_(input_moments, game_id,n_pl):
        ''' 1) the matrix always lays as home top and away bot VERIFIED
            2) the court index indicate which side the top team (home team) defends VERIFIED

            Reorder the team position s.t. the defending team is always the first 

            input_moments: A list moments
            game_id: str of the game id
        '''
        # now we want to reorder the team position based on meta data
        if n_pl == 5:
            court_index = pd.read_csv('./meta_data/court_index.csv')
            full_court = 94.
        #else: # soccer

        court_index = dict(zip(court_index.game_id, court_index.court_position))

        half_court = full_court/2. # feet
        home_defense = court_index[int(game_id)]
        moments = copy.deepcopy(input_moments)
        for i in range(len(moments)):
            home_moment_x = np.array(moments[i][5])[1:6,2]
            away_moment_x = np.array(moments[i][5])[6:11,2]
            quarter = moments[i][0]
            # if the home team's basket is on the left
            if home_defense == 0:
                # first half game
                if quarter <= 2:
                    # if the home team is over half court, this means they are doing offense
                    # and the away team is defending, so switch the away team to top
                    if sum(home_moment_x>=half_court)==5 and sum(away_moment_x>=half_court)==5:
                        moments[i][5][1:6], moments[i][5][6:11] = moments[i][5][6:11], moments[i][5][1:6]
                        for l in moments[i][5][1:6]:
                            l[2] = full_court - l[2]
                        for l in moments[i][5][6:11]:
                            l[2] = full_court - l[2]
                        # also normalize the bball x location
                        moments[i][5][0][2] = full_court - moments[i][5][0][2]
                # second half game      
                elif quarter > 2: # second half game, 3,4 quarter
                    # now the home actually gets switch to the other court
                    if sum(home_moment_x<=half_court)==5 and sum(away_moment_x<=half_court)==5:
                        moments[i][5][1:6], moments[i][5][6:11] = moments[i][5][6:11], moments[i][5][1:6]
                    elif sum(home_moment_x>=half_court)==5 and sum(away_moment_x>=half_court)==5:
                        for l in moments[i][5][1:6]:
                            l[2] = full_court - l[2]
                        for l in moments[i][5][6:11]:
                            l[2] = full_court - l[2]
                        moments[i][5][0][2] = full_court - moments[i][5][0][2]
                else:
                    print('Should not be here, check quarter value')
            # if the home team's basket is on the right
            elif home_defense == 1:
                # first half game
                if quarter <= 2:
                    # if the home team is over half court, this means they are doing offense
                    # and the away team is defending, so switch the away team to top
                    if sum(home_moment_x<=half_court)==5 and sum(away_moment_x<=half_court)==5:
                        moments[i][5][1:6], moments[i][5][6:11] = moments[i][5][6:11], moments[i][5][1:6]
                    elif sum(home_moment_x>=half_court)==5 and sum(away_moment_x>=half_court)==5:
                        for l in moments[i][5][1:6]:
                            l[2] = full_court - l[2]
                        for l in moments[i][5][6:11]:
                            l[2] = full_court - l[2]
                        moments[i][5][0][2] = full_court - moments[i][5][0][2]
                # second half game      
                elif quarter > 2: # second half game, 3,4 quarter
                    # now the home actually gets switch to the other court
                    if sum(home_moment_x>=half_court)==5 and sum(away_moment_x>=half_court)==5:
                        moments[i][5][1:6], moments[i][5][6:11] = moments[i][5][6:11], moments[i][5][1:6]
                        for l in moments[i][5][1:6]:
                            l[2] = full_court - l[2]
                        for l in moments[i][5][6:11]:
                            l[2] = full_court - l[2]
                        moments[i][5][0][2] = full_court - moments[i][5][0][2]
                else:
                    orint('Should not be here, check quarter value')
        return moments
    return [reorder_teams_(m, game_id, n_pl) for m in df.moments.values]


# ================================================================================================
# split into train and test data  ================================================================
# ================================================================================================
def split_testdata_basket(events_df, game_id):
    df = events_df.copy()
    moments_tr = []
    moments_te = []
    qs = []

    for m in df.moments.values: # length: segments
        ''' split dataset into train and test data
            test: fourth quarther, train: otherwise    

            input_moments: A list moments
            game_id: str of the game id
        '''
        quarter = m[1][0] 
        if quarter == 4:
            moments_te.append(m)
        else:
            moments_tr.append(m)
        qs.append(quarter)

    return moments_tr,moments_te

def process_game_data(Data, game_ids, args): # event_threshold, subsample_factor,dataset,n_roles):
    def process_game_data_(game_id, events_df, args):
        event_threshold = args.event_threshold
        subsample_factor = args.subsample_factor
        n_roles = args.n_roles
        dataset = args.data
        normalize = args.normalize
        filter = args.filter
        velocity = args.velocity
            
        if dataset == 'nba':
            n_pl = 5
            fs = 1/25.
        elif dataset == 'soccer': 
            n_pl = 11
            fs = 1/10.
        elif 'jleague' in dataset:
            n_pl = 11
            fs = 1/25.

        # remove non elevens
        result, _ = remove_non_eleven(events_df, event_threshold,n_pl,dataset)
        df = pd.DataFrame({'moments': result}) # list: maybe segments*frames*data (e.g. 263*150*6)

        if dataset == 'nba': # only basketball
            # chunk based on shot clock, Nones or stopped timer
            result = chunk_shotclock(df, event_threshold)
            df = pd.DataFrame({'moments': result}) # list: maybe segments*frames*data (e.g. 106*352*6)

        # chunk based on half court and normalize to all half court
        if dataset == 'nba':
            result = chunk_halfcourt(df, event_threshold,n_pl)
            df = pd.DataFrame({'moments': result}) # list: maybe segments*frames*data (e.g. 80*261*6)
        # the direction of attacking soccer data is positive(x)

        # reorder team matrix s.t. the first five players are always defend side players
        if dataset == 'nba':
            result = reorder_teams(df, game_id,n_pl)
            df = pd.DataFrame({'moments': result}) # list: the same above

        # split into train and test data (added) ----------------------------------------------------------
        if dataset == 'nba': 
            result_tr,result_te = split_testdata_basket(df, game_id) 
            df_tr = pd.DataFrame({'moments': result_tr}) # list: the same above        
            df_te = pd.DataFrame({'moments': result_te})
            # print(len(df_tr['moments']),' + ',len(df_te['moments']))
        else:
            df_tr = df
            df_te = []

        # features 
        # flatten data
        if dataset == 'nba': 
            result_tr, team_ids_tr = flatten_moments(df_tr,normalize) # [seq:np.ndarray][t:list][26-dim]
            result_te, team_ids_te = flatten_moments(df_te,normalize)
            df_tr = pd.DataFrame({'moments': result_tr}) # list: [seqs][t][23-dim]
            df_te = pd.DataFrame({'moments': result_te}) 
        else: # if dataset == 'soccer': 
            result_tr, _ = flatten_moments_soccer(df_tr,normalize)  
            df_tr = pd.DataFrame({'moments': result_tr})  # list: [seqs][t][46-dim]

        # filter
        if filter:
            result_tr = filters(df_tr,fs)
            df_tr = pd.DataFrame({'moments': result_tr})
            if dataset == 'nba': 
                result_te = filters(df_te,fs)
                df_te = pd.DataFrame({'moments': result_te}) 

        # static features
        if dataset == 'nba' or dataset == 'soccer':
            result_tr = create_static_features(df_tr,n_pl)
            df_tr = pd.DataFrame({'moments': result_tr})
            if dataset == 'nba':
                result_te = create_static_features(df_te,n_pl)
                df_te = pd.DataFrame({'moments': result_te}) 

        # dynamic features
        if dataset == 'nba' or dataset == 'soccer':
            result_tr = create_dynamic_features(df_tr, fs, n_pl,velocity)
            df_tr = pd.DataFrame({'moments': result_tr}) 

            if dataset == 'nba':
                result_te = create_dynamic_features(df_te, fs, n_pl,velocity)
                df_te = pd.DataFrame({'moments': result_te}) 

        if dataset == 'nba': 
            # remove sequence with too high speed
            result_tr, outlier_tr = remove_outlier(df_tr, n_pl)
            df_tr = pd.DataFrame({'moments': result_tr})
            result_te, outlier_te = remove_outlier(df_te, n_pl)
            df_te = pd.DataFrame({'moments': result_te})     
            outlier = outlier_tr+outlier_te   
        else:
            outlier = 0 

        # one hot encoding
        if False: # dataset != 'soccer':
            OHE = OneHotEncoding()
            result_tr = OHE.add_ohs(result_tr, team_ids_tr)
            df_tr = pd.DataFrame({'moments': result_tr}) 
            result_te = OHE.add_ohs(result_te, team_ids_te)
            df_te = pd.DataFrame({'moments': result_te}) 

        return df_tr,df_te, outlier

    game_tr = [] 
    game_te = []
    outlier = [] 
    event_threshold = args.event_threshold
    subsample_factor = args.subsample_factor
    n_roles = args.n_roles
    dataset = args.data
    hmm_iter = args.hmm_iter
    normalize = args.normalize

    if dataset == 'nba':
        n_pl = 5
        data_unit = 'games'
        iter = args.n_GorS
    elif dataset == 'soccer':
        n_pl = 11
        data_unit = 'datasets'
        iter = len(game_ids)
    elif 'jleague' in dataset:
        n_pl = 11
        data_unit = 'games'
        iter = args.n_GorS

    for i in range(iter):
        print('working on game {0:} | {1:} out of total {2:} {3:}'.format(game_ids[i], i+1, iter,data_unit)) # len(game_ids)
        game_data = Data.load_game(game_ids[i])

        if dataset == 'nba':
            events_df = pd.DataFrame(game_data['events'])
            # data: events_df.moments[seqs][t][5]
            for l, r in zip([game_tr,game_te,outlier], process_game_data_(game_ids[i], events_df, args)):
                l.append(r)
            print('Number of events:', len(game_tr[i]),' + ',len(game_te[i]), 'outlier:',outlier[i]) # np.sum(np.vstack())

        elif dataset == 'soccer':
            data_dict = {}
            data_dict = {'events':[]}
            if 'train_data' in game_ids[i]:
                len_seqs = args.n_GorS 
            else: 
                len_seqs = len(game_data)
            for j in range(len_seqs): 
                # data_list = [[] for _ in range(game_data["sequence_%d"%(j+1)].shape[0])]

                # identify the timing where two attackers in attacking third
                data = game_data["sequence_%d"%(j+1)]
                offenses_xy = data[:,n_pl*2:n_pl*4].reshape((-1,n_pl,2)) # OF
                t_off,p_off = np.where(offenses_xy[:,:,0]>=105/6)
                if len(t_off) > 0:
                    multiple = [k for k, v in collections.Counter(t_off).items() if v > 1]
                    if len(multiple):
                        start_off = multiple[0]
                        tt = 0
                        data_list = [[] for _ in range(start_off,game_data["sequence_%d"%(j+1)].shape[0])]
                        for t in range(start_off,game_data["sequence_%d"%(j+1)].shape[0]): 
                            data_list[tt].append(game_data["sequence_%d"%(j+1)][t]) # 46-dim
                            tt += 1

                        if tt >= event_threshold:
                            data_dict2 = {}
                            data_dict2 = {'moments':data_list}
                            data_dict['events'].append(data_dict2)
                
            events_df = pd.DataFrame(data_dict['events']) # events_df.moments[seqs][t][46-dim]
            if 'train_data' in game_ids[i]:
                game_tr, _, _ = process_game_data_(game_ids[i], events_df, args)
            elif 'test_data' in game_ids[i]:
                game_te0, _, _ = process_game_data_(game_ids[i], events_df, args)
                game_te.append(game_te0)

        elif 'jleague' in dataset: # hybrid style of NBA and soccer
            data_dict = {}
            data_dict = {'events':[]}
            if 'opponent' in game_ids[i] or '2019' in game_ids[i]: 
                args.event_threshold = event_threshold
            else:
                args.event_threshold = 20
            len_ts = []
            for j in range(len(game_data)):
                # game_data_ = [[] for _ in range(len(game_data[j]))]              
                for k in range(len(game_data[j])):
                    #if 'jleague' == dataset:
                    game_data_ = np.array(game_data[j][k])
                    #else:
                    #    try: game_data_ = game_data[j].to_numpy()[k]
                    #    except: import pdb; pdb.set_trace()
                    #if k == 1:
                    #    import pdb; pdb.set_trace()
                    len_t = game_data_.shape[0]
                    len_ts.append(len_t)
                    data_list = [[] for _ in range(len_t)]  
                    # print(len_t)     
                    for t in range(len_t):
                        data_list[t].append(game_data_[t])
                    data_dict2 = {}
                    data_dict2 = {'moments':data_list}
                    data_dict['events'].append(data_dict2)
            events_df = pd.DataFrame(data_dict['events']) 

            if 'opponent' in game_ids[i] or '2019' in game_ids[i]:
                game_tr0, _, _ = process_game_data_(game_ids[i], events_df, args)
                game_tr.append(game_tr0)
            else: 
                game_te0, _, _ = process_game_data_(game_ids[i], events_df, args)
                game_te.append(game_te0)
    
    if dataset == 'nba' or 'jleague' in dataset:
        df_tr = pd.concat(game_tr, axis=0) 
        df_te = pd.concat(game_te, axis=0)
    elif dataset == 'soccer': 
        df_tr = game_tr 
        df_te = pd.concat(game_te, axis=0)

    # hidden role learning
    if hmm_iter > 0:
        print('learning hidden roles')
    else: 
        print('hidden roles is not learned')
    hmm_iter_df = int(hmm_iter*2) # 600#
    if hmm_iter > 0:
        print('learn hidden roles using train data')
    HSL_tr = HiddenStructureLearning(df_tr, [], [], n_pl, n_roles, args, libmode='hmmlearn', tol=1e-4, defend_iter=hmm_iter_df, offend_iter=hmm_iter) # 1000,1000
    result_train, HSL_d, HSL_o = HSL_tr.reorder_moment() # [seqs]frames*features 
    # for test data (added)
    if hmm_iter > 0:
        print('predict hidden roles using test data')
    HSL_te = HiddenStructureLearning(df_te, HSL_d, HSL_o, n_pl, n_roles, args, libmode='hmmlearn',tol=1e-4, defend_iter=hmm_iter, offend_iter=hmm_iter) 
    result_test,_,_ = HSL_te.reorder_moment() # ,_,_ is critical

    # subsample
    result = subsample_sequence(result_train, subsample_factor) # [seqs]frames*features 
    #print(result[0][0].shape) # ndarray: [seqs][frames][features] 
    result_te = subsample_sequence(result_test, subsample_factor) #  
    return result, result_te, HSL_d, HSL_o
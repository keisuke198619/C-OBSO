# sequencing.py
import glob, os, sys, math, warnings, copy, time
import numpy as np
import pandas as pd
import itertools
from scipy import signal
import warnings
# modifying the code https://github.com/samshipengs/Coordinated-Multi-Agent-Imitation-Learning

# ===============================================================================
# subsample_sequence ============================================================
# ===============================================================================
def subsample_sequence(events, subsample_factor, random_sample=False):
    if subsample_factor == 0 or round(subsample_factor*10)==10:
        return events
    
    def subsample_sequence_(moments, subsample_factor, random_sample=False):#random_state=42):
        ''' 
            moments: a list of moment 
            subsample_factor: number of folds less than orginal
            random_sample: if true then sample a random one from the window of subsample_factor size
        '''
        seqs = np.copy(moments)
        moments_len = seqs.shape[0]
        if subsample_factor > 0:
            n_intervals = moments_len//subsample_factor # number of subsampling intervals
        else: 

            n_intervals = int(moments_len//-subsample_factor)

        left = moments_len % subsample_factor # reminder

        if random_sample:
            if left != 0:
                rs = [np.random.randint(0, subsample_factor) for _ in range(n_intervals)] + [np.random.randint(0, left)]
            else:
                rs = [np.random.randint(0, subsample_factor) for _ in range(n_intervals)]
            interval_ind = range(0, moments_len, subsample_factor)
            # the final random index relative to the input
            rs_ind = np.array([rs[i] + interval_ind[i] for i in range(len(rs))])
            return seqs[rs_ind, :]
        else:
            if round(subsample_factor*10) == round(subsample_factor)*10: # int
                s_ind = np.arange(0, moments_len, subsample_factor)
                return seqs[s_ind, :]
            else:
                if round(subsample_factor*10) == 4: # soccer
                    up = 5
                    down = 2
            
                # only when 10 Hz undersampling in NBA (25 Hz)
                elif round(subsample_factor*10) == 25:
                    up = 2
                    down = 5
                seqs2 = signal.resample_poly(seqs, up, down, axis=0, padtype='line')
                # seqs2 = seqs2[1:-1]

                return seqs2
                          
    return [subsample_sequence_(ms, subsample_factor) for ms in events]

def get_sequences_attack(single_game, policy, sequence_length, overlap, n_pl, k_nearest, n_feat, args, velocity = 0, test = 0):
    ''' create events where each event is a list of sequences from
        single_game with required sequence_legnth and overlap

        single_game: A list of events
        sequence_length: the desired length of each event (a sequence of moments)
        overlap: how much overlap wanted for the sequence generation
 
    '''
    npl = n_pl*2
    try: index0 = np.array(range(single_game[0].shape[1])).astype(int) # length of features
    except: import pdb; pdb.set_trace()
    n_feat_in = 4

    if test == 0: 
        X_all = [np.zeros((0,sequence_length,(n_pl*4+2)*int(n_feat/2))) for _ in policy]
        Y_all = [np.zeros((0,sequence_length-1,2)) for _ in policy]
    else:
        X_all = [np.zeros((0,args.totalTimeSteps+5,(n_pl*4+2)*int(n_feat/2))) for _ in policy]
        Y_all = [np.zeros((0,args.totalTimeSteps+5-1,2)) for _ in policy]
    #    X_all = [np.zeros((0,args.totalTimeSteps-1,(n_pl*4+2)*2)) for _ in policy]
    #    Y_all = [np.zeros((0,args.totalTimeSteps-1,2)) for _ in policy]        
    I_all = [np.zeros((0,npl+3)) for _ in policy]   
    
    ''' 
    # original---(velocity)
    # soccer:
       0-254: static_feature (positions and angles)
           0-43: positions(xy: DF->OF, each goalkeeper is the last)
           44-45: ball xy
     　    46-133: relations between all players and ball, 22*(dist,cos(th),sin(th), theta)
     　    134-221: relations between all players and goal (the same above)
           222-2157: relations between all players (the same above) 22*22*4
       2158-2203: dyanmics_feature (46 velocities)  
 
    # transform into: 
        In Le's code, for all players,
        0-2+pl*npl: distance, cos, sin with the defender (if oneself, zeros)
        3-7+pl*npl: position and velocity of the player oneself
        8-10+pl*npl: distance, cos, sin with the goal
        9-12+pl*npl: distance, cos, sin with the ball
        0-(k-1)+13*npl*2+pl*k: k nearest players
        + ball position
        total: 13*(22+3)+2 = 327 (soccer) 

    ''' 
    ball_threshold_m = 2 # [m]
    ball_threshold_frame = 5 # [m]
    iii = 0
    for ii,i in enumerate(single_game):
        #if ii <329:
        #    continue
        if ii%100==0:
            try: print('sequence '+str(ii+1)+' is being processed. total: '+str(len(Y_all[0])))
            except: import pdb; pdb.set_trace()
        index = [] 
        index = np.append(index,index0[:(npl+1)*2]) # position
        if 'jleague' in args.data:
            index = np.append(index,index0[(npl+1)*2:(npl+1)*4]) # velocity 
        elif args.data == 'soccer':
            index = np.append(index,index0[2158:2158+(npl+1)*2]) # velocity

        index = index.astype(int)
        i = i[:,index]
        
        # identify players in attacking third
        if 'jleague' in args.data: 
            #if np.mean(i[-20:,44]) < 0: # ball_x
            #    i[:,::2] = - i[:,::2] # flipped
            offenses_xy = i[:,:n_pl*2].reshape((-1,n_pl,2))
        else: 
            offenses_xy = i[:,n_pl*2:n_pl*4].reshape((-1,n_pl,2)) # OF
            
        
        if 'jleague' in args.data and test == 1:
            att_3rds_ = [[] for _ in range(2)]
            att_3rds_[0] = list(zip(*np.where(np.max(offenses_xy[:,:,0] ,axis=0)>=105/6)))
            # att_3rds_[1] = list(zip(*np.where(np.min(offenses_xy[:,:,0] ,axis=0)<=-105/6)))
            # ind_ = np.argmax([len(att_3rds_[0]),len(att_3rds_[1])])
            ind_ = 0
            att_3rds = att_3rds_[ind_]
            
            if ind_ == 1:
                offenses_xy[:,:,0] = -offenses_xy[:,:,0]  # flipped
                offenses_xy[:,:,1] = -offenses_xy[:,:,1]  # flipped
                i[:,::2] = - i[:,::2] # flipped
                i[:,1::2] = - i[:,1::2] # flipped
            if len(att_3rds)>0:
                print('sequence '+str(len(att_3rds))+' in '+str(ind_)+' for No. '+str(ii+1)+' is being processed')
            else:
                print('sequence No. '+str(ii+1)+' has no attaking third players')
                continue
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try: 
                    att_3rds = list(zip(*np.where(np.max(offenses_xy[:,:,0] ,axis=0)>=105/2-16.5))) # 105/6
                except RuntimeWarning as e: 
                    att_3rds = []
                #    import pdb; pdb.set_trace()

        # compute distances
        
        ball_xy = i[:,n_pl*4:n_pl*4+2] # 44:46
        # all_pvxy = i[:,:(n_pl*4+2)*2].reshape((-1,n_pl*4+2,2))
        
        if 'jleague' in args.data:
            defenses_xy = i[:,n_pl*2:n_pl*4].reshape((-1,n_pl,2))
            all_pxy = i[:,:n_pl*4+2].reshape((-1,n_pl*2+1,2))
            all_vxy = i[:,n_pl*4+2:].reshape((-1,n_pl*2+1,2))
            #   all_vxy = offenses_xy[1:]-offenses_xy[:-1] # tentative, true: i[:,(npl+1)*2:] 
            #    all_vxy = np.concatenate([all_vxy,defenses_xy[1:]-defenses_xy[:-1],ball_xy[1:,np.newaxis,:]-ball_xy[:-1,np.newaxis,:]],1) # tentative
            #    all_vxy = np.concatenate([all_vxy,np.expand_dims(all_vxy[-1,:,:],0)],0) # tentative
        elif args.data == 'soccer':
            defenses_xy = i[:,:n_pl*2].reshape((-1,n_pl,2))
            all_pxy = i[:,:n_pl*4+2].reshape((-1,n_pl*2+1,2)) # should be modified
            all_vxy = i[:,n_pl*6+2:n_pl*8+2].reshape((-1,n_pl,2)) # offense
            all_vxy = np.concatenate([all_vxy,i[:,n_pl*4+2:n_pl*6+2].reshape((-1,n_pl,2)),i[:,n_pl*8+2:n_pl*8+4].reshape((-1,1,2))],1) # defense and ball
        
        all_pvxy = np.stack([all_pxy,all_vxy]).transpose((1,2,0,3)) # time,agents,posvel,dim

        # identify the predicted players based on distances
        distances = [[] for _ in range(n_pl)]
        distances_ball = [[] for _ in range(n_pl)]

        for offs in att_3rds: # individual
            offs = int(offs[0])
            distances[offs] = np.mean(np.sqrt(np.sum((defenses_xy-offenses_xy[:,offs:offs+1,:].repeat(n_pl,1))**2,axis=2)),axis=0)
            distances_ball[offs] = np.sqrt(np.sum((ball_xy-offenses_xy[:,offs,:])**2,axis=1))

        # compute offenses_ids
        if 'jleague' in args.data and test == 1:
            offenses_ids = list(itertools.permutations(att_3rds,2)) # A1 and A2
        else:
            offenses_ids = list(itertools.combinations(att_3rds,2))
            
        if 'jleague' in args.data and test == 1: # A2 is already determined (as index 0: A2 is the start)
            offenses_ids = [iv for iv in offenses_ids if iv[1][0]==0]
            if len(offenses_ids)==0:
                print('A2 in sequence No. '+str(ii+1)+' was not in attaking third players')
        else: 
            if len(offenses_ids)==0:
                print('sequence No. '+str(ii+1)+' does not have attaking third players')    


        players_index = [] # np.zeros((npl,len(offenses_ids)))
        k = 0
        i2 = []
        
        # 329, 368, 373, 381(0)
        #if ii==329 or 368 or 373:
        #    import pdb; pdb.set_trace()
        for offs in offenses_ids: # combination of A1 and A2
            # offs[0][0]: A1 (predicted), offs[1][0]: A2 (evaluated) 
            D1, = np.where(distances[offs[0][0]]==np.min(distances[offs[0][0]]))
            A1_x = offenses_xy[:,offs[0][0],0]
            A1_x_start = 0
            if args.data == 'soccer':
                A1_x_start = np.nonzero(A1_x>=105/6)[0][0]

            distance_ball_A1 = np.sum(distances_ball[offs[0][0]]<ball_threshold_m)

            if test == 0: 
                flag = (np.min(distances[offs[0][0]]) < 10 and distance_ball_A1 < ball_threshold_frame)
            else:
                flag = True
            if flag: 
                # within 10 m (important threshold)
                D2, = np.where(distances[offs[1][0]]==np.min(distances[offs[1][0]]))
                if D1[0]==D2[0]:
                    distance2 = distances[offs[0][0]] + distances[offs[1][0]]
                    D2, = np.where(distance2==sorted(distance2)[1])

                # offense and defense indices (A1,D1,D2,A2-A11,D3-D11,Ball)
                index = []
                if 'jleague' in args.data: # offense->defense
                    index = np.append(index,index0[offs[0][0]])
                    index = np.append(index,index0[D1[0]+n_pl])
                    index = np.append(index,index0[D2[0]+n_pl])
                    index = np.append(index,index0[offs[1][0]])
                    
                    offense_all = np.arange(n_pl)
                    defense_all = np.arange(n_pl,n_pl*2)
                else: # defense -> offense
                    index = np.append(index,index0[offs[0][0]+n_pl])
                    index = np.append(index,index0[D1[0]])
                    index = np.append(index,index0[D2[0]])
                    index = np.append(index,index0[offs[1][0]+n_pl])
                    
                    offense_all = np.arange(n_pl,n_pl*2)
                    defense_all = np.arange(n_pl)
                    
                offense_all = np.delete(offense_all,[offs[0][0],offs[1][0]])
                index = np.append(index,index0[offense_all])
                defense_all = np.delete(defense_all,[D1[0],D2[0]])
                index = np.append(index,index0[defense_all])
                index = np.append(index,index0[npl]) # ball
                # index = np.append(index,index+npl+1) # velocity
                index = index.astype(int)


                '''players_index[0,k] = offs[0][0]+n_pl
                players_index[1,k] = D1[0]
                players_index[2,k] = D2[0]
                players_index[3,k] = offs[1][0]+n_pl
                offense_all = np.arange(n_pl,n_pl*2)
                offense_all = np.delete(offense_all,[offs[0][0],offs[1][0]])
                players_index[4:n_pl+2,k] = offense_all
                defense_all = np.arange(n_pl)
                defense_all = np.delete(defense_all,[D1[0],D2[0]])
                players_index[n_pl+2:,k] = defense_all'''

                # data matrix
                # aaaa = np.arange(24).reshape((2,3,4))
                if n_feat == 4:
                    i2.append(all_pvxy[A1_x_start:,index].reshape((-1,(n_pl*4+2)*2))) # time,(dim->posvel->agents)
                elif n_feat == 2:
                    i2.append(all_pxy[A1_x_start:,index].reshape((-1,(n_pl*4+2))))
                try: players_index.append(np.hstack((ii,k,index)))
                except: import pdb; pdb.set_trace() 
                # players_index.append(index)
                k += 1
                iii += 1

        # players_index = players_index[:,:k]

        
        # output 
        for p in policy:            
            for k,(sequence0, player_ind) in enumerate(zip(i2, players_index)): #  in enumerate(i2):
                i_len = sequence0.shape[0]
                # create sequences
                if test == 0: 
                    if i_len >= sequence_length and np.sum(np.isnan(sequence0))==0: 
                        '''sequences0 = [] # same as the below two lines
                        for j in range(0, i_len-overlap, sequence_length-overlap):
                            if j + sequence_length > i_len-1:
                                sequences0.append(sequence0[-sequence_length:,:])
                            else:
                                sequences0.append(sequence0[j:j+sequence_length,:])'''
                        sequences0 = [sequence0[-sequence_length:,:] if j + sequence_length > i_len-1 else sequence0[j:j+sequence_length,:] \
                                for j in range(0, i_len-overlap, sequence_length-overlap)] # for the states
                        state = [np.roll(kk, -1, axis=0)[:, :] for kk in sequences0] # state 
                        
                        if n_feat == 4:
                            action = [np.roll(kk[:, p*n_feat+2:p*n_feat+4], -1, axis=0)[:-1, :] for kk in sequences0] 
                        elif n_feat == 2:
                            action = [np.roll(kk[:, p*n_feat+0:p*n_feat+2], -1, axis=0)[:-1, :] for kk in sequences0]
                        #X += state  
                        #Y += action  
                        #I += [np.hstack((ii,player_index)) for kk in sequences0] 
                        # I = np.array([np.hstack((ii,player_index)) for kk in sequences0])
                        I = np.array([player_ind for kk in sequences0])

                        X_all[p] = np.concatenate([X_all[p],np.array(state)],0) 
                        Y_all[p] = np.concatenate([Y_all[p],np.array(action)],0)
                        I_all[p] = np.concatenate([I_all[p], I],0) 
                        #if args.data == 'jleague':
                        #    import pdb; pdb.set_trace()
                        #    上下反転してデータ増やす

                else: # variable length
                    len_t = sequence0.shape[0]
                    sequences0 = np.ones((args.totalTimeSteps+5,sequence0.shape[1]))*9999 # args.totalTimeSteps+5 121
                    if len_t > args.totalTimeSteps+5:
                        sequences0 = sequence0[-args.totalTimeSteps-5:]
                    else:
                        sequences0[:len_t] = sequence0
                    #sequences0 = sequence0
                    #if len_t < args.totalTimeSteps+5:
                    #    sequences0 = np.concatenate([sequences0,sequences0[-1]],0)
                    if np.sum(sequence0) == 0 or len(player_ind) == 0:
                        import pdb; pdb.set_trace()
                    try: X_all[p] = np.concatenate([X_all[p],sequences0[np.newaxis,:]],0) 
                    except: import pdb; pdb.set_trace()
                    #if ii == 1:
                    #    import pdb; pdb.set_trace()
                    if n_feat == 4:
                        Y_all[p] = np.concatenate([Y_all[p],sequences0[np.newaxis,1:,p*n_feat+2:p*n_feat+4]],0)
                    elif n_feat == 2:
                        Y_all[p] = np.concatenate([Y_all[p],sequences0[np.newaxis,1:,p*n_feat+0:p*n_feat+2]],0)

                    I_all[p] = np.concatenate([I_all[p], np.expand_dims(player_ind,0)],0) 
                    # I_all[p] = np.concatenate([I_all[p], np.expand_dims(np.hstack((ii,player_index)),0)],0) 
    return X_all, Y_all, I_all

def get_sequences(single_game, policy, sequence_length, overlap, n_pl, k_nearest, n_feat, velocity = 0, in_sma=False):
    ''' create events where each event is a list of sequences from
        single_game with required sequence_legnth and overlap

        single_game: A list of events
        sequence_length: the desired length of each event (a sequence of moments)
        overlap: how much overlap wanted for the sequence generation
 
    '''

    X_all = []
    Y_all = []   
    
    ''' 
    # original---(velocity)
    # basketball:
       0-254: static_feature (positions and angles)
           0-19: positions(xy: DF->OF)
           20-22: ball xyz
           23-25: quarter,time_left,shot clock 
     　    26-35: relations between all players and ball, 36-45:cos(th), 46-55:sin(th), 56-65:theta
     　    66-105: relations between all players and goal (the same above)
           106-505: relations between all players  (the same above)
       506-528: dyanmics_feature (23 velocities)  
     　529-578: one-hot_feature（25 team one-hot but actually 30 teams, DF->OF)

     # soccer:
       0-254: static_feature (positions and angles)
           0-43: positions(xy: DF->OF, each goalkeeper is the last)
           44-45: ball xy
     　    46-133: relations between all players and ball, 22*(dist,cos(th),sin(th), theta)
     　    134-221: relations between all players and goal (the same above)
           222-2157: relations between all players (the same above) 22*22*4
       2158-2203: dyanmics_feature (46 velocities)  
 
    # transform into: 
        In Le's code, for all players,
        0-2+pl*npl: distance, cos, sin with the defender (if oneself, zeros)
        3-7+pl*npl: position and velocity of the player oneself
        8-10+pl*npl: distance, cos, sin with the goal
        9-12+pl*npl: distance, cos, sin with the ball
        0-(k-1)+13*npl*2+pl*k: k nearest players
        + ball position (+ team one-hot)
        total: 13*(22+3)+2 = 327 (soccer) or 13*(10+3)+3+50 = 222 (NBA)

    # original---(acceleration)
    # basketball:
       0-254: static_feature (positions and angles)
           0-19: positions(xy: DF->OF)
           20-22: ball xyz
           23-25: quarter,time_left,shot clock 
     　    26-35: relations between all players and ball, 36-45:cos(th), 46-55:sin(th), 56-65:theta
     　    66-105: relations between all players and goal (the same above)
           106-505: relations between all players  (the same above)
       506-528: dyanmics_feature (23 velocities)  
       529-551: dyanmics_feature (23 acceleration)  
     　552-601: one-hot_feature（25 team one-hot but actually 30 teams, DF->OF)

     # soccer:
       0-254: static_feature (positions and angles)
           0-43: positions(xy: DF->OF, each goalkeeper is the last)
           44-45: ball xy
     　    46-133: relations between all players and ball, 22*(dist,cos(th),sin(th), theta)
     　    134-221: relations between all players and goal (the same above)
           222-2157: relations between all players (the same above) 22*22*4
       2158-2203: dyanmics_feature (46 velocities)  
       2204-2249: dyanmics_feature (46 acceleartion) 
 
    # transform into: 
        In Le's code, for all players,
        0-2+pl*npl: distance, cos, sin with the defender (if oneself, zeros)
        3-7+pl*npl: position and velocity of the player oneself
        8-10+pl*npl: distance, cos, sin with the goal
        9-12+pl*npl: distance, cos, sin with the ball
        0-(k-1)+13*npl*2+pl*k: k nearest players
        + ball pos/vel (+ team one-hot)
        total: 15*22 + 4 = 334 (soccer) or 15*10 + 4 = 154 (NBA)

    ''' 
    npl = n_pl*2
    index0 = np.array(range(single_game[0].shape[1])).astype(int) # length of features

    for p in policy:
        X = []
        Y = []
        # create index
        index = [] 
        if n_pl == 5:
            for pl in range(npl):
                if not in_sma:
                    index = np.append(index,index0[106+pl+p*npl*4]) # distance between players 0
                    index = np.append(index,index0[116+pl+p*npl*4]) # cos 1
                    index = np.append(index,index0[126+pl+p*npl*4]) # sin 2
                index = np.append(index,index0[pl*2:pl*2+2]) # positions 3-4
                if velocity >= 0:
                    index = np.append(index,index0[506+pl*2:506+pl*2+2]) # velocities 5-6
                if velocity == 2:
                    index = np.append(index,index0[529+pl*2:529+pl*2+2]) # acceleration 
                if not in_sma:    
                    index = np.append(index,index0[66+pl:95+pl:10]) # relation with the goal 7-9 (th is not used)
                    index = np.append(index,index0[26+pl:55+pl:10]) # relation with the ball 10-12
            # k nearest players  
            if k_nearest > 0 and k_nearest < 10: # players regardless of attackers and defenders
                index = np.append(index,np.zeros(n_feat*k_nearest)) # temporary
            
            index = np.append(index,index0[20:22]) # ball positions (excluding 3d)
            if velocity >= 0:
                index = np.append(index,index0[526:528])  # ball velocity (excluding 3d)
            #if velocity == 2:
            #    index = np.append(index,index0[549:551])
            # index = np.append(index,index0[529:579]) # team one-hot
        elif n_pl == 11:
            for pl in range(npl):
                if not in_sma:
                    index = np.append(index,index0[222+pl+p*npl*4]) # distance between players 0
                    index = np.append(index,index0[244+pl+p*npl*4]) # cos 1
                    index = np.append(index,index0[266+pl+p*npl*4]) # sin 2
                index = np.append(index,index0[pl*2:pl*2+2]) # positions 3-4
                if velocity >= 0:
                    index = np.append(index,index0[2158+pl*2:2158+pl*2+2]) # velocities 5-6
                if velocity == 2:
                    index = np.append(index,index0[2204+pl*2:2204+pl*2+2]) # velocities 5-6
                if not in_sma:
                    index = np.append(index,index0[134+pl:134+npl*3+pl-1:npl]) # relation with the goal 7-9 (th is not used)
                    index = np.append(index,index0[46+pl:46+npl*3+pl-1:npl]) # relation with the ball 10-12
            # k nearest players    
            if k_nearest > 0 and k_nearest < 10: # players regardless of attackers and defenders
                index = np.append(index,np.zeros(n_feat*k_nearest)) # temporary
            
            index = np.append(index,index0[44:46]) # ball positions
            if velocity >= 0:
                index = np.append(index,index0[2202:2204]) # ball velocity
            #if velocity == 2:
            #    index = np.append(index,index0[2248:2250])

        index = index.astype(int)
        #index = np.array([p*2,p*2+1, \
        #    25+p,35+p,45+p,55+p,65+p,75+p,85+p,95+p,\
        #    p*2+105,p*2+106])
        for i in single_game:
            i_len = len(i)
            i2 = np.array(i) # copy
            sequence0 = np.zeros((i_len,index.shape[0]))
            
            for t in range(i_len):
                # nearest players
                if k_nearest > 0 and k_nearest < 10: # players regardless of attackers and defenders
                    dist = i[t][index[0:npl*n_feat:n_feat]] # index of distances
                    ind_nearest = dist.argsort()[0:(k_nearest+1)] 
                    ind_nearest = ind_nearest[np.nonzero(ind_nearest)][:k_nearest] # eliminate zero and duplication
                    for k in range(k_nearest):
                        index[n_feat*npl+k*n_feat:n_feat*npl+(k+1)*n_feat] = index[ind_nearest[k]*n_feat:ind_nearest[k]*n_feat+n_feat]
                sequence0[t,:] = i2[t,index].T
            
            # create sequences
            if i_len >= sequence_length:
                sequences0 = [sequence0[-sequence_length:,:] if j + sequence_length > i_len-1 else sequence0[j:j+sequence_length,:] \
                    for j in range(0, i_len-overlap, sequence_length-overlap)] # for the states
                #sequences = [np.array(i[-sequence_length:]) if j + sequence_length > i_len-1 else np.array(i[j:j+sequence_length]) \
                #    for j in range(0, i_len-overlap, sequence_length-overlap)] # for the actions     

                state = [np.roll(kk, -1, axis=0)[:-1, :] for kk in sequences0] # state: drop the last row as the rolled-back is not real
                
                if velocity == 2:
                    action = [np.roll(kk[:, p*n_feat+3:p*n_feat+9], -1, axis=0)[:-1, :] for kk in sequences0] 
                    # action2 = [np.roll(kk[:, p*2:p*2+2], -1, axis=0)[:-1, :] for kk in sequences] 
                elif velocity == 1:
                    action = [np.roll(kk[:, p*n_feat+3:p*n_feat+7], -1, axis=0)[:-1, :] for kk in sequences0] 
                elif velocity:
                    action = [np.roll(kk[:, [p*n_feat+5,p*n_feat+6,p*n_feat+3,p*n_feat+4]], -1, axis=0)[:-1, :] for kk in sequences0] 
                else: # position only
                    action = [np.roll(kk[:, p*n_feat+3:p*n_feat+5], -1, axis=0)[:-1, :] for kk in sequences0] 
                    # action = [np.roll(kk[:, p*2:p*2+2], -1, axis=0)[:-1, :] for kk in sequences] # action    
                # sequences = [l[:-1, :] for l in sequences] # since target has dropped one then sequence also drop one
                X += state  
                Y += action  
        X_all.append(X) 
        Y_all.append(Y) 
    return X_all, Y_all


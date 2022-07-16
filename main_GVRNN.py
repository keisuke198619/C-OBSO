from datetime import datetime
from math import sqrt
import glob, os, sys, math, warnings
import copy, time 
from copy import deepcopy
import argparse
import random

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import hmmlearn 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

# customized ftns 
from vrnn.models import load_model
from vrnn.models.utils import num_trainable_params
from vrnn.datasets import GeneralDataset
from preprocessing import *
from helpers import *
from utilities import *
from sequencing import get_sequences, get_sequences_attack

#from scipy import signal

# modifying the codes
# https://github.com/samshipengs/Coordinated-Multi-Agent-Imitation-Learning
# https://github.com/ezhan94/multiagent-programmatic-supervision

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--n_GorS', type=int, required=True)
parser.add_argument('--n_roles', type=int, required=True)
parser.add_argument('--val_devide', type=int, default=10)
parser.add_argument('--hmm_iter', type=int, default=0)
parser.add_argument('-t_step', '--totalTimeSteps', type=int, default=95)
parser.add_argument('--overlap', type=int, default=40)
parser.add_argument('-k','--k_nearest', type=int, default=0)
parser.add_argument('--batchsize', type=int, required=True)
parser.add_argument('--n_epoch', type=int, required=True)
parser.add_argument('--attention', type=int, default=-1)
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('-ev_th','--event_threshold', type=int, required=True, help='event with frames less than the threshold will be removed')
parser.add_argument('--fs', type=int, default=10)
# parser.add_argument('-subs_fac','--subsample_factor', type=int, required=True, help='too much data should be downsampled by subs_fac')
# parser.add_argument('--filter', action='store_true')
parser.add_argument('--body', action='store_true')
parser.add_argument('--acc', type=int, default=0)
parser.add_argument('--vel_in', action='store_true')
parser.add_argument('--in_out', action='store_true')
parser.add_argument('--wo_cross', action='store_true')
# parser.add_argument('--in_sma', action='store_true')
# parser.add_argument('--meanHMM', action='store_true')
parser.add_argument('--cont', action='store_true')
parser.add_argument('--numProcess', type=int, default=16)
parser.add_argument('--TEST', action='store_true')
parser.add_argument('--Sanity', action='store_true')
parser.add_argument('--hard_only', action='store_true')
parser.add_argument('--wo_macro', action='store_true')
parser.add_argument('--res', action='store_true')
parser.add_argument('--jrk', type=float, default=0)
parser.add_argument('--lam_acc', type=float, default=0)
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--pretrain2', type=int, default=0)
parser.add_argument('--attack_defend', action='store_true')
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--drop_ind', action='store_true')
args, _ = parser.parse_known_args()

# directories
main_dir = '../' # './'
game_dir = main_dir+'data_'+args.data+'/'
Data = LoadData(main_dir, game_dir, args.data)
if args.data == 'jleague':
    path_weight = '../VRNN_Jleague_data/weights/' 
elif args.data == 'jleague2':
    path_weight = '../VRNN_Jleague_data/weights2/'   
# path_output = '../VRNN_Jleague_data/outputs/' 
def run_epoch(train,rollout,hp):
    loader = train_loader if train == 1 else val_loader if train == 0 else test_loader
 
    losses = {} 
    losses2 = {}
    i = 0 
    for batch_idx, (data, macro_intents, ind_player) in enumerate(loader):
        # print(str(batch_idx))
        d1 = {'batch_idx': batch_idx}
        hp.update(d1)

        if args.cuda:
            data = data.cuda() #, data_y.cuda()
            if 'MACRO' in args.model: 
                macro_intents = macro_intents.cuda() 
        # (batch, agents, time, feat) => (time, agents, batch, feat)
        data = data.permute(2, 1, 0, 3) #, data.transpose(0, 1)
        if 'MACRO' in args.model: 
            macro_intents = macro_intents.transpose(0, 1)
        
        if train == 1:
            if 'MACRO' in args.model: 
                batch_losses, batch_losses2 = model(data, rollout, train, macro_intents, hp=hp)
            else:
                batch_losses, batch_losses2 = model(data, rollout, train, hp=hp)
            optimizer.zero_grad()
            total_loss = sum(batch_losses.values())
            total_loss.backward()
            if hp['model'] != 'RNN_ATTENTION': 
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        else:
            '''if 'MACRO' in args.model:
                if hp['pretrain']:
                    batch_losses, batch_losses2 = model(data, rollout, train, macro_intents, hp=hp)
                else: 
                    _, _, _, batch_losses, batch_losses2 = model.sample(data, macro_intents, rollout=True, burn_in=hp['burn_in'], L_att=hp['L_att'])
            else:'''
            _, _, _, batch_losses, batch_losses2 = model.sample(data, rollout=True, burn_in=hp['burn_in'], L_att=hp['L_att'])
        
        for key in batch_losses:
            if batch_idx == 0:
                losses[key] = batch_losses[key].item()
            else:
                losses[key] += batch_losses[key].item()
        
        for key in batch_losses2:
            if batch_idx == 0:
                try: losses2[key] = batch_losses2[key].item()
                except: import pdb; pdb.set_trace()
            else:
                losses2[key] += batch_losses2[key].item()
        #if i > 3:
        #    break
        i += 1

    for key in losses:
        losses[key] /= len(loader.dataset)
    for key in losses2:
        losses2[key] /= len(loader.dataset)
    return losses, losses2

def loss_str(losses):
    ret = ''
    if args.n_roles < 5:
        for key in losses:
            if 'L_rec' in key or 'L_kl' in key: # or 'L_vel' in key :
                ret += ' {}: {:.0f} |'.format(key, losses[key])
            elif 'pos' in key or 'e_vel' in key or 'e_pmax' in key or 'e_vmax' in key:
                ret += ' {}: {:.3f} |'.format(key, losses[key])

    else:
        for key in losses:
            if 'L' in key and not 'mac' in key and not 'vel' in key and not 'acc' in key and not 'jrk' in key  :
                ret += ' {}: {:.0f} |'.format(key, losses[key])
            elif 'jrk' in key or 'vel' in key or 'acc' in key  :
                ret += ' {}: {:.3f} |'.format(key, losses[key])
            else: 
                ret += ' {}: {:.3f} |'.format(key, losses[key])
    return ret[:-2]


def batch_error(predict, true):
    error = np.sqrt(np.sum((predict[:,:2] - true[:,:2])**2,1))
    return error

def unnormalize(data,args):
    # not used (maybe wrong)!!!
    if args.normalize:
        if args.dataset == 'nba':
            feet_m = 0.3048
            LENGTH = 94*feet_m
            WIDTH = 50*feet_m
            SHIFT0 = [0,0] # [47*feet_m,25*feet_m]
        elif args.dataset  == 'soccer':
            LENGTH = 52.5
            WIDTH = 34
            SHIFT0 = [0,0]
        
        dim = data.ndim
        SEQUENCE_DIMENSION = data.shape[-1]
        NORMALIZE = np.array([LENGTH, WIDTH]) * int(SEQUENCE_DIMENSION/2)
        SHIFT = SHIFT0 * int(SEQUENCE_DIMENSION/2)

        if dim == 2: 
            NORMALIZE = np.tile(NORMALIZE, (data.shape[0], 1))
        data = np.multiply(data, NORMALIZE) # + SHIFT

    return data

def label_macro_intents(data,window_size=0):
    """Computes and saves labeling functions for basketball.
    Args:
        window_size (int): If positive, will label macro-intents every window_size timesteps.
                            Otherwise, will label stationary positions as macro-intents.
    """

    N_AGENTS, N, SEQUENCE_LENGTH, SEQUENCE_DIMENSION = data.shape
    n_all_agents = 10 if N_AGENTS == 5 else 22
    n_feat = int((SEQUENCE_DIMENSION-4)/n_all_agents)

    # Compute macro-intents
    macro_intents_all = np.zeros((N, SEQUENCE_LENGTH, N_AGENTS)) # data.shape[1]

    for i in range(N):
        for k in range(N_AGENTS):
            if n_feat < 10:
                data_in = data[0,i,:,2*k:2*k+2]
            else:
                data_in = data[0,i,:,2*k+3:2*k+5]
            if window_size > 0:
                macro_intents_all[i,:,k] = compute_macro_intents_fixed(data_in, N_AGENTS, window=window_size)
            else:
                macro_intents_all[i,:,k] = compute_macro_intents_stationary(data_in, N_AGENTS)

    return macro_intents_all

def compute_macro_intents_stationary(track,N_AGENTS):
    """Computes macro-intents as next stationary points in the trajectory."""
    
    SPEED_THRESHOLD = 0.5*0.3048 

    velocity = track[1:,:] - track[:-1,:]
    speed = np.linalg.norm(velocity, axis=-1)
    stationary = speed < SPEED_THRESHOLD
    stationary = np.append(stationary, True) # assume last frame always stationary

    T = len(track)
    macro_intents = np.zeros(T)
    for t in reversed(range(T)):
        if t+1 == T: # assume position in last frame is always a macro intent
            macro_intents[t] = get_macro_intent(track[t],N_AGENTS,t)
        elif stationary[t] and not stationary[t+1]: # from stationary to moving indicated a change in macro intent
            macro_intents[t] = get_macro_intent(track[t],N_AGENTS,t)
        else: # otherwise, macro intent is the same
            macro_intents[t] = macro_intents[t+1]
    return macro_intents 

def get_macro_intent(position,N_AGENTS,t):
    """Computes the macro-intent index."""
    N_MACRO_X = 9 if N_AGENTS == 5 else 17#26#34# # 105m/2/3
    N_MACRO_Y = 10 if N_AGENTS == 5 else 11#17#22# # 68m/2/3
    MACRO_SIZE = 50*0.3048/N_MACRO_Y if N_AGENTS == 5 else 34/N_MACRO_Y

    eps = 1e-4 # hack to make calculating macro_x and macro_y cleaner
    
    if N_AGENTS == 5: 
        x = bound(position[0], 0, N_MACRO_X*MACRO_SIZE-eps)
        y = bound(position[1], 0, N_MACRO_Y*MACRO_SIZE-eps)
        macro_x = int(x/MACRO_SIZE)
        macro_y = int(y/MACRO_SIZE)
        macro = macro_x*N_MACRO_Y + macro_y
    else:
        x = bound(position[0], -N_MACRO_X*MACRO_SIZE+eps, N_MACRO_X*MACRO_SIZE-eps)
        y = bound(position[1], -N_MACRO_Y*MACRO_SIZE+eps, N_MACRO_Y*MACRO_SIZE-eps)
        macro_x = int(x/MACRO_SIZE) + N_MACRO_X
        macro_y = int(y/MACRO_SIZE) + N_MACRO_Y
        macro = macro_x*N_MACRO_Y*2 + macro_y
        # if np.isnan(macro) or macro < 0:
    return macro

def bound(val, lower, upper):
    """Clamps val between lower and upper."""
    if val < lower:
        return lower
    elif val > upper:
        return upper
    else:
        return val

if __name__ == '__main__':
    numProcess = args.numProcess  
    os.environ["OMP_NUM_THREADS"]=str(numProcess) 
    TEST = args.TEST

    # pre-process----------------------------------------------
    args.meanHMM = True if args.hmm_iter>0 else False # sorting sequences using meanHMM
    args.in_sma = True # small multi-agent data
    # normalize = False
    acc = args.acc # output: 0: vel, 1: pos+vel, 2:vel+acc, 3: pos+vel+acc
    args.vel_in = 1 if args.vel_in else 2 # input 1: vel 2: vel+acc
    if acc == -1:
        args.vel_in = -1 # position only
    elif acc == 0 or acc == 1:
        args.vel_in = 1    
    vel_in = args.vel_in
    args.velocity = args.vel_in
    #args.hmm_iter = 500
    args.filter = True
    assert not (args.in_out and args.in_sma)
    assert not (args.vel_in == 1 and acc >= 2)

    # all game ids file name, note that '/' or '\\' depends on the environment
    # all_games_id = [i.split(os.sep)[-1].split('.')[0] for i in glob.glob(game_dir+'/*.pkl')]
    if args.totalTimeSteps == 56:
        if args.data == 'jleague':
            all_games_id = ['opponent_attack_seq','FM_shot_data_addvel_4sec']
        elif args.data == 'jleague2':
            all_games_id = ['attack_seq_tracking_2019','shot_tracking4sec_2020']
    elif args.totalTimeSteps == 76:
        all_games_id = ['opponent_attack_seq','FM_shot_data_addvel_6sec']
    elif args.totalTimeSteps == 96:
        all_games_id = ['opponent_attack_seq','FM_shot_data_addvel_8sec']
    elif args.totalTimeSteps == 116:
        all_games_id = ['opponent_attack_seq','FM_shot_data_addvel_10sec']
    
    global fs
    # fs = 1/args.fs
    if args.data == 'nba':
        n_pl = 5
        args.fs = 10
        fs = 1/10
        subsample_factor = 25*fs
    elif args.data == 'soccer':
        n_pl = 11
        fs = 1/10
        subsample_factor = 10*fs  
    elif 'jleague' in args.data:
        n_pl = 11   
        fs = 1/10
        subsample_factor = 25*fs
    

    args.subsample_factor = subsample_factor
    event_threshold = args.event_threshold
    n_roles = args.n_roles
    n_GorS = args.n_GorS # games if NBA and seqs if soccer
    val_devide = args.val_devide
    batchSize = args.batchsize # 
    overlapWindow = args.overlap # 
    totalTimeSteps =  args.totalTimeSteps # 

    # save the processed file to disk to avoid repeated work
    
    game_file0 = '../VRNN_Jleague_data/all_'+args.data+'_games_'+str(n_GorS)+'_'
    game_file0 = game_file0 + 'unnorm' # if not args.normalize else game_file0 + 'norm'

    game_file0 = game_file0 + '_filt'
    game_file0 = game_file0 + '_acc'
    k_nearest = args.k_nearest # 3
    if k_nearest==0:
        game_file0 = game_file0 + '_k0'

    if args.meanHMM:
        game_file0 = game_file0 + '_meanHMM'

    if args.attack_defend:
        game_file0 = game_file0 + '_roles_' + str(args.n_roles) +'_'

    game_file0 = game_file0 + '/'
    if not os.path.isdir(game_file0):
        os.makedirs(game_file0)

    game_files_pre = game_file0 + '_pre'

    game_file0 = game_file0 + 'Fs' + str(args.fs) 
    #if acc == -1:
    #    game_file0 = game_file0 + '_pos'
    if args.vel_in == 1:
        game_file0 = game_file0 + '_vel'
    if args.in_sma:
        game_file0 = game_file0 + '_inSimple'
    elif args.in_out:
        game_file0 = game_file0 + '_inout'
    #if args.normalize:
    #    game_file0 = game_file0 + '_norm' 
    game_file0 = game_file0 + '_' + str(batchSize) + '_' + str(totalTimeSteps)
    game_files = game_file0
    game_files_val = game_file0 + '_val'+'.pkl'
    game_files_te = game_file0 + '_te'+'.pkl'    

    activeRoleInd = range(n_roles)
    activeRole = []; 
    activeRole.extend([str(n) for n in range(n_roles)]) # need to be reconsidered

    if acc==0 or acc==-1 or acc == 4: # vel/pos/acc only
        outputlen0 = 2
    elif acc==3: # all
        outputlen0 = 6
    else:
        outputlen0 = 4
        
    numOfPrevSteps = 1 # We are only looking at the most recent character each time. 
    totalTimeSteps_test = totalTimeSteps # -4 if args.data == 'jleague' else totalTimeSteps
    if args.in_sma:
        n_feat = 6 if vel_in == 2 else 4
        if acc == -1:
            n_feat = 2 
    elif args.in_out:
        n_feat = 6 if vel_in == 2 else 4
    else:
        n_feat = 15 if vel_in == 2 else 13

    if os.path.isfile(game_files+'_te_0.pkl'): 
        print(game_files+'_te_0.pkl'+' can be loaded')
        with open(game_files+'_te_0.pkl', 'rb') as f:
            tmp_data,tmp_label,tmp_index,len_seqs_test = np.load(f,allow_pickle=True) 
        print('load '+game_files+'_tr0.pkl')
        
        try: 
            with open(game_files+'_tr'+str(0)+'.pkl', 'rb') as f:
                X_all,len_seqs_val,_,macro_intents = np.load(f,allow_pickle=True)         
        except: import pdb; pdb.set_trace()
        #if args.totalTimeSteps < 116:      

    else:
        if os.path.isfile(game_files_pre+'.pkl'):
            print(game_files_pre+'.pkl will be loaded')
            with open(game_files_pre+'.pkl', 'rb') as f:
                game_data,game_data_te = np.load(f,allow_pickle=True)[:2] # ,_,_

        else: 
            print(game_files_pre+'.pkl is not existed then will be created')
            game_data,game_data_te,HSL_d,HSL_o = process_game_data(Data, all_games_id, args) 
            with open(game_files_pre+'.pkl', 'wb') as f:
                pickle.dump([game_data,game_data_te,HSL_d,HSL_o], f, protocol=4)

        print('Final number of events:', len(game_data), '+', len(game_data_te)) # 
        game_ind = np.arange(len(game_data))
        if args.data == 'soccer':
            game_train, game_test,_,_ = train_test_split(game_ind, game_ind, test_size=1/val_devide, random_state=42)
            game_data_te = [game_data[i] for i in game_test] 
            game_data = [game_data[i] for i in game_train] 
        elif 'jleague' in args.data:
            len_time_te = np.array([len(game_data_te[i]) for i in range(len(game_data_te))])
            len_time_tr = np.array([len(game_data[i]) for i in range(len(game_data))])

        # create sequences -----------------------------------------------------------
        create_Train = True
        if create_Train:
            if args.attack_defend:
                X_train_all,Y_train_all,I_train_all = get_sequences_attack(game_data, activeRoleInd, 
                    totalTimeSteps+5, overlapWindow, n_pl, k_nearest, n_feat, args, vel_in, test=0)       
            else: 
                X_train_all,Y_train_all = get_sequences(game_data, activeRoleInd, 
                    totalTimeSteps+5, overlapWindow, n_pl, k_nearest, n_feat, vel_in, args.in_sma) # [role][seqs][steps,feats]
        
            if args.in_out:
                X_train_all = Y_train_all
            print('get train sequences')
            del game_data # -------------



            # split train/validation
            len_seqs = len(X_train_all[0]) 
            X_ind = np.arange(len_seqs)
            try: ind_train, ind_val,_,_ = train_test_split(X_ind, X_ind, test_size=1/val_devide, random_state=42)
            except: import pdb; pdb.set_trace()
            featurelen = X_train_all[0][0].shape[1] 
            len_seqs_tr = len(ind_train)
            offSet_tr = math.floor(len_seqs_tr / batchSize)
            batchSize_val = len(ind_val)

            X_all = np.zeros([n_roles, len(ind_train), totalTimeSteps+5, featurelen])
            X_val_all = np.zeros([n_roles, len(ind_val), totalTimeSteps+5, featurelen])
            for i, X_train in enumerate(X_train_all):
                i_tr = 0; i_val = 0
                for b in range(len_seqs):  
                    if set([b]).issubset(set(ind_train)):
                        for r in range(totalTimeSteps+5):
                            try: X_all[i][i_tr][r][:] = np.squeeze(X_train[b][r,:])
                            except: import pdb; pdb.set_trace()
                        i_tr += 1
                    else:
                        for r in range(totalTimeSteps+5):
                            X_val_all[i][i_val][r][:] = np.squeeze(X_train[b][r,:])
                        i_val += 1

            print('create train sequences')
            
            del X_train_all

            # macro intents
            macro_intents = label_macro_intents(X_all)
            macro_intents_val = label_macro_intents(X_val_all)
            
            len_seqs_val = len(X_val_all[0])
        

            J = 8
            batchval = math.ceil(len_seqs_val/J)
            for j in range(J):
                if j < J-1:
                    tmp_data = X_val_all[:,j*batchval:(j+1)*batchval,:,:]
                    tmp_label = macro_intents_val[j*batchval:(j+1)*batchval,:,:]
                else:
                    tmp_data = X_val_all[:,j*batchval:,:,:]
                    tmp_label = macro_intents_val[j*batchval:,:,:]         
                with open(game_files+'_val_'+str(j)+'.pkl', 'wb') as f:
                    pickle.dump([tmp_data,tmp_label], f, protocol=4)            

        # for test data-------------
        if args.attack_defend:
            test = 1 if 'jleague' in args.data else 0
            X_test_all,Y_test_all,I_test_all = get_sequences_attack(game_data_te, activeRoleInd, 
                totalTimeSteps+5, overlapWindow, n_pl, k_nearest, n_feat, args, vel_in, test=test)
        else :
            X_test_all,Y_test_all = get_sequences(game_data_te, activeRoleInd, 
                totalTimeSteps+5, overlapWindow, n_pl, k_nearest, n_feat, vel_in, args.in_sma) # [role][seqs][steps,feats]
        del game_data_te
        if args.in_out:
            X_test_all = Y_test_all

        featurelen = X_test_all[0][0].shape[1] 
        len_seqs_test = len(X_test_all[0])  
        batchSize_test = len_seqs_test # args.batchsize # 32
        len_seqs_test0 = len_seqs_test
        ind_test = np.arange(len_seqs_test)

        if args.data == 'nba':
            X_ind = np.arange(len_seqs_test)
            _, ind_test,_,_ = train_test_split(X_ind, X_ind, test_size=1/3, random_state=42)
            len_seqs_test = len(ind_test)

        X_test_test_all = np.zeros([n_roles, len_seqs_test, totalTimeSteps_test+5, featurelen]) 
        for i, X_test in enumerate(X_test_all):
            i_te = 0
            for b in range(len_seqs_test0):
                if args.data == 'nba':
                    if set([b]).issubset(set(ind_test)):
                        for r in range(totalTimeSteps+5):
                            X_test_test_all[i][i_te][r][:] = np.squeeze(X_test[b][r,:])
                        i_te += 1                    
                elif args.data == 'soccer' or 'jleague' in args.data:
                    for r in range(totalTimeSteps_test+5):
                        X_test_test_all[i][b][r][:] = np.squeeze(X_test[b][r,:])

        print('create test sequences')

        if create_Train:
            if offSet_tr > 0: 
                for j in range(offSet_tr):
                    tmp_data = X_all[:,j*batchSize:(j+1)*batchSize,:,:]
                    tmp_label = macro_intents[j*batchSize:(j+1)*batchSize,:,:]
                    with open(game_files+'_tr'+str(j)+'.pkl', 'wb') as f:
                        pickle.dump([tmp_data,len_seqs_val,len_seqs_test, tmp_label], f, protocol=4) 
            else: 
                import pdb; pdb.set_trace()

        J = 8
        macro_intents_te = label_macro_intents(X_test_test_all)
        batchte = math.ceil(len_seqs_test/J)
        I_test_all = np.array(I_test_all)
        for j in range(J):
            tmp_data = X_test_test_all[:,j*batchte:(j+1)*batchte,:,:]
            tmp_label = macro_intents_te[j*batchte:(j+1)*batchte,:,:]
            if args.attack_defend:
                tmp_index = I_test_all[0,j*batchte:(j+1)*batchte]
                if np.sum(tmp_data) == 0 or len(tmp_index) == 0:
                    import pdb; pdb.set_trace()        
            with open(game_files+'_te_'+str(j)+'.pkl', 'wb') as f:
                if args.attack_defend:
                    pickle.dump([tmp_data,tmp_label,tmp_index,len_seqs_test], f, protocol=4) 
                else:
                    pickle.dump([tmp_data,tmp_label], f, protocol=4)     
        #with open(game_files_te, 'wb') as f:
        #    pickle.dump([X_test_test_all,macro_intents_te], f, protocol=4)     
        
        del X_test_test_all, tmp_data
        

        print('save train and test sequences')
        with open(game_files+'_tr'+str(0)+'.pkl', 'rb') as f:
            X_all,len_seqs_val,_,macro_intents = np.load(f,allow_pickle=True) 
        # if args.totalTimeSteps < 116:      
        with open(game_files+'_te_0.pkl', 'rb') as f:
            _,_,_,len_seqs_test = np.load(f,allow_pickle=True) 
    # count batches 
    offSet_tr =  len(glob.glob(game_files+'_tr*.pkl'))
    # variables
    featurelen = X_all.shape[3] #[0][0][0]#see get_sequences in sequencing.py
    len_seqs_tr = batchSize*offSet_tr
    print('featurelen: '+str(featurelen)+' train_seqs: '+str(len_seqs_tr)+' val_seqs: '+str(len_seqs_val)+' test_seqs: '+str(len_seqs_test))

    # parameters for VRNN -----------------------------------
    init_filename0 = path_weight+ 'sub' + str(args.fs) + '_'
    init_filename0 = init_filename0 + 'filt_'  
    if args.vel_in == 1:
        init_filename0 = init_filename0 + 'vel_'
    if args.meanHMM:
        init_filename0 = init_filename0 + 'meanHMM_'
    if args.attack_defend:
        init_filename0 = init_filename0 + '_roles_' + str(args.n_roles) +'_'
    if args.in_sma:
        init_filename0 = init_filename0 + 'inSimple_'
    elif args.in_out:
        init_filename0 = init_filename0 + 'inout_'
    init_filename0 = init_filename0 + 'acc_' + str(args.acc) + '_'          
    init_filename0 = init_filename0 + 'norm/' if args.normalize else init_filename0 + 'unnorm/' 
    if args.attention == 3:
        init_filename00 = init_filename0 + args.data + '_att3/'
    else:
        init_filename00 = init_filename0 + args.data + '/'
    
    if args.attack_defend:
        args.wo_macro = True

    init_filename0 = init_filename0 + args.model + '_' + args.data + '/'
    init_filename0 = init_filename0 + 'att_' + str(args.attention) + '_' + str(batchSize) + '_' + str(totalTimeSteps)      
    if args.wo_macro and 'MACRO' in args.model:
        init_filename0 = init_filename0 + '_wo_macro' 
    if args.drop_ind:
        init_filename0 = init_filename0 + '_drop_ind' 
    init_filename000 = init_filename0
    if args.body:
        init_filename0 = init_filename0 + '_body' 
    if args.jrk > 0:
        init_filename0 = init_filename0 + '_jrk' 
    if args.lam_acc > 0:
        init_filename0 = init_filename0 + '_lacc' 
    if args.finetune:
        init_filename0 = init_filename0 + '_finetune' 
    if args.res:
        init_filename0 = init_filename0 + '_res' 
    #if args.wo_cross:
    #    init_filename0 = init_filename0 + '_wo_cross' 
    #if args.hard_only and args.attention == 3:
    #    init_filename0 = init_filename0 + '_hard_only' 

    if not os.path.isdir(init_filename0):
        os.makedirs(init_filename0)
    init_pthname = '{}_state_dict'.format(init_filename0)
    init_pthname0 = '{}_state_dict'.format(init_filename00)
    print('model: '+init_filename0)

    if not os.path.isdir(init_pthname):
        os.makedirs(init_pthname)
    if not os.path.isdir(init_pthname0):
        os.makedirs(init_pthname0)

    if (args.n_GorS==7500 and args.data == 'soccer'):
        batchSize = int(batchSize/2)
    # elif (args.model=='GVRNN'):
    #    batchSize = int(batchSize/4)
    
    # args.hard_only = True
    args.dataset = args.data
    args.n_feat = n_feat
    args.fs = fs
    args.game_files = game_files  
    args.game_files_val = game_files_val
    args.game_files_te = game_files_te
    args.start_lr = 1e-3 
    args.min_lr = 1e-3 
    clip = True # gradient clipping
    args.seed = 200
    save_every = 10
    args.batch_size = batchSize
    # args.normalize = normalize # default: False
    # args.cont = False # continue training previous best model
    args.x_dim = outputlen0 # output
    args.y_dim = featurelen # input
    args.m_dim = 90 if args.data == 'nba' else 34*22#26*17*4#34*22*4
    args.n_all_agents = 22 if args.data != 'nba' else 10 
    if args.model =='GVRNN':
        args.z_dim = (args.n_all_agents+1)*4
        args.rnn_dim = 64
    else:
        args.z_dim = 64 
        args.rnn_dim = 100 # 100
    args.h_dim = 64 #128 
    args.n_layers = 2
    args.rnn_micro_dim = args.rnn_dim
    args.rnn_macro_dim = 100
    args.burn_in = 19 # int(totalTimeSteps/3)
    args.horizon = totalTimeSteps+4
    args.n_agents = len(activeRole)
    
    if not torch.cuda.is_available():
        args.cuda = False
        print('cuda is not used')
    else:
        args.cuda = True
        print('cuda is used')
    ball_dim = 4 if acc >= 0 else 2
    '''if args.data == 'nba':
        ball_dim = 7 if acc else 5 
    elif args.data == 'soccer':    
        ball_dim = 6 if acc else 4'''
    # Parameters to save
    pretrain2_time = args.pretrain2 if args.body else 0
    args.pretrain2 = pretrain2_time
    temperature = 1 if args.data == 'soccer' else 1 
        
    params = {
        'model' : args.model,
        'attention' : args.attention,
        'wo_macro' : args.wo_macro,
        'wo_cross' : args.wo_cross,
        'res' : args.res,
        'dataset' : args.dataset,
        'x_dim' : args.x_dim,
        'y_dim' : args.y_dim,
        'z_dim' : args.z_dim,
        'h_dim' : args.h_dim,
        'm_dim' : args.m_dim,
        'rnn_dim' : args.rnn_dim,
        'rnn_att_dim' : 32,
        'n_layers' : args.n_layers, 
        'len_seq' : totalTimeSteps,    
        'generative' : False,  
        'n_agents' : args.n_agents,    
        'min_lr' : args.min_lr,
        'start_lr' : args.start_lr,
        'normalize' : args.normalize,
        'in_out' : args.in_out,
        'in_sma' : args.in_sma,
        'seed' : args.seed,
        'cuda' : args.cuda,
        'n_feat' : n_feat,
        'fs' : fs,
        'embed_size' : 32, # 8
        'embed_ball_size' : 32, # 8
        'burn_in' : args.burn_in,
        'horizon' : args.horizon,
        'rnn_micro_dim' : args.rnn_micro_dim,
        'rnn_macro_dim' : args.rnn_macro_dim,
        'acc' : acc,
        'body' : args.body,
        'hard_only' : args.hard_only,
        'jrk' : args.jrk,
        'lam_acc' : args.lam_acc,
        'ball_dim' : ball_dim,
        'n_all_agents' : args.n_all_agents,
        'temperature' : temperature,
        'drop_ind' : args.drop_ind,
        'pretrain2' : args.pretrain2,
        'init_pthname0' : init_pthname0
    }
        
    #'pretrain' : args.pretrain,
        
    # Set manual seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    ####### Sanity check ##################
    if args.Sanity:
        losses = run_sanity(args,game_files+'_te_')

    # Load model
    
    model = load_model(args.model, params, parser)

    if args.cuda:
        model.cuda()
    # Update params with model parameters
    params = model.params
    params['total_params'] = num_trainable_params(model)

    # Create save path and saving parameters
    pickle.dump(params, open(init_filename0+'/params.p', 'wb'), protocol=2)

    # Continue a previous experiment, or start a new one
    if args.cont:
        print('args.cont = True')
        if 'MACRO' in args.model and args.pretrain > 0:
            if os.path.exists('{}_best_pretrain.pth'.format(init_pthname0)): 
                state_dict = torch.load('{}_best_pretrain.pth'.format(init_pthname0))
                model.load_state_dict(state_dict)      
                print('best pretrain model was loaded')   
            else:
                print('args.cont = True but file did not exist')

        elif args.pretrain2 > 0:
            if os.path.exists('{}_best_pretrain2.pth'.format(init_pthname0)): 
                state_dict = torch.load('{}_best_pretrain2.pth'.format(init_pthname0))
                model.load_state_dict(state_dict)      
                print('best pretrain body model was loaded')   
            else:
                print('args.cont = True but file did not exist')
        else:
            if os.path.exists('{}_best.pth'.format(init_pthname)): 
                # state_dict = torch.load('{}_12.pth'.format(init_pthname))
                state_dict = torch.load('{}_best.pth'.format(init_pthname))
                model.load_state_dict(state_dict)
                print('best model was loaded')
            else:
                print('args.cont = True but file did not exist')
    else:
        print('args.cont = False')
        if 'MACRO' in args.model and not args.wo_macro and args.pretrain == 0:
            # https://discuss.pytorch.org/t/how-to-transfer-learned-weight-in-the-same-model-without-last-layer/32824
            pretrained_dict = torch.load('{}_best_pretrain.pth'.format(init_pthname0))
            model_dict = model.state_dict()
            pretrained_list = list(pretrained_dict.items())
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_list[:20] if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            model.load_state_dict(model_dict)
            print('pretrained model was loaded')
        if args.finetune: # args.pretrain2 == 0 and args.body:
            # this did not work well
            pretrained_dict = torch.load('{}_state_dict_best.pth'.format(init_filename000)) # _pretrain2
            model_dict = model.state_dict()
            pretrained_list = list(pretrained_dict.items())
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
            lt=14 if args.wo_macro else 15 # 14-16: decoder
            lt2=17 if args.wo_macro else 18 # 17: microRNN
            cntr=0
            for child in model.children():
                cntr+=1
                if cntr < lt or cntr > lt2:
                    #print(str(cntr))
                    # print(child)
                    for param in child.parameters():
                        param.requires_grad = False
            print('pretrained model2 was loaded')

    print('############################################################')

    # Dataset loaders
    num_workers = int(args.numProcess/2)
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if args.cuda else {}
    kwargs2 = {'num_workers': num_workers, 'pin_memory': True} if args.cuda else {}
    print('num_workers:'+str(num_workers))
    batchSize_val = len_seqs_val if len_seqs_val <= batchSize else batchSize
    batchSize_test = len_seqs_test if len_seqs_test <= int(batchSize/2) else batchSize # int(/4)
    if (args.n_GorS==7500 and args.dataset == 'soccer'):
        batchSize_val = int(batchSize/4*3)
        batchSize_test = 128
        if 'MACRO' in args.model and (not args.wo_macro or args.attention == 3):
            batchSize_test = 80
            batchSize_val = 80
        elif 'MACRO' in args.model and (not args.wo_macro and args.attention == 3):
            batchSize_test = 64
            batchSize_val = 64
        if 'MACRO' in args.model and not args.wo_macro:
            batchSize = int(batchSize/4*3)
        if args.attention == 3:
            batchSize_val = 80 # int(batchSize_val/4*3)
    elif (args.n_GorS>=50 and args.dataset == 'nba'):
        batchSize_test = int(batchSize/8)
        #if args.attention == 3:
        #    batchSize_test = int(batchSize_test/4*3)
    if (args.model=='GVRNN'):
        batchSize_val = int(batchSize/2)
        batchSize_test = int(batchSize/2)

    if not TEST:    
        train_loader = DataLoader(
            GeneralDataset(args, len_seqs_tr, train=1, normalize_data=args.normalize),
            batch_size=batchSize, shuffle=False, **kwargs)    
        val_loader = DataLoader(
            GeneralDataset(args, len_seqs_val, train=0, normalize_data=args.normalize),
            batch_size=batchSize_val, shuffle=False, **kwargs2)
    test_loader = DataLoader(
        GeneralDataset(args, len_seqs_test, train=-1, normalize_data=args.normalize),
        batch_size=batchSize_test, shuffle=False, **kwargs2)
    print('batch train: '+str(batchSize)+' val:'+str(batchSize_val)+' test: '+str(batchSize_test))
    ###### TRAIN LOOP ##############
    best_val_loss = 0
    epochs_since_best = 0
    lr = max(args.start_lr, args.min_lr) # if not args.finetune else 1e-4
    epoch_first_best = -1
    #print('epoch_first_best: '+str(epoch_first_best))

    pretrain_time = args.pretrain if 'MACRO' in args.model else 0
    
    L_att = False
    # hyperparams = {'model': args.model,'acc': acc,'burn_in': args.burn_in}
    hyperparams = {'model': args.model,'acc': acc,'burn_in': args.horizon,'L_att':L_att,
            'pretrain':(0 < pretrain_time),'pretrain2':(0 < pretrain2_time)}
    
    if not TEST:
        for e in range(args.n_epoch):
            epoch = e+1
            print('epoch '+str(epoch))
            pretrain = (epoch <= pretrain_time)
            pretrain2 = (epoch <= pretrain2_time)
            hyperparams['pretrain'] = pretrain
            hyperparams['pretrain2'] = pretrain2

            # Set a custom learning rate schedule
            if epochs_since_best == 5: # and lr > args.min_lr:
                # Load previous best model
                filename = '{}_best.pth'.format(init_pthname)
                if epoch <= pretrain_time:
                    filename = '{}_best_pretrain.pth'.format(init_pthname0)
                elif epoch <= pretrain_time+pretrain2_time:
                    filename = '{}_best_pretrain2.pth'.format(init_pthname)

                state_dict = torch.load(filename)

                # Decrease learning rate
                # lr = max(lr/3, args.min_lr)
                # print('########## lr {} ##########'.format(lr))
                epochs_since_best = 0
            else:
                if not hyperparams['pretrain'] and not args.finetune:
                    # lr = lr*0.99 # 9
                    print('########## lr {:.4e} ##########'.format(lr)) 
                    epochs_since_best += 1
                

            # Remove parameters with requires_grad=False (https://github.com/pytorch/pytorch/issues/679)
            if not args.finetune:
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr)
            else:
                optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=5e-4, momentum=0.9)
            start_time = time.time()
            
            print('pretrain:'+str(hyperparams['pretrain'])+' pretrain2:'+str(hyperparams['pretrain2'])+' L_att:'+str(L_att))
            hyperparams['burn_in'] = args.horizon
            hyperparams['L_att'] = L_att
            # hyperparams = {'model': args.model,'acc': acc,'burn_in': args.horizon,'L_att':L_att}
            train_loss,train_loss2 = run_epoch(train=1, rollout=False, hp=hyperparams)
            print('Train:\t'+loss_str(train_loss)+'|'+loss_str(train_loss2))
            
            if not hyperparams['pretrain'] : #epoch % 5 == 3:
                hyperparams['burn_in'] = args.burn_in
                # hyperparams = {'model': args.model,'acc': acc,'burn_in': args.burn_in,'L_att':L_att}
                val_loss,val_loss2 = run_epoch(train=0, rollout=True, hp=hyperparams)
                print('RO Val:\t'+loss_str(val_loss)+'|'+loss_str(val_loss2))
                
            else:
                hyperparams['burn_in'] = args.horizon
                val_loss,val_loss2 = run_epoch(train=0, rollout=False, hp=hyperparams)
                print('Val:\t'+loss_str(val_loss)+'|'+loss_str(val_loss2))

            total_val_loss = sum(val_loss.values())

            epoch_time = time.time() - start_time
            print('Time:\t {:.3f}'.format(epoch_time))

            # Best model on test set
            if e > epoch_first_best and (best_val_loss == 0 or total_val_loss < best_val_loss): 
                best_val_loss_prev = best_val_loss
                best_val_loss = total_val_loss
                epochs_since_best = 0

                filename = '{}_best.pth'.format(init_pthname)
                if epoch <= pretrain_time:
                    filename = '{}_best_pretrain.pth'.format(init_pthname0)
                elif epoch <= pretrain_time+pretrain2_time:
                    filename = '{}_best_pretrain2.pth'.format(init_pthname)

                torch.save(model.state_dict(), filename)
                print('##### Best model #####')
                if epoch > pretrain_time and (best_val_loss_prev-best_val_loss)/best_val_loss < 0.0001 and best_val_loss_prev != 0:
                    print('best loss - current loss: ' +str(best_val_loss_prev)+' - '+str(best_val_loss))
                    break 


            # Periodically save model
            if epoch % save_every == 0:
                filename = '{}_{}.pth'.format(init_pthname, epoch)
                torch.save(model.state_dict(), filename)
                print('########## Saved model ##########')

            # End of pretrain stage
            if epoch == pretrain_time:
                print('########## END pretrain ##########')
                best_val_loss = 0
                epochs_since_best = 0
                lr = max(args.start_lr, args.min_lr)

                state_dict = torch.load('{}_best_pretrain.pth'.format(init_pthname0))
                model.load_state_dict(state_dict)

            elif epoch == pretrain_time+pretrain2_time:
                print('########## END pretrain2 ##########')
                best_val_loss = 0
                epochs_since_best = 0
                lr = max(args.start_lr, args.min_lr)

                state_dict = torch.load('{}_best_pretrain2.pth'.format(init_pthname))
                model.load_state_dict(state_dict) 
                pretrain2_model = model
                pretrained2_list = list(state_dict.items())
                
                params['pretrain2'] = False
                model = load_model(args.model, params, parser)
                if args.cuda:
                    model.cuda()
                model_dict = model.state_dict()
                
                pretrained2_dict = {k: v for k, v in pretrained2_list if k in model_dict}
                model_dict.update(pretrained_dict) 
                model.load_state_dict(model_dict)
                print('pretrained2 model was loaded')
                           
        print('Best Val Loss: {:.4f}'.format(best_val_loss))
    
    # Load params
    '''if args.data =='jleague': # temporary: use soccer model
        init_filename0000 = '../VRNN_Jleague_data/weights/sub10_filt_vel__roles_3_inSimple_acc_0_unnorm/MACRO_VRNN_soccer/att_-1_256_80_wo_macro'
        params = pickle.load(open(init_filename0000+'/params.p', 'rb'))
        init_pthname = '../VRNN_Jleague_data/weights/sub10_filt_vel__roles_3_inSimple_acc_0_unnorm/MACRO_VRNN_soccer/att_-1_256_80_wo_macro_state_dict'
        state_dict = torch.load('{}_best.pth'.format(init_pthname, params['model']), map_location=lambda storage, loc: storage)
    else: '''
    params = pickle.load(open(init_filename0+'/params.p', 'rb'))
    # Load model
    state_dict = torch.load('{}_best.pth'.format(init_pthname, params['model']), map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    # Load ground-truth states from test set
    loader = test_loader 
    n_sample = 10
    n_smp_b = 10 if args.dataset == 'nba' else 1
    if args.n_GorS>=50 and args.dataset == 'nba' and args.attention == 3:
        n_smp_b = 5
    rep_smp = int(n_sample/n_smp_b)

    if True:
        # GT 
        print('GT sample')
        # Sample trajectory
        samples_GT = np.zeros((args.horizon+1,args.n_agents,len_seqs_test,
            featurelen)) # [for t in range(n_sample)]
        ind_players = np.zeros((len_seqs_test,n_pl*2+3))
        for batch_idx, (data, macro_intents, ind_player) in enumerate(loader):
            # (batch, agents, time, feat) => (time, agents, batch, feat) 
            data = data.permute(2, 1, 0, 3)
            try: 
                for i in range(n_sample):
                    samples_GT[:,:,batch_idx*batchSize_test:(batch_idx+1)*batchSize_test] = data[:args.horizon+1,:,:,:].detach().cpu().numpy() 
                ind_players[batch_idx*batchSize_test:(batch_idx+1)*batchSize_test,:] = ind_player  
            except: import pdb; pdb.set_trace()
        if args.normalize:
            for i in range(n_sample):
                samples_GT[:,:,:,0::2] *= 52.5
                samples_GT[:,:,:,1::2] *= 34
        missing_indices = np.zeros((len_seqs_test))
        for n in range(len_seqs_test):
            # if np.sum(samples_GT[:,0,n]>9998)>0:
            missing_indices[n] = np.sum(samples_GT[:,0,n,0]<9999)
            samples_GT[int(missing_indices[n]):,:,n,:] = np.nan

        print('test sample')
        # Sample trajectory
        samples = [np.zeros((args.horizon+1,args.n_agents,len_seqs_test,
            featurelen)) for t in range(n_sample)]
        hard_att = 0
        
        loss_i = [{} for t in range(n_sample)]
        losses = {}
        losses2 = {}
        losses_t = np.zeros((args.horizon+1,2,n_sample, args.n_agents,len_seqs_test))


        for r in range(rep_smp):
            start_time = time.time()
            if r > 0:
                state_dict = torch.load('{}_best.pth'.format(init_pthname, params['model']), map_location=lambda storage, loc: storage)
                model.load_state_dict(state_dict)

            for batch_idx, (data, macro_intents, index_) in enumerate(loader):
                if args.cuda:
                    data = data.cuda() #, data_y.cuda()
                    # (batch, agents, time, feat) => (time, agents, batch, feat) 
                data = data.permute(2, 1, 0, 3)

                sample, _, _, output, output2 = model.sample(data, rollout=True, burn_in=args.burn_in, L_att=L_att, CF_pred=False, n_sample=n_smp_b, TEST = True)

                for i in range(n_smp_b):
                    # sample0 = sample.detach().cpu().numpy() if n_smp_b == 1 else sample[i].detach().cpu().numpy()   
                    try: 
                        sample0 = sample[i].detach().cpu().numpy()  
                        samples[r*n_smp_b+i][:,:,batch_idx*batchSize_test:(batch_idx+1)*batchSize_test] = sample0 # [:-3]
                    except: import pdb; pdb.set_trace()

                for key in output:
                    if batch_idx == 0 and r == 0:
                        losses[key] = np.zeros(n_sample)
                        losses2[key] = np.zeros((n_sample, len_seqs_test))
                    try:
                        losses[key][r*n_smp_b:(r+1)*n_smp_b] += np.sum(output[key].detach().cpu().numpy(),axis=1)
                        losses2[key][r*n_smp_b:(r+1)*n_smp_b, batch_idx*batchSize_test:(batch_idx+1)*batchSize_test] = output[key].detach().cpu().numpy()
                    except: import pdb; pdb.set_trace()

                for key in output2:
                    if batch_idx == 0 and r == 0:
                        losses[key] = np.zeros(n_sample)
                        losses2[key] = np.zeros((n_sample, len_seqs_test))
                    try:
                        losses[key][r*n_smp_b:(r+1)*n_smp_b] += np.sum(output2[key].detach().cpu().numpy(),axis=1)
                        losses2[key][r*n_smp_b:(r+1)*n_smp_b, batch_idx*batchSize_test:(batch_idx+1)*batchSize_test] = output2[key].detach().cpu().numpy()
                    except: import pdb; pdb.set_trace()

            # temporal error
            for i in range(args.n_agents):
                try:
                    # samples_GT[:,0,0,n_feat*i:n_feat*i+2] - samples[r][:,0,0,n_feat*i:n_feat*i+2] 
                    tmp_data = samples_GT[:,:,:,n_feat*i:n_feat*i+n_feat] - samples[r][:,:,:,n_feat*i:n_feat*i+n_feat] 
                    losses_t[:,0,r,i,:] = np.sqrt(np.nansum(tmp_data[:,i,:,0:2]**2,2))
                    losses_t[:,1,r,i,:] = np.sqrt(np.nansum(tmp_data[:,i,:,2:4]**2,2))
                except: import pdb; pdb.set_trace()

            for i in range(n_smp_b):
                for key in losses:
                    #non_nan = np.sum(samples_GT[1:,0,:,0]<9999)
                    #if non_nan < len(test_loader.dataset)*args.horizon: 
                    #    loss_i[r*n_smp_b+i][key] = losses[key][r*n_smp_b+i] * args.horizon / non_nan
                    #else:
                    loss_i[r*n_smp_b+i][key] = losses[key][r*n_smp_b+i] / len(test_loader.dataset)
                print('Test sample '+str(r*n_smp_b+i)+':\t'+loss_str(loss_i[r*n_smp_b+i]))

            epoch_time = time.time() - start_time
            print('Time:\t {:.3f}'.format(epoch_time)) # Sample {} r*n_smp_b,

        
        if True: # create Mean + SD Tex Table for positions------------------------------------------------
            avgL2_m = {}
            avgL2_sd = {}
            bestL2_m = {}
            bestL2_sd = {}
            for key in losses2:
                mean = np.mean(losses2[key],0)
                avgL2_m[key] =  np.mean(mean)
                avgL2_sd[key] = np.std(mean)
                best = np.min(losses2[key],0)
                bestL2_m[key] =  np.mean(best)
                bestL2_sd[key] = np.std(best)    

            print(args.model+'att'+str(args.attention)+' body:'+str(args.body))
            print('(mean):'
                +' $' + '{:.2f}'.format(avgL2_m['e_pos'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_pos'])+'$ &'
                +' $' + '{:.2f}'.format(avgL2_m['e_vel'])+' \pm '+'{:.2f}'.format(avgL2_sd['e_vel'])+'$ &'
                ) 
            print('(best):'
                +' $' + '{:.2f}'.format(bestL2_m['e_pos'])+' \pm '+'{:.2f}'.format(bestL2_sd['e_pos'])+'$ &'
                +' $' + '{:.2f}'.format(bestL2_m['e_vel'])+' \pm '+'{:.2f}'.format(bestL2_sd['e_vel'])+'$ &'
                ) 
        # Save samples
        experiment_path = '{}/experiments/sample'.format(init_filename0)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)

        if args.normalize:
            for i in range(n_sample):
                samples[:,:,:,0::2] *= 52.5
                samples[:,:,:,1::2] *= 34
        #if not 'MACRO' in args.model: 
        hard_att = []
        macros = []
        #else:
        #    macros = []
        # save
        # samples: [10 samples][time, predicted players (A1,D1,D2), possible combinations, data dim (92-dim)]
        # samples_GT: [10 samples][time, predicted players, possible combinations, data dim (92-dim)]
        # ind_players: [samples, predicted players, 25-dim indices]. 0 in 25-dim: game id; 1 in 25-dim: seq id;  2-24: player id.
        # losses_t: [horizon, (pos,vel),n_sample, n_agents,len_seqs_test]
        #if 'MACRO' in args.model: 
        try: pickle.dump([samples, samples_GT, hard_att, ind_players, losses2, macros,losses_t], open(experiment_path+'/samples.p', 'wb'), protocol=4)
        except: import pdb; pdb.set_trace()
        
        #else:
        #    pickle.dump([samples, samples_GT, ind_players, losses2,losses_t], open(experiment_path+'/samples.p', 'wb'), protocol=4)
        
        loss_t = np.nanmean(losses_t,axis=(2,3,4))
        print('computation is finished!')
        import pdb; pdb.set_trace()
# features.py
import glob, os, sys, math, warnings, copy, time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import pandas as pd
from scipy.stats import multivariate_normal
from hmmlearn import hmm

import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

# modifying the code https://github.com/samshipengs/Coordinated-Multi-Agent-Imitation-Learning
 
# ===================================================================
class HiddenStructureLearning:
    def __init__(self, events_df, Model_d, Model_o, n_pl, n_roles, args, libmode, tol=1e-1, defend_iter=100, offend_iter=100):
        self.df = events_df.copy()
        self.libmode = libmode
        self.tol = tol
        self.defend_iter = defend_iter
        self.offend_iter = offend_iter
        self.Model_d = Model_d
        self.Model_o = Model_o
        self.n_pl = n_pl
        self.n_roles = n_pl if n_pl == 5 else n_pl - 1 
        self.acc = args.acc # added
        self.velocity = args.velocity
        self.meanHMM = args.meanHMM # added

        if n_pl == 5: # for basketball, use all players
            self.defend_players = list(range(n_pl))
            self.offend_players = list(range(n_pl, n_pl*2))
        elif n_pl == 11: # for soccer, eliminate goalkeeper
            self.defend_players = list(range(n_pl-1))
            self.offend_players = list(range(n_pl, n_pl*2-1))

    # =================================
    # find_features_ind ===============
    # =================================
    def find_features_ind(self, player):
        # extract feature index regarding the player
        n_pl = self.n_pl
        # assert player < n_pl*2
        pxy_ind = [player*2, player*2+1] # 0-1

        if n_pl == 5:
            n_f0 = 26 # number of raw features
        elif n_pl == 11:
            n_f0 = 46

        n_f1 = n_f0 + n_pl*2*4 # index of relation with ball 66/134 
        n_f2 = n_f1 + n_pl*2*4 # index of relation with goal 106/222
        n_f3 = n_f2 + n_pl*2*4*n_pl*2 # index of relation with all players 506/2158
        if n_pl == 5:
            n_f4 = n_f3 + n_pl*2*2 + 3
        elif n_pl == 11:
            n_f4 = n_f3 + n_pl*2*2 + 2 # 2204

        polar_bball_ind = [n_f0+player, n_f0+player+n_pl*2, n_f0+player+n_pl*4, n_f0+player+n_pl*6] # 2-5
        polar_hoop_ind = [n_f1+player, n_f1+player+n_pl*2, n_f1+player+n_pl*4, n_f1+player+n_pl*6]  # 6-9
        players_ind = list(range(n_f2 + player*n_pl*8, n_f2 + (player+1)*n_pl*8)) # 12-51/12-99
        pvxy_ind = [n_f3+player*2, n_f3+player*2+1] # 10-11 
        paxy_ind = [n_f4+player*2, n_f4+player*2+1] # 12-13
        
        if self.velocity < 2: # not self.acc == -1:
            player_features_ind = pxy_ind + polar_bball_ind + polar_hoop_ind + pvxy_ind + players_ind 
        else:
            player_features_ind  = pxy_ind + polar_bball_ind + polar_hoop_ind + pvxy_ind + paxy_ind + players_ind 
        
        player_features_ind2 = player_features_ind # all

        # posHMM:
        player_features_ind = pxy_ind + polar_bball_ind[:2] + polar_hoop_ind[:2] 

        features_ind = np.array(player_features_ind)
        return player_features_ind, features_ind, player_features_ind2

    def find_features_ind2(self, player, reorderedPlayers):
        n_pl = self.n_pl
        nlpd = np.array(n_pl, dtype=int)
        _, _, player_features_ind = self.find_features_ind(player)
        features_ind = np.array(player_features_ind)

        # relationship between players
        n_f0 = 12 if self.velocity < 2 else 14 # not self.acc else 14
        V = 4 # dist, cos, sin, theta
        player_features_ind0 = player_features_ind.copy()
        # for i in range(nlpd): # defense players
        for i, p in enumerate(reorderedPlayers): # i: assigned player, p: order(1:5)
            for v in range(V): # variables
                player_features_ind[n_f0+v+p*V] = player_features_ind0[n_f0+v+i*V]
                # player_features_ind[n_f0+i+v*nlpd] = player_features_ind[players_ind[reorderedPlayers[i]]]
        features_ind = np.array(player_features_ind)
        return player_features_ind, features_ind  

    # =================================
    # create_hmm_input ================
    # =================================
    def create_hmm_input(self, player_inds):
        event = self.df.moments.values
        # create X: array-like, shape (n_samples, n_features)
        player_fts = [ms[:, self.find_features_ind(player)[1]] for player in player_inds for ms in event]            
        
        if self.libmode == 'pom':
            return player_fts

        X = np.concatenate(player_fts, axis=0)            
        # create lengths : array-like of integers, shape (n_sequences, )
        lengths = [len(ms) for player in player_inds for ms in event]

        assert len(event[0]) == lengths[0]
        assert len(event[len(event)//2]) == lengths[len(lengths)//len(player_inds)//2]
        assert len(event[-1]) == lengths[-1]
        return X, lengths
    
    def train_hmm(self, player_inds, n_iter, random_state=42, verbose=True):
        print('Training for {0} players and {1} player_roles with iterations: {2}'.format(len(player_inds), self.n_roles, n_iter))
        n_pl = self.n_pl
        if n_pl == 5:
            assert len(player_inds) == n_pl # defend and offend players each are five
        elif n_pl == 11:
            assert len(player_inds) == n_pl-1
        
        X, lengths = self.create_hmm_input(player_inds=player_inds)
        if True: 
            model = hmm.GaussianHMM(n_components=self.n_roles, 
                                    covariance_type='diag', 
                                    algorithm='map',
                                    n_iter=n_iter, 
                                    tol=self.tol,
                                    random_state=random_state,
                                    verbose=verbose)
            model.fit(X, lengths)
        else: # check
            if n_pl == 11:
                game_files_pre = './data/all_soccer_games_7500_unnorm_filt_acc_k0/_pre_'
            else:
                game_files_pre = './data/all_nba_games_100_unnorm_filt_acc_k0/_pre_'

            with open(game_files_pre+'.pkl', 'rb') as f:
                model = np.load(f,allow_pickle=True)[self.team+2]
                print('soccer HMM model of team ' + str(self.team) + ' was loaded')

        cmeans = model.means_
        # covars = model.covars_[:,:2,:2] 
        self.visualize_HMM(model)
        return {'X': X,
                'lengths': lengths,
                'model': model, # added
                # 'state_sequence': state_sequence.reshape(5, -1),  # the shape here can be done because the original input is ordered by players chunk
                # 'state_sequence_prob': [state_sequence_prob[i:i+n_samples//5] for i in range(0, n_samples, n_samples//5)], 
                'cmeans': cmeans}


    def visualize_HMM(self,model):
        if (self.defend_iter > 10 and self.team==0) or (self.offend_iter > 10 and self.team==1):
            fig = plt.figure(figsize=(24, 12)) 
            self.draw_HMM(model)     
            if self.n_pl == 5:
                data = 'NBA'
            else:
                data = 'soccer'
            
            XO = 'DF' if self.team==0 else 'OF'
            if not os.path.isdir('figure/HMM/'):
                os.makedirs('figure/HMM/')
            try: plt.savefig("figure/HMM/"+data+"_Gaussian_HMM_"+XO+".png", bbox_inches='tight')
            except: import pdb; pdb.set_trace()
            plt.close()
            print(XO+' Gaussian_HMM was visualized')
        
    def draw_HMM(self,model):
        cmeans = model.means_
        covars = model.covars_   
        # team_A: defense  team_B: attack
        K,D = cmeans.shape
        cmeans_mat = cmeans[:,:2]
        covars_mat = covars[:,:2,:2]
        
        #plt.axis('equal')
        self.plotCourt(K)
        # timestep = pred_len

        # initial marker 
        self.plotDistribution(cmeans_mat,covars_mat,K)

    def plotDistribution(self,cmeans_mat,covars_mat,K):

        ax = plt.gca() 
        clr = 'b' if self.team == 0 else 'r'
        for j in range(K):
            mx = cmeans_mat[j, 0]
            my = cmeans_mat[j, 1]    
            cx = covars_mat[j, 0, 0]
            cy = covars_mat[j, 1, 1]        
            e = patches.Ellipse(xy=(mx, my), width=cx, height=cy,fill=False,ec=clr)
            ax.add_patch(e)

            # player jersey # (text)
            ax.text(mx,my,str(j+1),color='k',ha='center',va='center')
        ax.set_title('Gaussian HMM')

    def predict_hmm(self, trainModel, player_inds, n_iter, random_state=42, verbose=True):    
        X, lengths = self.create_hmm_input(player_inds=player_inds)
        # Z = trainModel.predict(X, lengths) # unnecessary
        cmeans = trainModel.means_
        return {'X': X,
                'lengths': lengths,
                'model': trainModel,  
                'cmeans': cmeans}

    def assign_roles(self, trainModel, player_inds, n_iter, mode='euclidean'): 
        n_pl = self.n_pl
        n_roles = self.n_roles 

        if not trainModel:
            result = self.train_hmm(player_inds=player_inds, n_iter=n_iter) # train
        else: 
            result = self.predict_hmm(trainModel, player_inds=player_inds, n_iter=n_iter)

        lengths = result['lengths']
        n_seq = len(lengths)

        if mode == 'euclidean':
            ed = distance.cdist(result['X'], result['cmeans'], 'euclidean') 
        elif mode == 'cosine':
            ed = distance.cdist(result['X'], result['cmeans'], 'cosine') # (seqs*players)*roles  

        if self.meanHMM:
            # ed2 = np.zeros(n_seq,n_roles)
            start = 0
            for i in range(n_seq): 
                ed[start:start+lengths[i],:] = np.mean(ed[start:start+lengths[i],:],axis=0)
                start += lengths[i]

        if n_pl == 5:
            assert len(player_inds) == n_pl # defend and offend players each are five
        elif n_pl == 11:
            assert len(player_inds) == n_pl-1

        n = len(ed)//len(player_inds) # number of sequences for each players 
        assert len(ed) % len(player_inds) == 0 # it should be divisibe by number of players
        # n = len(ed)

        # unnecessary to be corrected when n_pl != n_roles 
        role_assignments = np.zeros((n,len(player_inds)), dtype=np.int)
        for i in range(n):
            cost = ed[np.arange(len(player_inds))*n + i] # n_roles*n_roles  row i is assigned to column j.
            cost = cost.transpose() # column j is assigned to row i.        
            try: role_assignments[i,:] = np.array(self.assign_ind(cost))
            except: import pdb; pdb.set_trace()
        # role_assignments = np.array([self.assign_ind(ed[np.arange(len(player_inds))*n + i]) for i in range(n)])
        return role_assignments, result # role_assignments: (seqs,players), result['X']: (seqs*players)*features

    def assign_ind(self, cost):
        # cost: n_players*n_roles matrix
        _, col_ind = linear_sum_assignment(cost)
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
        return col_ind
    
    def reorder_moment(self):
        original = copy.deepcopy(self.df.moments.values)  
        if not self.Model_d: # train
            self.Model_d = []
            self.Model_o = []
            self.team = 0
        if self.offend_iter > 0:     
            t1 = time.time()

            defend_role_assignments, defend_result = self.assign_roles(self.Model_d,player_inds=self.defend_players, n_iter=self.defend_iter)
            self.team = 1
            offend_role_assignments, offend_result = self.assign_roles(self.Model_o,player_inds=self.offend_players, n_iter=self.offend_iter) 
                    
            if not self.Model_d: # train
                print('Total HMM training took {0:.2f}mins'.format((time.time()-t1)/60))
 
            
            reordered = copy.deepcopy(self.df.moments.values)
            # offset is to map the reordered index back to original range for offense players
            def reorder_moment_(players, original, reordered, role_assignments, offset):
                divider = 0
                lengths = [len(m) for m in original]
                # iteratve through each moments length
                for i in range(len(lengths)):
                    # grab the corresponding moments' reordered roles
                    ra_i = role_assignments[divider:divider+lengths[i]]
                    # update the next starting index
                    divider += lengths[i]
                    # iterate through each moment in the current moments
                    for j in range(lengths[i]):
                        # iterate through each players
                        for k, p in enumerate(players):
                            # get the current player feature index
                            p_ind = self.find_features_ind(p)[2] # [0]
                            # get the player feature index corresponding to the reordered role
                            try: 
                                re_p_ind = self.find_features_ind(ra_i[j][k]+offset)[0]
                                if ra_i.shape[1] == self.n_roles: # n_pl==N_roles(Le+17)
                                    re_p_ind = self.find_features_ind2(ra_i[j][k]+offset,ra_i[j])[0]
                            
                                reordered[i][j][p_ind] = original[i][j][re_p_ind]
                            except: import pdb; pdb.set_trace()
                return reordered
            reordered_defend = copy.deepcopy(reorder_moment_(self.defend_players, original, reordered, defend_role_assignments, 0))
            reordered_all = copy.deepcopy(reorder_moment_(self.offend_players, original, reordered_defend, offend_role_assignments, self.n_pl))
            self.visualize(original,reordered_all)
            return reordered_all, defend_result['model'], offend_result['model']
        else: 
            defend_result = [] ; offend_result = [] 
            reordered_all = original
            return reordered_all, defend_result, offend_result
        

    def visualize(self,original,reordered_all):
        N1 = len(original)
        if self.n_pl == 5: 
            data = 'NBA'
        else:
            data = 'soccer'
        allPlayers = True

        for n1 in range(0,5,1): 
            fig = plt.figure(figsize=(24, 12)) # ,constrained_layout=True

            #############################
            if self.n_pl == 5:
                f_ax = fig.add_subplot(1,2,1) 
            else:
                f_ax = fig.add_subplot(2,1,1) 
            self.draw_trajectory3(original[n1],f_ax, allPlayers)     
            f_ax.set_title('original')

            #############################       
            if self.n_pl == 5:
                f_ax = fig.add_subplot(1,2,2) 
            else:
                f_ax = fig.add_subplot(2,1,2) 
            self.draw_trajectory3(reordered_all[n1],f_ax, allPlayers)  
            f_ax.set_title('reordered_all')


            plt.savefig("figure/HMM/{}_{}.png".format(data,n1), bbox_inches='tight')
            plt.close()
            print('test sample '+str(n1)+' was visualized')
            # draw_trajectory(samples[n1], samples_CF[n1], n2, pred_len, allPlayers)

    def draw_trajectory3(self,samples, f_ax, allPlayers):
        # team_A: defense  team_B: attack
        T,D = samples.shape
        data = samples

        if D < 555:
            K = 5
            n_agents = K
            n_all_agents = 10
        else:
            K = 10
            n_agents = K+1
            n_all_agents = 22

        ball = data[:,n_all_agents*2:n_all_agents*2+2]
        data_mat = data[:,:n_all_agents*2].reshape(T,n_all_agents,2)
        team_A = data_mat[:,:n_agents,:]
        team_B = data_mat[:,n_agents:n_agents*2,:]
        
        #plt.axis('equal')
        self.plotCourt(K)
        timestep = 50

        markersize = 10 
        TextCircle = 0
        for i in range(0, timestep-2, 1): # 
            self.plotPosition3(team_A,team_B,ball,K,i,markersize,TextCircle,allPlayers)

        # initial marker 
        index = 0 
        if K == 5:
            markersize = 0.5
            TextCircle = 1
        else:
            markersize = 0.5
            TextCircle = 2
        self.plotPosition3(team_A,team_B,ball,K,index,markersize,TextCircle,allPlayers)

        # last marker 
        if K == 5:
            markersize = 0.5
        else:
            markersize = 60
        index = timestep-1 
        TextCircle = 2
        self.plotPosition3(team_A,team_B,ball,K,index,markersize,TextCircle,allPlayers)

    def plotPosition3(self,team_A,team_B,ball,K,i,markersize,TextCircle,allPlayers):
        if allPlayers:
            start_pl = 0  
            Tm = 2
        else:
            start_pl = 0  if K == 5 else 1
            Tm = 1

        n_all_agent = 10 if K == 5 else 22
        n_team_agent = 5 if K == 5 else 11

        ax = plt.gca() 
        im_team = [[],[]] 

        if allPlayers:
            if TextCircle == 1:
                im_ball = patches.Circle((ball[i, 0], ball[i, 1]), radius=markersize-0.15, fc='orange',ec="k")
                ax.add_patch(im_ball)
                ax.text(ball[i, 0], ball[i, 1],'B',color='w',ha='center',va='center')
            elif TextCircle == 0:
                im_ball = patches.Circle((ball[i, 0], ball[i, 1]), radius=0.1, fc='orange',ec="k")
                ax.add_patch(im_ball)
                im_ball = plt.scatter(ball[i, 0], ball[i, 1], marker=".", s=markersize, fc='orange',ec="k") # for legend only

        for tm in range(Tm): 
            pos = team_A if tm == 0 else team_B
            clr = 'b' if tm == 0 else 'r'
            for j in range(start_pl,n_team_agent):
                xx = pos[i, j, 0]
                yy = pos[i, j, 1]         

                if TextCircle == 1: # player circle

                    if tm == 1: # offense
                        im_team[tm] = patches.Circle((xx,yy), radius=markersize-0.15,
                                        fc=clr,ec='k')
                    else:
                        im_team[tm] = patches.CirclePolygon((xx,yy), radius = markersize,
                            resolution = 3, fc = clr, ec = "k") # triangle

                    ax.add_patch(im_team[tm])

                    # player jersey # (text)
                    ax.text(xx,yy,str(j+1),color='w',ha='center',va='center')
                elif TextCircle == 2:
                    ax.text(xx,yy,str(j+1),color='k',ha='center',va='center')

                else: 
                    im_team[tm] = plt.scatter(xx,yy, marker=".", s=markersize, ec=clr, color=clr)


        if allPlayers:
            if TextCircle == 2:
                ax.text(ball[i, 0], ball[i, 1],'B',color='k',ha='center',va='center')
                im_ball = []
        else:
            im_ball = []

        im_teamA, im_teamB = im_team

        return im_ball, im_teamA, im_teamB 

    def plotCourt(self,K):
        if K == 5:
            court_path ='meta_data/nba_court_T.png'
            feet_m = 0.3048 
            img = mpimg.imread(court_path) 
            plt.imshow(img, extent=[0,94*feet_m,0,50*feet_m], zorder=0) 
            plt.xlim(0,47*feet_m)  
            plt.ylim(0,50*feet_m) 

        else: # if K == 10: 
            plt.xlim(-52.5,52.5)  #  -52.5~52.5, -34~34 10, 40
            plt.ylim(-34,34) # -34,34 -21, 21

            plt.vlines(0, -34, 34, linestyles="solid") # center line
            plt.vlines(36, -20.16, 20.16, linestyles="solid") # penalty area
            plt.hlines(-20.16, 36, 52.5, linestyles="solid")
            plt.hlines(20.16, 36, 52.5, linestyles="solid")
            plt.vlines(-36, -20.16, 20.16, linestyles="solid")
            plt.hlines(-20.16, -36, -52.5, linestyles="solid")
            plt.hlines(20.16, -36, -52.5, linestyles="solid")
            plt.vlines(47, -9.16, 9.16, linestyles="solid") # goal area
            plt.hlines(9.16, 47, 52.5, linestyles="solid")
            plt.hlines(-9.16, 47, 52.5, linestyles="solid")
            plt.vlines(-47, -9.16, 9.16, linestyles="solid")
            plt.hlines(9.16, -47, -52.5, linestyles="solid")
            plt.hlines(-9.16, -47, -52.5, linestyles="solid")

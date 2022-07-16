import torch
import torch.nn as nn

from vrnn.models.utils import parse_model_params, get_params_str, cudafy_list, index_by_agent, get_macro_ohe
from vrnn.models.utils import sample_gauss, nll_gauss, kld_gauss, sample_multinomial
from vrnn.models.utils import batch_error, roll_out, sample_gumbel, sample_gumbel_softmax
import torch.nn.functional as F

# Keisuke Fujii, 2020
# modifying the code https://github.com/ezhan94/multiagent-programmatic-supervision

class RNN_GAUSS(nn.Module):
    """RNN model for each agent."""

    def __init__(self, params, parser=None):
        super().__init__()

        self.model_args = ['x_dim', 'y_dim', 'z_dim', 'h_dim', 'rnn_dim', 'n_layers', 'n_agents']
        self.params = params
        self.params_str = get_params_str(self.model_args, params)

        x_dim = params['x_dim'] # action
        y_dim = params['y_dim'] # state 
        z_dim = params['z_dim']
        h_dim = params['h_dim']
        rnn_dim = params['rnn_dim']
        n_layers = params['n_layers']
        self.in_out = params['in_out'] 
        self.in_sma = params['in_sma'] 
        self.wo_cross = params['wo_cross']

        # embedding
        embed_size = params['embed_size']
        self.embed_size = embed_size
        embed_ball_size = params['embed_ball_size'] 
        self.embed_ball_size = embed_ball_size

        # parameters 
        n_all_agents = params['n_all_agents'] # all players        
        n_agents = params['n_agents']
        n_feat = params['n_feat']  # dim
        ball_dim = params['ball_dim']
        
        dropout = 0.5 # 
        # dropout2 = 0
        self.xavier = True # initial value
        self.att_in = False # customized attention input
        self.res = params['res'] # False # like resnet  

        self.batchnorm = True # if self.attention >= 2 else False
        self.in_state0 = True # raw current state input
        self.fixedsigma = False 
        print('batchnorm = '+str(self.batchnorm)+ ', fixedsigma = '+str(self.fixedsigma))
        # currently not considerd 
        if params['acc'] == -1: # and self.params['body']: # body_pretrain:
            x_dim = 2

        # network parameters 
        if self.in_state0:
            in_state0 = x_dim
        else: 
            in_state0 = 0

        # RNN
        if self.batchnorm:
            self.bn_dec = nn.ModuleList([nn.BatchNorm1d(h_dim) for i in range(n_agents)]) 

        in_enc = x_dim+in_state0+y_dim+rnn_dim           

        self.dec = nn.ModuleList([nn.Sequential(
            nn.Linear(rnn_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout)) for i in range(n_agents)])
        self.dec_mean = nn.ModuleList([nn.Linear(h_dim, x_dim) for i in range(n_agents)])
        self.dec_std = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus()) for i in range(n_agents)])
        
        self.dec = nn.ModuleList([nn.Sequential(
            nn.Linear(rnn_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout)) for i in range(n_agents)])
        self.dec_mean = nn.ModuleList(
            [nn.Linear(h_dim, x_dim) for i in range(n_agents)])
        self.dec_std = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus()) for i in range(n_agents)])

        self.rnn = nn.ModuleList([nn.GRU(in_enc, rnn_dim, n_layers) for i in range(n_agents)])
        # self.rnn = nn.ModuleList([nn.GRU(y_dim, rnn_dim, n_layers) for i in range(n_agents)])

    def weights_init(self,m):
        # https://discuss.pytorch.org/t/weight-initialization-with-a-custom-method-in-nn-sequential/24846
        # https://blog.snowhork.com/2018/11/pytorch-initialize-weight
        if self.xavier: 
            # for mm in range(len(m)):
            mm=0
            if type(m) == nn.Linear: # in ,nn.GRU
                nn.init.xavier_normal_(m.weight)
            elif type(m) == nn.GRU:
                nn.init.xavier_normal_(m.weight_hh_l0)
                nn.init.xavier_normal_(m.weight_ih_l0)

    def forward(self, states, rollout, train, hp=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        acc = self.params['acc']
        out = {}
        out2 = {}
        out['L_rec'] = torch.zeros(1).to(device)
        out2['e_pos'] = torch.zeros(1)
        out2['e_vel'] = torch.zeros(1)
        out2['e_acc'] = torch.zeros(1)  
        out2['e_jrk'] = torch.zeros(1)  
        
        n_agents = self.params['n_agents']
        n_feat = self.params['n_feat'] # added
        ball_dim = self.params['ball_dim']
        fs = self.params['fs'] # added
        x_dim = self.params['x_dim']  
        burn_in = hp['burn_in'] # self.params['burn_in']

        batchSize = states.size(2)
        len_time = self.params['horizon'] #states.size(0)

        h = [torch.zeros(self.params['n_layers'], batchSize, self.params['rnn_dim']) for i in range(n_agents)]
        if self.params['cuda']:
            h = cudafy_list(h)

        for t in range(len_time):
            prediction_all = torch.zeros(batchSize, n_agents, x_dim)

            for i in range(n_agents):  
                y_t = states[t][i].clone() # state
                if self.in_out:
                    x_t0 = states[t+1][i].clone() # pos, vel, acc
                elif self.in_sma:
                    x_t0 = states[t+1][i][:,n_feat*i:n_feat*i+n_feat].clone() 
                elif n_feat == 13:
                    x_t0 = states[t+1][i][:,n_feat*i+3:n_feat*i+x_dim+5].clone() 
                elif n_feat == 15:
                    x_t0 = states[t+1][i][:,n_feat*i+3:n_feat*i+x_dim+7].clone() # pos, vel, acc

                # action
                if acc == 0: 
                    x_t = x_t0[:,2:4] # vel 
                elif acc == 1: 
                    x_t = x_t0[:,0:4] # pos,vel 
                elif acc == 2: 
                    x_t = x_t0[:,2:6] # vel,acc 
                elif acc == 3: 
                    x_t = x_t0[:,0:6] 
                elif acc == -1:
                    x_t = x_t0[:,:2] # pos

                if self.in_sma:
                    current_pos = y_t[:,n_feat*i:n_feat*i+2]
                    if acc >= 0:
                        current_vel = y_t[:,n_feat*i+2:n_feat*i+4]    
                        v0_t1 = x_t0[:,2:4]
                        
                        if acc >= 2:
                            current_acc = y_t[:,n_feat*i+4:n_feat*i+6]   
                        else:
                            current_acc = (x_t0[:,2:4] - current_vel)/fs
                        
                        current_acc = (v0_t1 - current_vel)/fs
                elif self.in_out:
                    current_pos = y_t[:,0:2]
                    current_vel = y_t[:,2:4]
                    # current_acc
                else:
                    current_pos = y_t[:,n_feat*i+3:n_feat*i+5]
                    current_vel = y_t[:,n_feat*i+5:n_feat*i+7]
                
                if self.in_state0:
                    if acc == 3:
                        state_in0 = torch.cat([current_pos,current_vel,current_acc], 1)
                    elif acc == 2:
                        state_in0 = torch.cat([current_vel,current_acc], 1)
                    elif acc == 0:
                        state_in0 = current_vel
                    elif acc == -1:
                        state_in0 = current_pos
                else:
                    state_in0 = torch.zeros(batchSize,0).to(device)

                # RNN
                state_in = y_t
                enc_in = torch.cat([x_t, state_in0, state_in, h[i][-1]], 1)

                dec_t = self.dec[i](h[i][-1])
                dec_mean_t = self.dec_mean[i](dec_t)
                if not self.fixedsigma:
                    dec_std_t = self.dec_std[i](dec_t)
                else:
                    dec_std_t = self.fixedsigma**2*torch.ones(dec_mean_t.shape).to(device)  
                _, h[i] = self.rnn[i](enc_in.unsqueeze(0), h[i])

                # objective function
                if acc == -1: 
                    out['L_rec'] += nll_gauss(dec_mean_t[:,:2], dec_std_t[:,:2], torch.cat([x_t],1))
                else:
                    out['L_rec'] += nll_gauss(dec_mean_t, dec_std_t, x_t)

                # body constraint
                # acc                    
                if acc == 1 or acc == 3:
                    v_t1 = dec_mean_t[:,2:4]
                    next_pos = dec_mean_t[:,:2]
                elif acc == 0 or acc == 2:
                    v_t1 = dec_mean_t[:,:2]   
                    next_pos = current_pos + v_t1*fs

                if t >= burn_in or burn_in==len_time:
                    # prediction
                    prediction_all[:,i,:] = dec_mean_t[:,:x_dim]    

                    # error (not used when backward)
                    out2['e_pos'] += batch_error(next_pos, x_t0[:,:2])
                    out2['e_vel'] += batch_error(v_t1, v0_t1)

                    if rollout and self.in_out: # for acc == 3, TBD
                        states[t+1,i,:,:] = torch.cat([next_pos,v_t1],dim=1)
                    del v_t1, current_pos, next_pos
            # role out
            if t >= burn_in: #  and not self.in_out: # if rollout:
                import pdb; pdb.set_trace()
                for i in range(n_agents):
                    y_t = states[t][i].clone() # state
                    y_t1i = states[t+1][i].clone() 
                    states[t+1][i] = roll_out(y_t,y_t1i,prediction_all,acc,self.params['normalize'],
                        n_agents,n_feat,ball_dim,fs,batchSize,i)#,self.wo_cross

        if burn_in==len_time:
            out2['e_pos'] /= (len_time)*n_agents
            out2['e_vel'] /= (len_time)*n_agents
        else: 
            out2['e_pos'] /= (len_time-burn_in)*n_agents
            out2['e_vel'] /= (len_time-burn_in)*n_agents
        out['L_rec'] /= (len_time)*n_agents
        return out, out2

    def sample(self, states, rollout, burn_in=0, L_att = False, CF_pred=False, n_sample=1,TEST=False):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        out = {}
        out2 = {}
        batchSize = states.size(2)
        len_time = self.params['horizon'] #states.size(0)
        out['L_rec'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
        out['e_pos'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
        out2['e_vel'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
        out2['e_pmax'] = torch.zeros(n_sample,batchSize,len_time).to(device) if len_time==burn_in else torch.zeros(n_sample,batchSize,len_time-burn_in).to(device)
        out2['e_vmax'] = torch.zeros(n_sample,batchSize,len_time).to(device) if len_time==burn_in else torch.zeros(n_sample,batchSize,len_time-burn_in).to(device)
                                
        Sum = True if not TEST else False

        acc = self.params['acc']
        body = self.params['body']
        n_agents = self.params['n_agents']
        n_all_agents = self.params['n_all_agents']
        n_feat = self.params['n_feat'] # added
        ball_dim = self.params['ball_dim']
        fs = self.params['fs'] # added
        x_dim = self.params['x_dim']
        burn_in = self.params['burn_in']

        batchSize = states.size(2)
        len_time = self.params['horizon'] #states.size(0)
        len_time0 = len_time

        h = [[torch.zeros(self.params['n_layers'], batchSize, self.params['rnn_dim']) for _ in range(n_sample)] for i in range(n_agents)]
        # missing value
        
        missing_indices = torch.zeros(batchSize).to(device)
        for b in range(batchSize):
            #if torch.sum(states[:,:,b]>9998)>0:
            missing_indices[b] = torch.sum(states[:,0,b,0]<9999)
            states[int(missing_indices[b]):,:,b,:] = states[int(missing_indices[b])-1,:,b,:]

        if self.params['cuda']:
            states = cudafy_list(states)
            for i in range(n_agents):
                h[i] = cudafy_list(h[i])
                self.rnn[i] = self.rnn[i].to(device)
                self.dec[i] = self.dec[i].to(device)
                self.dec_std[i] = self.dec_std[i].to(device) 
                self.dec_mean[i] = self.dec_mean[i].to(device)
                if self.batchnorm:
                    self.bn_dec[i] = self.bn_dec[i].to(device)   

        states_n = [states.clone() for _ in range(n_sample)]

        for t in range(len_time):
            for n in range(n_sample):
                prediction_all = torch.zeros(batchSize, n_agents, x_dim)

                for i in range(n_agents):
                    y_t = states_n[n][t][i].clone() # states[t][i].clone() # state

                    if self.in_out:
                        x_t0 = states[t+1][i].clone() # pos, vel, acc
                    elif n_feat < 10:
                        x_t0 = states[t+1][i][:,n_feat*i:n_feat*i+n_feat].clone() 
                    elif n_feat == 13:
                        x_t0 = states[t+1][i][:,n_feat*i+3:n_feat*i+x_dim+5].clone() 
                    elif n_feat == 15:
                        x_t0 = states[t+1][i][:,n_feat*i+3:n_feat*i+x_dim+7].clone() # pos, vel, acc

                    # action
                    if acc == 0: 
                        x_t = x_t0[:,2:4] # vel 
                    elif acc == 1: 
                        x_t = x_t0[:,0:4] # pos,vel 
                    elif acc == 2: 
                        x_t = x_t0[:,2:6] # vel,acc 
                    elif acc == 3: 
                        x_t = x_t0[:,0:6] 
                    elif acc == -1:
                        x_t = x_t0[:,:2] # pos

                    # for evaluation
                    if self.in_sma:
                        current_pos = y_t[:,n_feat*i:n_feat*i+2]
                        if acc >= 0:
                            current_vel = y_t[:,n_feat*i+2:n_feat*i+4]    
                            v0_t1 = x_t0[:,2:4]

                            if acc >= 2:
                                current_acc = y_t[:,n_feat*i+4:n_feat*i+6]   
                            else:
                                current_acc = (x_t0[:,2:4] - current_vel)/fs
                                
                    elif self.in_out:
                        current_pos = y_t[:,0:2]
                        current_vel = y_t[:,2:4]
                    else:
                        current_pos = y_t[:,n_feat*i+3:n_feat*i+5]
                        current_vel = y_t[:,n_feat*i+5:n_feat*i+7]

                    if self.in_state0:
                        if acc == 3:
                            state_in0 = torch.cat([current_pos,current_vel,current_acc], 1)
                        elif acc == 2:
                            state_in0 = torch.cat([current_vel,current_acc], 1)
                        elif acc == 0:
                            state_in0 = current_vel
                        elif acc == -1:
                            state_in0 = current_pos
                    else:
                        state_in0 = torch.zeros(batchSize,0).to(device)

                    state_in = y_t
                    enc_in = torch.cat([x_t, state_in0, state_in, h[i][n][-1]], 1)

                    dec_t = self.dec[i](h[i][n][-1])
                    if self.batchnorm:
                        try:
                            dec_t = self.bn_dec[i](dec_t) 
                        except: import pdb; pdb.set_trace() 
                    dec_mean_t = self.dec_mean[i](dec_t)
                    if not self.fixedsigma:
                        dec_std_t = self.dec_std[i](dec_t)
                    else:
                        dec_std_t = self.fixedsigma**2*torch.ones(dec_mean_t.shape).to(device)  
                    # objective function
                    out['L_rec'][n] += nll_gauss(dec_mean_t, dec_std_t, x_t, Sum)

                    # body constraint
                    # acc 
                    if acc == 1 or acc == 3:
                        v_t1 = dec_mean_t[:,2:4]
                        next_pos = dec_mean_t[:,:2]
                    elif acc == 0 or acc == 2:
                        v_t1 = dec_mean_t[:,:2]   
                        next_pos = current_pos + v_t1*fs

                    if t >= burn_in: # and not CF_pred:
                        # prediction
                        prediction_all[:,i,:] = dec_mean_t[:,:x_dim]

                        # error (not used when backward)
                        out['e_pos'][n] += batch_error(next_pos, x_t0[:,:2], Sum)
                        out2['e_vel'][n] += batch_error(v_t1, v0_t1, Sum)

                        if burn_in==len_time:
                            out2['e_pmax'][n,:,t] += batch_error(next_pos, x_t0[:,:2], Sum=False)
                            # TBD
                        else:
                            out2['e_pmax'][n,:,t-burn_in] += batch_error(next_pos, x_t0[:,:2], Sum=False)
                            out2['e_vmax'][n,:,t-burn_in] += batch_error(v_t1, v0_t1, Sum=False)

                        if rollout and self.in_out: # for acc == 3, TBD
                            states[n][t+1][i] = torch.cat([next_pos,v_t1],dim=1)
                        del current_pos, current_vel, next_pos

                    _, h[i][n] = self.rnn[i](enc_in.unsqueeze(0), h[i][n])

                # role out
                if t >= burn_in and not self.in_out: # rollout:
                    for i in range(n_agents):
                        y_t = states_n[n][t][i].clone() # state
                        y_t1i = states_n[n][t+1][i].clone() 
                        states_n[n][t+1][i] = roll_out(y_t,y_t1i,prediction_all,acc,self.params['normalize'],
                                n_agents,n_feat,ball_dim,fs,batchSize,i,self.wo_cross)

        if burn_in==len_time:
            out['e_pos'] /= (len_time)*n_agents
            out2['e_vel'] /= (len_time)*n_agents       
            # TBD
            # out2['e_pmax'] = torch.max(out2['e_pmax']/n_agents,dim=2)[0]
        else: 
            non_nan = missing_indices 
            non_nan[non_nan<len_time0+1] -= burn_in + 1
            non_nan[non_nan==len_time0+1] = len_time0 -burn_in
            for n in range(n_sample):
                #if torch.sum(non_nan) == (len_time0-burn_in)*batchSize:
                if not TEST:
                    out['e_pos'][n] /= (len_time0-burn_in)*n_agents
                    out2['e_vel'][n] /= (len_time0-burn_in)*n_agents
                else:
                    out['e_pos'][n] /= non_nan*n_agents
                    out2['e_vel'][n] /= non_nan*n_agents
            if not TEST: # validation
                out2['e_pmax'] = torch.sum(torch.max(out2['e_pmax']/n_agents,dim=2)[0])
                out2['e_vmax'] = torch.sum(torch.max(out2['e_vmax']/n_agents,dim=2)[0])
            else:
                out2['e_pmax'] = torch.max(out2['e_pmax']/n_agents,dim=2)[0] 
                out2['e_vmax'] = torch.max(out2['e_vmax']/n_agents,dim=2)[0] 
        for n in range(n_sample):
            out['L_rec'][n] /= (len_time)*n_agents

        if TEST: # n_sample > 1:
            states = states_n
        soft_att = []
        hard_att = []
        return states, soft_att, hard_att, out, out2

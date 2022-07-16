import torch
import torch.nn as nn

from vrnn.models.utils import parse_model_params, get_params_str, cudafy_list, index_by_agent, get_macro_ohe
from vrnn.models.utils import sample_gauss, nll_gauss, kld_gauss, sample_multinomial
from vrnn.models.utils import batch_error, roll_out, sample_gumbel, sample_gumbel_softmax
import torch.nn.functional as F

# Keisuke Fujii, 2020
# modifying the code https://github.com/ezhan94/multiagent-programmatic-supervision

class MACRO_VRNN(nn.Module):

    def __init__(self, params, parser=None):
        super().__init__()
        self.model_args = ['x_dim', 'y_dim', 'z_dim', 'h_dim', 'm_dim', 'rnn_micro_dim', 'rnn_macro_dim', 'rnn_att_dim', 'n_layers', 'n_agents']
        self.params = params
        self.params_str = get_params_str(self.model_args, params)
        self.attention = params['attention'] 
        self.wo_macro = params['wo_macro'] 
        self.wo_cross = params['wo_cross'] 
        self.macro = (not self.wo_macro)
        if self.attention > 5: 
            self.attention = -1
        self.hard_only = params['hard_only'] 
        self.in_out = params['in_out'] 
        self.in_sma = params['in_sma'] 
        self.indep = (self.attention ==3) #>= -1)

        x_dim = params['x_dim']
        y_dim = params['y_dim']
        z_dim = params['z_dim']
        h_dim = params['h_dim']
        m_dim = params['m_dim'] if self.macro else 0
        rnn_micro_dim = params['rnn_micro_dim']
        rnn_macro_dim = params['rnn_macro_dim']
        rnn_att_dim = params['rnn_att_dim']
        n_layers = params['n_layers']
        n_agents = params['n_agents']

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
        self.L_acc = self.params['body'] # constraints
        self.jrk = self.params['jrk'] if self.L_acc else 0 # constraints
        self.init_pthname0 = self.params['init_pthname0'] 

        dropout = 0.5 # 
        self.xavier = True # initial value
        self.res = params['res'] # like resnet  

        self.beta = 0.01 if params['dataset'] == 'nba' else 0.01 
        self.gamma1 = 0.1 if params['dataset'] == 'nba' else 0.01  
        self.gamma2 = self.params['lam_acc'] 
        self.batchnorm = True  
        self.fixedsigma = False
        if self.params['body']:
            print('beta = '+str(self.beta)+', gamma1 = '+str(self.gamma1)+', gamma2 = '+str(self.gamma2)+', jrk = '+str(self.jrk)) # 
        print('batchnorm = '+str(self.batchnorm)+ ', fixedsigma = '+str(self.fixedsigma))
        self.in_state0 = True # raw current state input
        
        # currently not considerd 
        if params['acc'] == -1: # and self.params['body']: # body_pretrain:
            x_dim = 2
        self.gru_att = False # for attention RNN (currently not considered)
        self.body_pretrain = (params['pretrain2']>0) # two-state learning  (currently not considered)

        # network parameters 
        if self.in_state0:
            in_state0 = x_dim
        else: 
            in_state0 = 0
        rnn_in_x = x_dim
        in_state = embed_size*n_all_agents + embed_ball_size

        # macro intents decoder 
        if self.indep: # w/ attention
            self.dec_macro = nn.ModuleList([nn.Sequential(
                nn.Linear(in_state+rnn_macro_dim, h_dim), # y_dim
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h_dim, m_dim),
                nn.LogSoftmax(dim=-1)) for i in range(n_agents)])
        elif self.macro:# w/o attention
            if self.batchnorm:
                self.dec_macro = nn.ModuleList([nn.Sequential(
                    nn.Linear(n_all_agents*2+2+rnn_macro_dim, h_dim), # y_dim
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(h_dim),
                    nn.Linear(h_dim, m_dim),
                    nn.LogSoftmax(dim=-1)) for i in range(n_agents)])     
            else:
                self.dec_macro = nn.ModuleList([nn.Sequential(
                    nn.Linear(n_all_agents*2+2+rnn_macro_dim, h_dim), # y_dim
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(h_dim, m_dim),
                    nn.LogSoftmax(dim=-1)) for i in range(n_agents)])        
        #if params['pretrain'] == 0:
        #    self.dec_macro.apply(self.pretrain_load)

        # att =  0: no attention (do not consider)
        # att =  1: whole attention (do not consider)
        # att =  2: individual soft attention
        # att =  3: individual soft+hard attention
        # att = -1: w/o embedding (decentralized)
        # att = -2: w/o embedding (centralized)

        feat_in = n_feat # 
        feat_in_ball = ball_dim #  
        if self.attention > 0 and n_feat == 6:
            feat_in = 4

        # individual embedding
        self.enc_ind = nn.ModuleList([nn.ModuleList() for i in range(n_agents)])  
        self.enc_ind2 = nn.ModuleList([nn.ModuleList() for i in range(n_agents)])  
        self.dim_ind_embed = 1 if self.attention == 2 else 2 # if 3　self.hard_only == True or

        # batch normalization
        if self.batchnorm:
            self.bn1 = nn.ModuleList([nn.ModuleList() for i in range(n_agents)]) 
            self.bn2 = nn.ModuleList([nn.ModuleList() for i in range(n_agents)])   

        
        for i in range(n_agents):
            if self.attention >= 2: # individual
                # player
                self.enc_ind[i] = nn.ModuleList([nn.Sequential( 
                    nn.Linear(feat_in, embed_size),
                    nn.Softplus()) for ii in range(n_all_agents)])  

                if self.batchnorm:
                    self.bn1[i] = nn.ModuleList([nn.BatchNorm1d(embed_size) for ii in range(n_all_agents)]) 
                    self.bn2[i] = nn.ModuleList([nn.BatchNorm1d(self.dim_ind_embed) for ii in range(n_all_agents)]) 

                self.enc_ind2[i] = nn.ModuleList([nn.Sequential(      
                    nn.Linear(embed_size, self.dim_ind_embed),
                    nn.Softplus()) for ii in range(n_all_agents)]) # nn.Sigmoid()

                for ii in range(n_all_agents):
                    self.enc_ind[i][ii].apply(self.weights_init)
                    self.enc_ind2[i][ii].apply(self.weights_init)
                
                # ball
                self.enc_ind[i].append(nn.Sequential(
                    nn.Linear(feat_in_ball, embed_ball_size),
                    nn.Softplus()))  
                self.enc_ind2[i].append(nn.Sequential(
                    nn.Linear(embed_ball_size, self.dim_ind_embed),
                    nn.Softplus())) # nn.Sigmoid()
                self.enc_ind[i][ii+1].apply(self.weights_init)
                self.enc_ind2[i][ii+1].apply(self.weights_init)

                if self.batchnorm:
                    self.bn1[i].append(nn.BatchNorm1d(embed_ball_size)) 
                    self.bn2[i].append(nn.BatchNorm1d(self.dim_ind_embed)) 

            elif self.attention >= 0: # whole
                self.enc_ind[i] = nn.Sequential(nn.Linear(y_dim, in_state)) #n_all_agents+1))
                self.enc_ind[i].apply(self.weights_init)

        # VRNN
        if self.attention == -1: # or not self.indep: 
            in_state = y_dim        
        
        if self.batchnorm:
            self.bn_enc = nn.ModuleList([nn.BatchNorm1d(h_dim) for i in range(n_agents)]) 
            self.bn_prior = nn.ModuleList([nn.BatchNorm1d(h_dim) for i in range(n_agents)])   
            self.bn_dec = nn.ModuleList([nn.BatchNorm1d(h_dim) for i in range(n_agents)]) 
         
        in_prior = in_state0 + in_state+m_dim+rnn_micro_dim  
        in_enc = in_prior + rnn_in_x
        in_enc_pre = in_prior + x_dim       

        self.enc = nn.ModuleList([nn.Sequential(
            nn.Linear(in_enc, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout)) for i in range(n_agents)])
        for i in range(n_agents):
            self.enc[i].apply(self.weights_init)
            self.enc[i].apply(self.weights_init)

        self.enc_mean = nn.ModuleList([nn.Linear(h_dim, z_dim) for i in range(n_agents)])
        self.enc_std = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus()) for i in range(n_agents)])

        self.prior = nn.ModuleList([nn.Sequential(
            nn.Linear(in_prior, h_dim), # m_dim+rnn_micro_dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout)) for i in range(n_agents)])
        for i in range(n_agents):
            self.prior[i].apply(self.weights_init)
            self.prior[i].apply(self.weights_init)
        self.prior_mean = nn.ModuleList([nn.Linear(h_dim, z_dim) for i in range(n_agents)])
        self.prior_std = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus()) for i in range(n_agents)])

        self.dec = nn.ModuleList([nn.Sequential(
            nn.Linear(in_state0 + in_state+m_dim+z_dim+rnn_micro_dim, h_dim), # y_dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout)) for i in range(n_agents)])
        for i in range(n_agents):
            self.dec[i].apply(self.weights_init)
            self.dec[i].apply(self.weights_init)
        self.dec_mean = nn.ModuleList([nn.Linear(h_dim, rnn_in_x) for i in range(n_agents)])

        #if not self.fixedsigma:
        self.dec_std = nn.ModuleList([nn.Sequential(
            nn.Linear(h_dim, rnn_in_x),
            nn.Softplus()) for i in range(n_agents)])       

        self.gru_micro = nn.ModuleList([nn.GRU(rnn_in_x+z_dim, rnn_micro_dim, n_layers) for i in range(n_agents)])
        if self.macro:
            self.gru_macro = nn.ModuleList([nn.GRU(m_dim, rnn_macro_dim, n_layers) for i in range(n_agents)])
            
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

    def func_attention(self, y_t, h, i, batchSize,Sample=False,macro=False):
        
        n_feat = self.params['n_feat'] 
        if n_feat == 6:
            pl_dim = 4
        else:
            pl_dim = n_feat
        n_all_agents = self.params['n_all_agents'] 
        temperature = self.params['temperature']
        ball_dim = self.params['ball_dim']
        embed_size =  self.params['embed_size']
        embed_ball_size =  self.params['embed_ball_size'] 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if Sample:
            self.enc_ind = self.enc_ind.to(device)
            self.enc_ind2 = self.enc_ind2.to(device)
            # $self.fc_ind_att = self.fc_ind_att.to(device)
            if self.attention == 1:
                self.fc_soft_att = self.fc_soft_att.to(device)
            if self.gru_att:
                self.rnn_att = self.rnn_att.to(device)

        # individual embedding 
        if self.attention >= 2: # individual
            ind_embed = []
            ind_embed2 = []
            for ii in range(n_all_agents+1):
                ii_dim = pl_dim if ii < n_all_agents else ball_dim
                ball = False if ii < n_all_agents else True

                tmp_ind = self.enc_ind[i][ii](y_t[:, ii*n_feat:ii*n_feat+ii_dim])
                if self.batchnorm:
                    tmp_ind = self.bn1[i][ii](tmp_ind)
                ind_embed.append(tmp_ind)
                tmp_ind2 = self.enc_ind2[i][ii](tmp_ind)
                if self.batchnorm:
                    tmp_ind2 = self.bn2[i][ii](tmp_ind2)
                ind_embed2.append(tmp_ind2) # (batch, embed_dim)
        
        # soft attention
        if self.attention >= 2: # individual attention
            emb_cat = torch.stack(ind_embed2,dim=1)

        if self.attention == 2:
            axis_soft = 1 if self.attention == 2 else 2
            soft_att = F.softmax(emb_cat, dim=axis_soft)[:,:,0]
        else:
            soft_att = []

        # hard attention
        if self.attention == 3: 
            hard_att = sample_gumbel_softmax(emb_cat,temperature)[:,:,0]
        else:
            hard_att = []


        if torch.sum(torch.abs(self.enc_ind[0][0][0].weight))>1e10:
            print('weigh is very large')
            print(self.enc_ind[0][0][0].weight)
            print(self.enc_ind[0][0][0].bias)
            import pdb; pdb.set_trace()

        if torch.isnan(hard_att).any():
            print('hard_att includes NaN')
            print(self.enc_ind[0][0][0].weight)
            print(self.enc_ind[0][0][0].bias)
            import pdb; pdb.set_trace()
            #hard_att = torch.ones(hard_att.shape).to(device)
        ind_embed = torch.cat(ind_embed,dim=1)

        if macro:
            if self.attention == 2:
                state_in = self.multiply_attention(ind_embed,soft_att,device,batchSize,n_all_agents)
            elif self.attention == 3:
                state_in = self.multiply_attention(ind_embed,hard_att,device,batchSize,n_all_agents)
            dec_macro_t = self.dec_macro[i](torch.cat([state_in, h], 1))
        else: 
            dec_macro_t = []
        return soft_att, hard_att, ind_embed, dec_macro_t

    def multiply_attention(self,ind_embed,att,device,batchSize,n_all_agents):
        state_in = torch.zeros(0, batchSize).to(device)
        for k in range(n_all_agents+1):
            state_in = torch.cat([state_in,ind_embed[:,self.embed_size*k:self.embed_size*(k+1)].transpose(0,1)*att[:,k]],0)
        return state_in.transpose(0,1)
    
    # only in acc == -1
    def state2pva(self,y_t,pva):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_all_agents = self.params['n_all_agents']
        n_feat = self.params['n_feat'] 
        batchSize = y_t.size(0)
        
        pos_t = torch.zeros(batchSize, n_all_agents*2+2).to(device)
        for i in range(n_all_agents+1):
            if self.in_sma:
                pos_t[:,i*2:i*2+2] = y_t[:,n_feat*i:n_feat*i+2]  
            elif self.in_out:
                pos_t[:,i*2:i*2+2] = y_t[:,0:2]
            else:
                pos_t[:,i*2:i*2+2] = y_t[:,n_feat*i+3:n_feat*i+5]

        if pva==1:
            out_t = pos_t
        return out_t

    def forward(self, states, rollout, train, macro=None, hp=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        acc = self.params['acc']
        body = self.params['body']
        out = {}
        out2 = {}
        if hp['pretrain']:
            out['L_mac'] = torch.zeros(1).to(device)
            out2['dummy'] = torch.zeros(1)
        else:
            out['L_kl'] = torch.zeros(1).to(device)
            out['L_rec'] = torch.zeros(1).to(device)
            out2['e_pos'] = torch.zeros(1)
            out2['e_vel'] = torch.zeros(1)
            out2['e_acc'] = torch.zeros(1)  
            out2['e_jrk'] = torch.zeros(1)  
            if self.macro:
                out2['e_mac'] = torch.zeros(1)
            if self.attention == 3:
                out2['att'] = torch.zeros(1).to(device) 
            if self.L_acc:
                out['L_vel'] = torch.zeros(1).to(device)
                out['L_acc'] = torch.zeros(1).to(device)
            if body:
                out['L_jrk'] = torch.zeros(1).to(device)           

        n_agents = self.params['n_agents']
        n_feat = self.params['n_feat'] # added
        ball_dim = self.params['ball_dim']
        x_dim = self.params['x_dim']
        fs = self.params['fs']  
        burn_in = hp['burn_in'] 
        n_all_agents = self.params['n_all_agents']

        batchSize = states.size(2)
        len_time = self.params['horizon'] #states.size(0)

        # states_single = index_by_agent(states, n_agents)
        h_micro = [torch.zeros(self.params['n_layers'], batchSize, self.params['rnn_micro_dim']) for i in range(n_agents)]
        if self.macro:
            macro_single = get_macro_ohe(macro, n_agents, self.params['m_dim']) # one-hot encode
            h_macro = [torch.zeros(self.params['n_layers'], batchSize, self.params['rnn_macro_dim']) for i in range(n_agents)]
        else:
            m_t = [torch.zeros(batchSize,0).to(device) for i in range(n_agents)]
            h_macro = [torch.zeros(self.params['n_layers'], batchSize, 0) for i in range(n_agents)]   

        if self.params['cuda']:
            if self.macro:
                h_macro = cudafy_list(h_macro)
            h_micro = cudafy_list(h_micro)

        for t in range(len_time):
            
            if self.macro:
                m_t = macro_single[t].clone() # (agents,batch,one-hot)
                if hp['pretrain'] or not self.indep:
                    for i in range(n_agents):
                        y_t = states[t][i].clone()
                        if self.indep:
                            _, _, _, dec_macro_t = self.func_attention(y_t, h_macro[i][-1], i, batchSize,macro=True)
                        else:
                            pos_t = self.state2pva(y_t,1)
                            dec_macro_t = self.dec_macro[i](torch.cat([pos_t, h_macro[i][-1]], 1)) # (batch, m_dim)
                        _, h_macro[i] = self.gru_macro[i](torch.cat([m_t[i]], 1).unsqueeze(0), h_macro[i])

                        if hp['pretrain']:
                            out['L_mac'] -= torch.sum(m_t[i]*dec_macro_t)
                        elif not self.indep:
                            out2['e_mac'] -= torch.sum(m_t[i]*dec_macro_t)

            if not hp['pretrain']:
                prediction_all = torch.zeros(batchSize, n_agents, x_dim)
                for i in range(n_agents):
                    y_t = states[t][i].clone()                      

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
                    elif acc == 4: 
                        x_t = x_t0[:,4:6] 
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
                                
                    elif self.in_out:
                        current_pos = y_t[:,0:2]
                        current_vel = y_t[:,2:4]
                    else:
                        current_pos = y_t[:,n_feat*i+3:n_feat*i+5]
                        current_vel = y_t[:,n_feat*i+5:n_feat*i+7]
                    
                    if self.in_state0:
                        if acc == 3:
                            state_in0 = torch.cat([current_pos,current_vel,current_acc], 1)
                        elif acc == 4:
                            state_in0 = current_acc
                        elif acc == 2:
                            state_in0 = torch.cat([current_vel,current_acc], 1)
                        elif acc == 0:
                            state_in0 = current_vel
                        elif acc == -1:
                            state_in0 = current_pos
                    else:
                        state_in0 = torch.zeros(batchSize,0).to(device)

                    # attention 
                    if self.attention >= 1: # individual 
                        soft_att, hard_att, ind_embed, dec_macro_t = self.func_attention(y_t, h_macro[i][-1], i, batchSize, Sample=False, macro=(self.macro))
                        if self.indep and self.macro:
                            out2['e_mac'] -= torch.sum(m_t[i]*dec_macro_t)

                        if self.attention == 3: # hard
                            out2['att'] += torch.sum(hard_att)#/(n_all_agents+1)
                            state_in = self.multiply_attention(ind_embed,hard_att,device,batchSize,n_all_agents)

                        elif self.attention >= 1: # soft
                            state_in = self.multiply_attention(ind_embed,soft_att,device,batchSize,n_all_agents)

                    elif self.attention == 0 : # w/ whole embedding 
                        state_in = self.enc_ind[i](y_t)

                    elif self.attention == -1: # w/o embedding and attention
                        state_in = y_t
                    #elif not self.indep:
                    #    state_in = torch.zeros(batchSize,0).to(device)

                    prior_in = torch.cat([state_in0, state_in, m_t[i], h_micro[i][-1]], 1)
                    enc_in = torch.cat([x_t, prior_in], 1)
                    enc_t = self.enc[i](enc_in)

                    if self.batchnorm:
                        enc_t = self.bn_enc[i](enc_t)
                    enc_mean_t = self.enc_mean[i](enc_t)
                    enc_std_t = self.enc_std[i](enc_t)

                    prior_t = self.prior[i](prior_in)
                    if self.batchnorm:
                        prior_t = self.bn_prior[i](prior_t)
                    prior_mean_t = self.prior_mean[i](prior_t)
                    prior_std_t = self.prior_std[i](prior_t)
                    
                    z_t = sample_gauss(enc_mean_t, enc_std_t)

                    dec_t = self.dec[i](torch.cat([state_in0,state_in, m_t[i], z_t, h_micro[i][-1]], 1))
                    if self.batchnorm:
                        dec_t = self.bn_dec[i](dec_t)

                    dec_mean_t = self.dec_mean[i](dec_t)
                    if self.res:
                        if acc == 3:
                            dec_mean_t[:,4:6] += state_in0[:,4:6]
                        elif acc == -1:
                            dec_mean_t += state_in0     
        
                    if not self.fixedsigma:
                        dec_std_t = self.dec_std[i](dec_t)
                    else:
                        dec_std_t = self.fixedsigma**2*torch.ones(dec_mean_t.shape).to(device)  
                    
                    _, h_micro[i] = self.gru_micro[i](torch.cat([x_t, z_t], 1).unsqueeze(0), h_micro[i])

                    if torch.isnan(h_micro[i][-1][0][0]):
                        import pdb; pdb.set_trace()
                    # objective function
                    out['L_kl'] += kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)


                    if acc == -1: 
                        out['L_rec'] += nll_gauss(dec_mean_t[:,:2], dec_std_t[:,:2], torch.cat([x_t],1))
                    else:
                        out['L_rec'] += nll_gauss(dec_mean_t, dec_std_t, x_t)
                        if self.L_acc:
                            if acc == 3: 
                                out['L_rec'] += self.gamma2*nll_gauss(dec_mean_t[:,2:6], dec_std_t[:,2:6], x_t[:,2:6])
                            elif acc == 4: 
                                out['L_rec'] += self.gamma2*nll_gauss(dec_mean_t, dec_std_t, x_t)
                            elif acc == 2: #or acc == 0:
                                out['L_rec'] += self.gamma2*nll_gauss(dec_mean_t[:,2:4], dec_std_t[:,2:4], x_t[:,2:4])

                    del enc_t, prior_t, dec_t, z_t

                    # body constraint
                    # acc                    
                    if acc == 1 or acc == 3:
                        v_t1 = dec_mean_t[:,2:4]
                        next_pos = dec_mean_t[:,:2]
                    elif acc == 4:
                        v_t1 = current_vel + current_acc*fs 
                        next_pos = current_pos + current_vel*fs 
                    elif acc == 0 or acc == 2:
                        v_t1 = dec_mean_t[:,:2]   
                        next_pos = current_pos + current_vel*fs 
                    elif acc == -1:
                        next_pos = dec_mean_t[:,:2]

                    if self.L_acc:
                        if acc == 2:
                            if t > 0: 
                                out['L_vel'] += self.beta*nll_gauss(current_pos+dec_mean_t0[:,0:2]*fs, dec_std_t0[:,0:2]*fs, next_pos)
                        elif acc == 0:
                            out['L_vel'] += self.beta*batch_error(next_pos, x_t0[:,:2])
                        else:                    
                            out['L_vel'] += self.beta*batch_error(v_t1, v0_t1)               

                    # evaluation (not learned)
                    if t >= burn_in or burn_in==len_time:
                        # prediction
                        prediction_all[:,i,:] = dec_mean_t[:,:x_dim]    
 
                        # error (not used when backward)
                        out2['e_pos'] += batch_error(next_pos, x_t0[:,:2])
                        out2['e_vel'] += batch_error(v_t1, v0_t1)

                        if rollout and self.in_out: # for acc == 3, TBD
                            states[t+1,i,:,:] = torch.cat([next_pos,v_t1],dim=1)
                        del v_t1, current_pos, next_pos
                        if acc >= 0:
                            del current_vel
                    # update 
                    if acc == 2 and self.L_acc: 
                        dec_mean_t0 = dec_mean_t
                        dec_std_t0 = dec_std_t
                # role out
                if t >= burn_in: #  and not self.in_out: # if rollout:
                    for i in range(n_agents):
                        y_t = states[t][i].clone() # state
                        y_t1i = states[t+1][i].clone() 
                        states[t+1][i] = roll_out(y_t,y_t1i,prediction_all,acc,self.params['normalize'],
                            n_agents,n_feat,ball_dim,fs,batchSize,i)

        if hp['pretrain']:
            out['L_mac'] /= (len_time)*n_agents
        else:
            if burn_in==len_time:
                out2['e_pos'] /= (len_time)*n_agents
                out2['e_vel'] /= (len_time)*n_agents
                if self.macro:         
                    out2['e_mac'] /= (len_time)*n_agents  
            else: 
                out2['e_pos'] /= (len_time-burn_in)*n_agents
                out2['e_vel'] /= (len_time-burn_in)*n_agents
                if self.macro:
                    out2['e_mac'] /= (len_time-burn_in)*n_agents
            out['L_kl'] /= (len_time)*n_agents
            out['L_rec'] /= (len_time)*n_agents
            if self.L_acc:
                out['L_vel'] /= (len_time)*n_agents
            if self.attention == 3:
                out2['att'] /= (len_time)*n_agents
        return out, out2

    def sample(self, states, rollout, burn_in=0, fix_m=[], L_att = False, CF_pred=False, n_sample=1, TEST=False): # macro
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        out = {}
        out2 = {}
        batchSize = states.size(2)
        len_time = self.params['horizon'] 
        out2['L_kl'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
        out['L_rec'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
        out['e_pos'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
        out2['e_vel'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
        out2['e_acc'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)  
        out2['e_jrk'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)  
        out2['e_pmax'] = torch.zeros(n_sample,batchSize,len_time).to(device) if len_time==burn_in else torch.zeros(n_sample,batchSize,len_time-burn_in).to(device)
        out2['e_vmax'] = torch.zeros(n_sample,batchSize,len_time).to(device) if len_time==burn_in else torch.zeros(n_sample,batchSize,len_time-burn_in).to(device)
        out2['e_amax'] = torch.zeros(n_sample,batchSize,len_time).to(device) if len_time==burn_in else torch.zeros(n_sample,batchSize,len_time-burn_in).to(device)
        if self.attention == 3:
            out2['att'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)   
        acc = self.params['acc']
        body = self.params['body']
        # if self.L_acc:
        out2['L_vel'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
        out2['L_acc'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
        #if body:
        out2['L_jrk'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)  
        out2['a_acc'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
        
        Sum = True if not TEST else False

        # missing value
        
        missing_indices = torch.zeros(batchSize).to(device)
        for b in range(batchSize):
            #if torch.sum(states[:,:,b]>9998)>0:
            missing_indices[b] = torch.sum(states[:,0,b,0]<9999)
            states[int(missing_indices[b]):,:,b,:] = states[int(missing_indices[b])-1,:,b,:]

        n_agents = self.params['n_agents']
        n_all_agents = self.params['n_all_agents']
        n_feat = self.params['n_feat'] # added
        ball_dim = self.params['ball_dim']
        fs = self.params['fs'] # added
        x_dim = self.params['x_dim']
        
        if self.macro:
            macro_single = get_macro_ohe(macro, n_agents, self.params['m_dim'])
            h_macro = [[torch.zeros(self.params['n_layers'], batchSize, self.params['rnn_macro_dim']) for _ in range(n_sample)] for i in range(n_agents)]
            macro_intents = torch.zeros(len_time, batchSize, n_agents, n_sample)# states.size(0)
        else:
            macro_intents = []
            m_t = [torch.zeros(batchSize,0).to(device) for i in range(n_agents)]
            h_macro = [[torch.zeros(self.params['n_layers'], batchSize, 0) for _ in range(n_sample)] for i in range(n_agents)]

        h_micro = [[torch.zeros(self.params['n_layers'], batchSize, self.params['rnn_micro_dim']) for _ in range(n_sample)] for i in range(n_agents)]
               
        soft_att = [] # torch.zeros(len_time,n_agents,batchSize, n_all_agents+1, n_sample) 
        hard_att = torch.zeros(len_time,n_agents,batchSize, n_all_agents+1, n_sample) 

        if self.params['cuda']:
            if self.macro:
                macro_intents = macro_intents.cuda()
                macro_single = macro_single.cuda()

            # soft_att = soft_att.cuda()
            hard_att = hard_att.cuda()
            states = cudafy_list(states)
            for i in range(n_agents):
                h_micro[i] = cudafy_list(h_micro[i])
                self.gru_micro[i] = self.gru_micro[i].to(device)
                if self.macro:
                    h_macro[i] = cudafy_list(h_macro[i])
                    self.gru_macro[i] = self.gru_macro[i].to(device)
                    self.dec_macro[i] = self.dec_macro[i].to(device)
                
                self.enc_ind[i] = self.enc_ind[i].to(device)
                self.enc[i] = self.enc[i].to(device)
                self.enc_mean[i] = self.enc_mean[i].to(device)
                self.enc_std[i] = self.enc_std[i].to(device)
                self.prior[i] = self.prior[i].to(device)
                self.prior_mean[i] = self.prior_mean[i].to(device)
                self.prior_std[i] = self.prior_std[i].to(device)
                self.dec[i] = self.dec[i].to(device)
                self.dec_std[i] = self.dec_std[i].to(device) 
                self.dec_mean[i] = self.dec_mean[i].to(device)

                if self.batchnorm:
                    self.bn_enc[i] = self.bn_enc[i].to(device) 
                    self.bn_prior[i] = self.bn_prior[i].to(device)  
                    self.bn_dec[i] = self.bn_dec[i].to(device)   

        states_n = [states.clone() for _ in range(n_sample)]

        len_time0 = len_time
        for t in range(len_time):
            missing_index = (missing_indices > t)
            #else:
            for n in range(n_sample):
                if self.macro:
                    if t < burn_in:
                        m_t = macro_single[t].clone() # (agents,batch,one-hot)

                    if not self.indep:
                        for i in range(n_agents):
                            if t >= burn_in:
                                y_t = states_n[n][t][i].clone()
                                pos_t = self.state2pva(y_t,1)
                                
                                dec_macro_t = self.dec_macro[i](torch.cat([pos_t, h_macro[i][n][-1]], 1))
                                m_t[i] = sample_multinomial(torch.exp(dec_macro_t))
                                del y_t, pos_t

                            _, h_macro[i][n] = self.gru_macro[i](torch.cat([m_t[i]], 1).unsqueeze(0), h_macro[i][n])
                        
                        macro_intents[t,:,:,n] = torch.max(m_t, 2)[1].transpose(0,1)

                prediction_all = torch.zeros(batchSize, n_agents, x_dim)
                for i in range(n_agents):
                    y_t = states_n[n][t][i].clone()

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
                    elif acc == 4: 
                        x_t = x_t0[:,4:6] 
                    elif acc == -1:
                        x_t = x_t0[:,:2] # pos

                    # for evaluation
                    if self.in_sma:
                        current_pos = y_t[:,n_feat*i:n_feat*i+2]
                        if acc >= 0:
                            current_vel = y_t[:,n_feat*i+2:n_feat*i+4]    
                            v0_t1 = x_t0[:,2:4]
                                
                    elif self.in_out:
                        current_pos = y_t[:,0:2]
                        current_vel = y_t[:,2:4]
                    else:
                        current_pos = y_t[:,n_feat*i+3:n_feat*i+5]
                        current_vel = y_t[:,n_feat*i+5:n_feat*i+7]
                    
                    if self.in_state0:
                        if acc == 3:
                            state_in0 = torch.cat([current_pos,current_vel,current_acc], 1)
                        elif acc == 4:
                            state_in0 = current_acc
                        elif acc == 2:
                            state_in0 = torch.cat([current_vel,current_acc], 1)
                        elif acc == 0:
                            state_in0 = current_vel
                        elif acc == -1:
                            state_in0 = current_pos
                    else:
                        state_in0 = torch.zeros(batchSize,0).to(device)

                    # attention 
                    if self.attention >= 1: # individual 
                        _, hard_att[t,i,:,:,n], ind_embed, dec_macro_t = self.func_attention(y_t, h_macro[i][n][-1], i, batchSize, Sample=False, macro=(self.macro)) # soft_att[t,i,:,:,n]
                        if self.indep and self.macro:    
                            if t >= burn_in:
                                m_t[i] = sample_multinomial(torch.exp(dec_macro_t))

                            _, h_macro[i][n] = self.gru_macro[i](torch.cat([m_t[i]], 1).unsqueeze(0), h_macro[i][n])
                        
                        if self.attention == 3: # hard
                            if not TEST:
                                out2['att'][n] += torch.sum(hard_att[t,i,:,:,n])#/(n_all_agents+1)
                            else:
                                out2['att'][n] += torch.sum(hard_att[t,i,:,:,n],1)#/(n_all_agents+1)
                            state_in = self.multiply_attention(ind_embed,hard_att[t,i,:,:,n],device,batchSize,n_all_agents)

                        elif self.attention >= 1: # soft
                            state_in = self.multiply_attention(ind_embed,soft_att[t,i,:,:,n],device,batchSize,n_all_agents)
                        del ind_embed
                    elif self.attention == 0 : # w/ whole embedding 
                        state_in = self.enc_ind[i](y_t)

                    elif self.attention == -1: # w/o embedding and attention # or not self.indep
                        state_in = y_t # torch.zeros(batchSize,0).to(device) # 

                    prior_in = torch.cat([state_in0, state_in, m_t[i], h_micro[i][n][-1]], 1)
                    if False: # acc == -1: #  and self.body_pretrain:
                        enc_in = torch.cat([x_t,v0_t1,a0_t1, prior_in],1)
                    else:
                        enc_in = torch.cat([x_t, prior_in], 1)

                    prior_t = self.prior[i](prior_in)
                    if self.batchnorm:
                        prior_t = self.bn_prior[i](prior_t)                              
                    prior_mean_t = self.prior_mean[i](prior_t)
                    prior_std_t = self.prior_std[i](prior_t)

                    z_t = sample_gauss(prior_mean_t, prior_std_t)

                    dec_t = self.dec[i](torch.cat([state_in0, state_in, m_t[i], z_t, h_micro[i][n][-1]], 1))
                    if self.batchnorm:
                        dec_t = self.bn_dec[i](dec_t)  
                    
                    dec_mean_t = self.dec_mean[i](dec_t)
                    if self.res:
                        if acc == 3:
                            dec_mean_t[:,4:6] += state_in0[:,4:6]
                        elif acc == -1:
                            dec_mean_t += state_in0        
                    if not self.fixedsigma:       
                        dec_std_t = self.dec_std[i](dec_t)
                    else:
                        dec_std_t = self.fixedsigma**2*torch.ones(dec_mean_t.shape).to(device)  
                    # objective function
                    # for evaluation only
                    enc_t = self.enc[i](enc_in)
                    if self.batchnorm:
                        enc_t = self.bn_enc[i](enc_t)                     
                    enc_mean_t = self.enc_mean[i](enc_t)
                    enc_std_t = self.enc_std[i](enc_t)
                    out2['L_kl'][n] += kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t, Sum)
                    if acc == -1: 
                        out['L_rec'][n] += nll_gauss(dec_mean_t[:,:2], dec_std_t[:,:2], torch.cat([x_t],1))
                    else:                   
                        out['L_rec'][n] += nll_gauss(dec_mean_t, dec_std_t, x_t, Sum)

                    # body constraint
                    # acc 
                    if acc == 1 or acc == 3:
                        v_t1 = dec_mean_t[:,2:4]
                        next_pos = dec_mean_t[:,:2]
                    elif acc == 4:
                        v_t1 = current_vel + current_acc*fs 
                        next_pos = current_pos + current_vel*fs 
                    elif acc == 0 or acc == 2:
                        v_t1 = dec_mean_t[:,:2]   
                        next_pos = current_pos + current_vel*fs 
                    elif acc == -1:
                        next_pos = dec_mean_t[:,:2]               

                    if t >= burn_in or burn_in==len_time: # and not CF_pred:
                        # prediction
                        prediction_all[:,i,:] = dec_mean_t[:,:x_dim]

                        # error (not used when backward)
                        out['e_pos'][n] += batch_error(next_pos, x_t0[:,:2], Sum, index=missing_index)
                        #if t == len_time-1:
                        #    import pdb; pdb.set_trace()
                        out2['e_vel'][n] += batch_error(v_t1, v0_t1, Sum, index=missing_index)

                        if burn_in==len_time:
                            out2['e_pmax'][n,:,t] += batch_error(next_pos, x_t0[:,:2], Sum=False, index=missing_index)
                            # TBD
                        else:
                            out2['e_pmax'][n,:,t-burn_in] += batch_error(next_pos, x_t0[:,:2], Sum=False, index=missing_index)
                            out2['e_vmax'][n,:,t-burn_in] += batch_error(v_t1, v0_t1, Sum=False, index=missing_index)

                        # out['L_rec'][n] += out2['e_vel'][n] + out2['e_acc'][n]

                        if rollout and self.in_out: # for acc == 3, TBD
                            states[n][t+1][i] = torch.cat([next_pos,v_t1],dim=1)

                    del enc_t, prior_t, dec_t, enc_mean_t, prior_mean_t, enc_std_t, prior_std_t, enc_in, prior_in 
                    # update 
                    if acc == 2: # and self.L_acc: 
                        dec_mean_t0 = dec_mean_t
                        dec_std_t0 = dec_std_t
                    del dec_std_t, state_in, x_t0, v_t1, current_pos, next_pos, y_t 

                    if acc >= 0:
                        del current_vel
                    _, h_micro[i][n] = self.gru_micro[i](torch.cat([x_t, z_t], 1).unsqueeze(0), h_micro[i][n])
                    del x_t, z_t

                if self.macro and self.indep:    
                    macro_intents[t,:,:,n] = torch.max(m_t, 2)[1].transpose(0,1)
                # role out
                if t >= burn_in and not self.in_out: # rollout:
                    for i in range(n_agents):
                        y_t = states_n[n][t][i].clone() # state
                        y_t1i = states[t+1][i].clone() 
                        states_n[n][t+1][i] = roll_out(y_t,y_t1i,prediction_all,acc,self.params['normalize'],
                                n_agents,n_feat,ball_dim,fs,batchSize,i,self.wo_cross)
                        del y_t1i

        if self.macro:
            macro_intents.data[-1] = macro_intents.data[-2] # the last time step
        if burn_in==len_time:
            out['e_pos'] /= (len_time)*n_agents
            out2['e_vel'] /= (len_time)*n_agents
        else: 
            # non_nan = torch.sum(states[burn_in+1:,0,:,0]<9999,0)
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
                out2['e_amax'] = torch.sum(torch.max(out2['e_amax']/n_agents,dim=2)[0])
            else:
                out2['e_pmax'] = torch.max(out2['e_pmax']/n_agents,dim=2)[0] 
                out2['e_vmax'] = torch.max(out2['e_vmax']/n_agents,dim=2)[0] 
                out2['e_amax'] = torch.max(out2['e_amax']/n_agents,dim=2)[0] 


        for n in range(n_sample):
            out2['L_kl'][n] /= (len_time0)*n_agents
            out['L_rec'][n] /= (len_time0)*n_agents
            out2['L_vel'][n] /= (len_time0)*n_agents
        if self.attention == 3:
            for n in range(n_sample):
                out2['att'][n] /= (len_time0)*n_agents

        if TEST: # n_sample > 1:
            states = states_n
        return states, macro_intents, hard_att, out, out2

import torch
import torch.nn as nn
import numpy as np

from vrnn.models.utils import parse_model_params, get_params_str, cudafy_list, index_by_agent, get_macro_ohe
from vrnn.models.utils import sample_gauss, nll_gauss, kld_gauss, sample_multinomial
from vrnn.models.utils import batch_error, roll_out, sample_gumbel, sample_gumbel_softmax
import torch.nn.functional as F

# anonymous, 2021
# modifying the code https://github.com/ezhan94/multiagent-programmatic-supervision

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)



class GVRNN(nn.Module):

    def __init__(self, params, parser=None):
        super().__init__()
        self.model_args = ['x_dim', 'y_dim', 'z_dim', 'h_dim', 'm_dim', 'rnn_micro_dim', 'rnn_att_dim', 'n_layers', 'n_agents']
        self.params = params
        self.params_str = get_params_str(self.model_args, params)
        self.wo_cross = params['wo_cross'] 
        self.hard_only = params['hard_only'] 
        # self.in_out = params['in_out'] 
        self.in_sma = params['in_sma'] 

        x_dim = params['x_dim']
        y_dim = params['y_dim']
        z_dim = params['z_dim']
        # h_dim = params['h_dim']
        m_dim = 0
        rnn_micro_dim = params['rnn_micro_dim']
        rnn_att_dim = params['rnn_att_dim']
        n_layers = params['n_layers']
        n_agents = params['n_agents']
        

        # embedding
        embed_size = params['embed_size']
        self.embed_size = embed_size
        embed_ball_size = params['embed_ball_size'] 
        self.embed_ball_size = embed_ball_size

        # parameters 
        n_all_agents = params['n_all_agents']+1 # all players        
        n_agents = params['n_agents']
        n_feat = params['n_feat']  # dim
        ball_dim = params['ball_dim']
        self.init_pthname0 = self.params['init_pthname0'] 
        self.z_dim_each = int(z_dim//n_all_agents)

        dropout = 0.5 # 
        self.xavier = True # initial value
        self.res = params['res'] # like resnet  

        self.beta = 0.01 if params['dataset'] == 'nba' else 0.01 
        self.gamma1 = 0.1 if params['dataset'] == 'nba' else 0.01  
        self.gamma2 = self.params['lam_acc'] 
        self.batchnorm = True  
        self.fixedsigma = False
        print('batchnorm = '+str(self.batchnorm)+ ', fixedsigma = '+str(self.fixedsigma))

        rnn_in_x = x_dim
        in_state = embed_size*n_all_agents + embed_ball_size

        feat_in = n_feat # 
        feat_in_ball = ball_dim #  

        # individual embedding
        # self.enc_ind = nn.ModuleList([nn.ModuleList() for i in range(n_agents)])  
        #for i in range(n_agents):
        #    self.enc_ind[i] = nn.Sequential(nn.Linear(y_dim, in_state)) #n_all_agents+1))
        #    self.enc_ind[i].apply(self.weights_init)
        # self.enc_ind = nn.Sequential(nn.Linear(y_dim, in_state))

        # GNN--------------------------------------------
        self.factor = True
        self.n_edge_type = 2
        self.n_node_type = 2
        self.mlp1 = nn.ModuleList([nn.ModuleList() for i in range(3)])  
        self.mlp2 = nn.ModuleList([nn.ModuleList() for i in range(3)])  
        self.mlp3 = nn.ModuleList([nn.ModuleList() for i in range(3)])  
        #self.mlp4 = nn.ModuleList([nn.ModuleList() for i in range(3)])  
        self.fc_out = nn.ModuleList([nn.ModuleList() for i in range(3)])  

        for ped in range(3): # 0: prior, 1:encoder, 2:decoder
            if ped <= 1:
                n_in = n_feat # +rnn_micro_dim #*n_all_agents # 
            elif ped == 2:
                n_in = self.z_dim_each # +rnn_micro_dim 
            n_out = 8
            n_hid = 8
            do_prob = 0

            self.mlp1[ped] = MLP(n_in, n_hid, n_hid, do_prob)
            self.mlp2[ped] = MLP(n_hid * 2 + self.n_edge_type, n_hid, n_hid, do_prob)
            self.mlp3[ped] = MLP(n_hid + self.n_node_type, n_hid, n_out, do_prob)
            #if self.factor:
            #    self.mlp4[ped] = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            #else:
            #    self.mlp4[ped] = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            # self.fc_out[ped] = nn.Linear(n_hid, n_out)

            self.mlp1[ped].apply(self.weights_init)
            self.mlp2[ped].apply(self.weights_init)
            self.mlp3[ped].apply(self.weights_init)
            #self.mlp4[ped].apply(self.weights_init)
            # self.fc_out[ped].apply(self.weights_init)
            # self.init_weights()

        # rel_rec, rel_send with numpy 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        off_diag = np.ones([n_all_agents, n_all_agents]) - np.eye(n_all_agents)

        rel_rec = np.array(self.encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(self.encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        rel_rec = torch.FloatTensor(rel_rec).to(device)
        rel_send = torch.FloatTensor(rel_send).to(device)

        # corresponding with rel_rec/rel_send 
        edge_type = torch.zeros(1,n_all_agents*(n_all_agents-1),2).to(device)
        nonball_index = torch.arange(0,n_all_agents*(n_all_agents-1),(n_all_agents-1))[1:-1]
        # edge_type[:,:-(n_all_agents-1),0] = 1 # non-ball -> non-ball: [1 0]
        edge_type[:,nonball_index,0] = 1 # non-ball -> non-ball: [1 0] (receiver) :-(n_all_agents-1)?
        edge_type[:,-(n_all_agents-1):,1] = 1 # non-ball -> ball: [0 1] (sender) OK

        node_type = torch.zeros(1,n_all_agents,2).to(device)
        node_type[:,:-1,0] = 1 # non-ball: [1 0]
        node_type[:,-1,1] = 1 # ball: [0 1]
        self.rel_rec = rel_rec
        self.rel_send = rel_send
        self.edge_type = edge_type
        self.node_type = node_type
        # VRNN --------------------------------------------
         
        out_prior = n_out*n_all_agents + rnn_micro_dim*n_agents # *(n_all_agents-1) # in_state0 + in_state+m_dim+rnn_micro_dim  
        out_enc = out_prior # in_prior + rnn_in_x
        out_dec = n_out*n_all_agents 

        self.enc_mean = nn.Linear(out_enc, z_dim) 
        self.enc_std = nn.Sequential(
            nn.Linear(out_enc, z_dim),
            nn.Softplus()) 

        self.prior_mean = nn.Linear(out_prior, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(out_prior, z_dim),
            nn.Softplus())
        
        self.dec_mean = nn.Linear(out_dec, x_dim*n_agents) 

        if not self.fixedsigma:
            self.dec_std = nn.Sequential(
                nn.Linear(out_dec, x_dim*n_agents),
                nn.Softplus())   

        # self.gru_micro = nn.GRU(rnn_in_x+z_dim, rnn_micro_dim, n_layers)
        self.gru_micro = nn.ModuleList([nn.GRU(x_dim+self.z_dim_each, rnn_micro_dim, n_layers) for i in range(n_agents)])

    def edge2node(self, x, rel_rec, rel_send, node_type):
        # x: (batch,agents*(agents-1),hidden)
        # rel_rec: (agents*(agents-1),agents)
        # incoming: (batch,agents,hidden)
        # node_type: (batch,agents,2)
        incoming = torch.matmul(rel_rec.t(), x)
        incoming = torch.cat([incoming, node_type], dim=2)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send, edge_type):
        # x: (batch,agents,hidden)
        # rel_rec,rel_send: (agents*(agents-1),agents)
        # edge_type: (batch,agents*(agents-1),2)
        receivers = torch.matmul(rel_rec, x) # (batch,agents*(agents-1),hidden)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers, edge_type], dim=2)
        return edges

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

    def encode_onehot(self,labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                dtype=np.int32)
        return labels_onehot

    def GNN(self, x, ped): # 0: prior, 1:encoder, 2:decoder
        rel_rec, rel_send = self.rel_rec, self.rel_send
        edge_type, node_type = self.edge_type, self.node_type
        batchSize = x.shape[0]
        edge_type = edge_type.repeat(batchSize,1,1)
        node_type = node_type.repeat(batchSize,1,1)

        x = self.mlp1[ped](x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send, edge_type)
        x = self.mlp2[ped](x)
        # x_skip = x

        # if self.factor:
        x = self.edge2node(x, rel_rec, rel_send, node_type)
        x = self.mlp3[ped](x)
        #    # x = self.node2edge(x, rel_rec, rel_send, edge_type)
        #    # x = torch.cat((x, x_skip), dim=2)  # Skip connection
        #    # x = self.mlp4[ped](x)
        # else:
        #     x = self.mlp3[ped](x)
        #     # x = torch.cat((x, x_skip), dim=2)  # Skip connection
        #     # x = self.mlp4[ped](x)

        return x # self.fc_out[ped](x)

    def unnormalize(self,data,pos_vel):
        if self.params['normalize']:
            data_ = data.clone()
            if pos_vel == 0:
                data_[:,0] = data[:,0]*52.5
                data_[:,1] = data[:,1]*34
            elif pos_vel == 1:
                data_ = data*10
            return data_
        else:
            return data

    def forward(self, states, rollout, train, macro=None, hp=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        acc = self.params['acc']
        out = {}
        out2 = {}

        out['L_kl'] = torch.zeros(1).to(device)
        out['L_rec'] = torch.zeros(1).to(device)
        out2['e_pos'] = torch.zeros(1)
        if acc == 0:
            out2['e_vel'] = torch.zeros(1) 

        n_agents = self.params['n_agents']
        n_feat = self.params['n_feat'] # added
        ball_dim = self.params['ball_dim']
        x_dim = self.params['x_dim']
        fs = self.params['fs']  
        burn_in = hp['burn_in'] 
        n_all_agents = self.params['n_all_agents']+1
        normalize = self.params['normalize']

        batchSize = states.size(2)
        len_time = self.params['horizon'] #states.size(0)

        h_micro = [torch.zeros(self.params['n_layers'], batchSize, self.params['rnn_micro_dim']) for i in range(n_agents)]

        if self.params['cuda']:
            h_micro = cudafy_list(h_micro)

        for t in range(len_time):
            micro_in = [[] for _ in range(n_agents)]
            prediction_all = torch.zeros(batchSize, n_agents, x_dim)

            y_t = states[t][0].clone()  
            y_t_ = y_t.reshape(batchSize,n_all_agents,n_feat)
            x_t_ = states[t+1][0].reshape(batchSize,n_all_agents,n_feat)
            
            for i in range(n_agents):
                micro_in[i] = h_micro[i][-1]
                
            # GNN-----------------------------------------
            # prior & encorder
            
            micro_in = torch.stack(micro_in,dim=1)
            prior_in = y_t_ # torch.cat([,micro_in],dim=2) #.reshape(batchSize,-1)
            enc_in = x_t_ # torch.cat([,micro_in],dim=2) #.reshape(batchSize,-1)

            try: 
                prior_t = self.GNN(prior_in, ped=0)
                enc_t = self.GNN(enc_in, ped=1)
            except: import pdb; pdb.set_trace()

            prior_t = torch.cat([prior_t.reshape(batchSize,-1),micro_in.reshape(batchSize,-1)],1)
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            enc_t = torch.cat([enc_t.reshape(batchSize,-1),micro_in.reshape(batchSize,-1)],1)
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            # sample
            z_t = sample_gauss(enc_mean_t, enc_std_t)
            # objective function
            out['L_kl'] += kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

            del micro_in, prior_in, prior_t, enc_t, 
            del prior_mean_t, prior_std_t, enc_mean_t, enc_std_t 

            # decoder 
            z_t_ = z_t.reshape(batchSize,n_all_agents,self.z_dim_each)
            dec_in = z_t_ # torch.cat([, micro_in],dim=2)

            try: dec_t = self.GNN(dec_in, ped=2)
            except: import pdb; pdb.set_trace()

            if acc == 0: 
                # y_t__ = y_t_[:,:n_agents,2:4] # vel 
                dec_mean_t = self.dec_mean(dec_t.reshape(batchSize,-1))
            elif acc == -1:
                y_t__ = y_t_[:,:n_agents,:2] # pos
                dec_mean_t = self.dec_mean(dec_t.reshape(batchSize,-1)) + y_t__.reshape(batchSize,-1)
            
            if not self.fixedsigma:
                dec_std_t = self.dec_std(dec_t.reshape(batchSize,-1))
            else:
                dec_std_t = self.fixedsigma**2*torch.ones(dec_mean_t.shape).to(device)  

            # RNN update
            for i in range(n_agents):
                x_t0 = states[t+1][i][:,n_feat*i:n_feat*i+n_feat].clone() 
                
                current_pos = self.unnormalize(y_t[:,n_feat*i:n_feat*i+2],pos_vel=0)
                next_pos_true = self.unnormalize(x_t0[:,:2],pos_vel=1)

                # action
                if acc == 0: 
                    x_t = x_t0[:,2:4] # vel 
                    dec_mean_t_ = dec_mean_t[:,2*i:2*i+2]
                    dec_std_t_ = dec_std_t[:,2*i:2*i+2]

                    current_vel = self.unnormalize(y_t[:,n_feat*i+2:n_feat*i+4],pos_vel=1)    
                    v0_t1 = self.unnormalize(x_t0[:,2:4],pos_vel=1)  
                elif acc == -1:
                    x_t = x_t0[:,:2] # pos
                    dec_mean_t_ = dec_mean_t[:,4*i:4*i+2]
                    dec_std_t_ = dec_std_t[:,4*i:4*i+2]

                try: _, h_micro[i] = self.gru_micro[i](torch.cat([x_t, z_t_[:,i,:]], 1).unsqueeze(0), h_micro[i])#
                except: import pdb; pdb.set_trace()

                if torch.isnan(h_micro[i][-1][0][0]):
                    import pdb; pdb.set_trace()
                
                # objective function
                out['L_rec'] += nll_gauss(dec_mean_t_, dec_std_t_, x_t)
     
                if acc == 0:
                    v_t1 = self.unnormalize(dec_mean_t_,pos_vel=1) 
                    next_pos = current_pos + current_vel*fs 
                elif acc == -1:
                    next_pos = self.unnormalize(dec_mean_t_,pos_vel=0)      

                # evaluation (not learned)
                if t >= burn_in or burn_in==len_time:
                    # prediction
                    prediction_all[:,i,:] = dec_mean_t_    

                    # error (not used when backward)
                    out2['e_pos'] += batch_error(next_pos, next_pos_true)
                    if acc == 0:
                        out2['e_vel'] += batch_error(v_t1, v0_t1)
                
                del dec_mean_t_, dec_std_t_, current_pos
            del dec_in, dec_mean_t, dec_std_t, z_t, z_t_
            # role out (usually not used)
            if t >= burn_in:
                for i in range(n_agents):
                    y_t = states[t][i].clone() # state
                    y_t1i = states[t+1][i].clone() 
                    states[t+1][i] = roll_out(y_t,y_t1i,prediction_all,acc,self.params['normalize'],
                        n_agents,n_feat,ball_dim,fs,batchSize,i)

        
        if burn_in==len_time:
            out2['e_pos'] /= (len_time)*n_agents
            if acc == 0:
                out2['e_vel'] /= (len_time)*n_agents
        else: 
            out2['e_pos'] /= (len_time-burn_in)*n_agents
            if acc == 0:
                out2['e_vel'] /= (len_time-burn_in)*n_agents
        out['L_kl'] /= (len_time)*n_agents
        out['L_rec'] /= (len_time)*n_agents

        return out, out2

    def sample(self, states, rollout, burn_in=0, L_att = False, CF_pred=False, n_sample=1, TEST=False):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        out = {}
        out2 = {}
        batchSize = states.size(2)
        len_time = self.params['horizon'] 
        # out2['L_kl'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
        out['L_rec'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
        out['e_pos'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)
        out2['e_vel'] = torch.zeros(n_sample).to(device) if not TEST else torch.zeros(n_sample,batchSize).to(device)  
        out2['e_pmax'] = torch.zeros(n_sample,batchSize,len_time).to(device) if len_time==burn_in else torch.zeros(n_sample,batchSize,len_time-burn_in).to(device)
        out2['e_vmax'] = torch.zeros(n_sample,batchSize,len_time).to(device) if len_time==burn_in else torch.zeros(n_sample,batchSize,len_time-burn_in).to(device)

        acc = self.params['acc']
        normalize = self.params['normalize']
        Sum = True if not TEST else False

        # missing value
        missing_indices = torch.zeros(batchSize).to(device)
        for bb in range(batchSize):
            missing_indices[bb] = torch.sum(states[:,0,bb,0]<9999)
            states[int(missing_indices[bb]):,:,bb,:] = states[int(missing_indices[bb])-1,:,bb,:]
            # print(str(bb)+' '+str(missing_indices[bb]))
            if missing_indices[bb] < 1:
                import pdb; pdb.set_trace()
        
        n_agents = self.params['n_agents']
        n_all_agents = self.params['n_all_agents']+1
        n_feat = self.params['n_feat'] # added
        ball_dim = self.params['ball_dim']
        fs = self.params['fs'] # added
        x_dim = self.params['x_dim']
        
        h_micro = [[torch.zeros(self.params['n_layers'], batchSize, self.params['rnn_micro_dim']) for _ in range(n_sample)] for i in range(n_agents)]

        if self.params['cuda']:
            states = cudafy_list(states)
            for i in range(n_agents):
                h_micro[i] = cudafy_list(h_micro[i])
            self.gru_micro = self.gru_micro.to(device)

            # self.enc_mean = self.enc_mean.to(device)
            # self.enc_std = self.enc_std.to(device)
            self.prior_mean = self.prior_mean.to(device)
            self.prior_std = self.prior_std.to(device)
            self.dec_std = self.dec_std.to(device) 
            self.dec_mean = self.dec_mean.to(device)

        states_n = [states.clone() for _ in range(n_sample)]

        len_time0 = len_time
        for t in range(len_time):
            missing_index = (missing_indices > t)
            #else:
            for n in range(n_sample):
                micro_in = [[] for _ in range(n_agents)]
                prediction_all = torch.zeros(batchSize, n_agents, x_dim)

                y_t = states_n[n][t][0].clone()  
                y_t_ = y_t.reshape(batchSize,n_all_agents,n_feat)
                try: x_t_ = states_n[n][t+1][0].reshape(batchSize,n_all_agents,n_feat)
                except: import pdb; pdb.set_trace()

                for i in range(n_agents):
                    micro_in[i] = h_micro[i][n][-1]
                    
                # GNN-----------------------------------------
                # prior & encorder
                micro_in = torch.stack(micro_in,dim=1)
                prior_in = y_t_ # torch.cat([,micro_in],dim=2) #.reshape(batchSize,-1)
                # enc_in = x_t_ # torch.cat([,micro_in],dim=2) #.reshape(batchSize,-1)

                try: 
                    prior_t = self.GNN(prior_in, ped=0)
                    # enc_t = self.GNN(enc_in, ped=1)
                except: import pdb; pdb.set_trace()

                prior_t = torch.cat([prior_t.reshape(batchSize,-1),micro_in.reshape(batchSize,-1)],1)
                prior_mean_t = self.prior_mean(prior_t)
                prior_std_t = self.prior_std(prior_t)

                # enc_t = torch.cat([enc_t.reshape(batchSize,-1),micro_in.reshape(batchSize,-1)],1)
                # enc_mean_t = self.enc_mean(enc_t)
                # enc_std_t = self.enc_std(enc_t)

                # sample
                z_t = sample_gauss(prior_mean_t, prior_std_t)

                # for evaluation only
                # out2['L_kl'][n] += kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

                del micro_in, prior_in, prior_t, prior_mean_t,prior_std_t 
                # del enc_t, enc_mean_t, enc_std_t 

                # decoder 
                z_t_ = z_t.reshape(batchSize,n_all_agents,self.z_dim_each)
                dec_in = z_t_ # torch.cat([, micro_in],dim=2)

                try: dec_t = self.GNN(dec_in, ped=2)
                except: import pdb; pdb.set_trace()

                if acc == 0: 
                    # y_t__ = y_t_[:,:n_agents,2:4] # vel 
                    dec_mean_t = self.dec_mean(dec_t.reshape(batchSize,-1)) 
                elif acc == -1:
                    y_t__ = y_t_[:,:n_agents,:2] # pos
                    dec_mean_t = self.dec_mean(dec_t.reshape(batchSize,-1)) + y_t__.reshape(batchSize,-1)
            
                
                
                if not self.fixedsigma:
                    dec_std_t = self.dec_std(dec_t.reshape(batchSize,-1))
                else:
                    dec_std_t = self.fixedsigma**2*torch.ones(dec_mean_t.shape).to(device)  

                # evaluate prediction 
                for i in range(n_agents):    
                    y_t = states_n[n][t][i].clone()

                    x_t0 = states[t+1][i][:,n_feat*i:n_feat*i+n_feat].clone() 
                    current_pos = self.unnormalize(y_t[:,n_feat*i:n_feat*i+2],pos_vel=0)
                    next_pos_true = self.unnormalize(x_t0[:,:2],pos_vel=1)

                    # action
                    if acc == 0: 
                        x_t = x_t0[:,2:4] # vel 
                        dec_mean_t_ = dec_mean_t[:,2*i:2*i+2]
                        dec_std_t_ = dec_std_t[:,2*i:2*i+2]
                        
                        current_vel = self.unnormalize(y_t[:,n_feat*i+2:n_feat*i+4],pos_vel=1)    
                        v0_t1 = self.unnormalize(x_t0[:,2:4],pos_vel=1)  

                    elif acc == -1:
                        x_t = x_t0[:,:2] # pos
                        dec_mean_t_ = dec_mean_t[:,4*i:4*i+2]
                        dec_std_t_ = dec_std_t[:,4*i:4*i+2]

                    # for evaluation only
                    out['L_rec'][n] += nll_gauss(dec_mean_t_, dec_std_t_, x_t)

                    if acc == 0:
                        v_t1 = self.unnormalize(dec_mean_t_,pos_vel=1) 
                        next_pos = current_pos + current_vel*fs 
                    elif acc == -1:
                        next_pos = self.unnormalize(dec_mean_t_,pos_vel=0)                 

                    if t >= burn_in or burn_in==len_time: 
                        # prediction
                        prediction_all[:,i,:] = dec_mean_t_

                        # error (not used when backward)
                        out['e_pos'][n] += batch_error(next_pos, next_pos_true, Sum, index=missing_index)
                        #if t == len_time-1:
                        #    import pdb; pdb.set_trace()
                        out2['e_vel'][n] += batch_error(v_t1, v0_t1, Sum, index=missing_index)
   
                        out2['e_pmax'][n,:,t-burn_in] += batch_error(next_pos, next_pos_true, Sum=False, index=missing_index)
                        out2['e_vmax'][n,:,t-burn_in] += batch_error(v_t1, v0_t1, Sum=False, index=missing_index)

                    del dec_mean_t_, dec_std_t_, current_pos
                del dec_in, dec_mean_t, dec_std_t, z_t
                # role out
                if t >= burn_in: # rollout:
                    for i in range(n_agents):
                        y_t = states_n[n][t][i].clone() # state
                        y_t1i = states[t+1][i].clone() 
                        states_n[n][t+1][i] = roll_out(y_t,y_t1i,prediction_all,acc,self.params['normalize'],
                                n_agents,n_feat,ball_dim,fs,batchSize,i,self.wo_cross)
                        del y_t1i

                # RNN update 
                for i in range(n_agents):
                    x_t0 = states[t+1][i][:,n_feat*i:n_feat*i+n_feat].clone() 
                    # action
                    if acc == 0: 
                        x_t = x_t0[:,2:4] # vel 
                    elif acc == -1:
                        x_t = x_t0[:,:2] # pos

                    try: _, h_micro[i][n] = self.gru_micro[i](torch.cat([x_t, z_t_[:,i,:]], 1).unsqueeze(0), h_micro[i][n])#
                    except: import pdb; pdb.set_trace()

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
            else:
                out2['e_pmax'] = torch.max(out2['e_pmax']/n_agents,dim=2)[0] 
                out2['e_vmax'] = torch.max(out2['e_vmax']/n_agents,dim=2)[0] 

        if torch.sum(torch.isinf(out['e_pos']))>0:
            import pdb; pdb.set_trace()
        for n in range(n_sample):
            # out2['L_kl'][n] /= (len_time0)*n_agents
            out['L_rec'][n] /= (len_time0)*n_agents

        if TEST: # n_sample > 1:
            states = states_n

        macro_intents = None
        hard_att = None
        return states, macro_intents, hard_att, out, out2

import torch
from torch.utils.data import Dataset
from importlib import import_module
import numpy as np 
import math
import cloudpickle

class GeneralDataset(Dataset):

    def __init__(self, args, len_seqs, train=0, normalize_data=True):
        #if normalize_data:
        #    for index in range(len(activeRole)):
        #        data_x[index] = dataset.normalize(data[index],datasets)

        self.train = train
        self.len_seqs = len_seqs
        self.game_files = args.game_files
        self.game_files_val = args.game_files_val
        self.game_files_te = args.game_files_te
        self.batchsize = args.batchsize
        self.model = args.model
        self.n_GorS = args.n_GorS
        self.attack_defend = args.attack_defend 

        self.normalize_data = normalize_data
        #self.datasets = args.dataset
 
    def __getitem__(self, index):
        if self.train == 1:
            batch_no = index // self.batchsize
            batch_index = index % self.batchsize
            with open(self.game_files+'_tr'+str(batch_no)+'.pkl', 'rb') as f:
                data,_,_,label = cloudpickle.load(f) # ,allow_pickle=True
            ind_player = []
        else:
            J = 4 if self.n_GorS == 1 else 8
            batch_no = index // math.ceil(self.len_seqs/J)
            # if batch_no < J:
            batch_index = index % math.ceil(self.len_seqs/J)
            #else:
            #    batch_no = J-1
            #    batch_index = index % math.ceil(self.len_seqs/J) + math.ceil(self.len_seqs/J)
            filename = self.game_files+'_val' if self.train == 0 else self.game_files +'_te'
            with open(filename+'_'+str(batch_no)+'.pkl', 'rb') as f:
                if self.attack_defend and self.train == -1 :
                    data,label,ind_player_,_ = cloudpickle.load(f)  
                    ind_player_ = np.array(ind_player_) # .transpose((1,0,2))
                    ind_player = ind_player_[batch_index]
                    #if batch_no==0:
                    #    import pdb; pdb.set_trace()
                else:
                    data,label = cloudpickle.load(f) 
                    ind_player = []

        self.data = torch.Tensor(data)
        self.data = self.data.permute(1, 0, 2, 3)

        if self.normalize_data:
            self.data[:,:,:,0:46:2] /= 52.5
            self.data[:,:,:,1:47:2] /= 34
            self.data[:,:,:,47:] /= 10
        #    self.data = self.dataset.normalize(self.data,self.datasets)
        return self.data[batch_index], label[batch_index], ind_player # data[index] 

    def __len__(self):
        return self.len_seqs
        
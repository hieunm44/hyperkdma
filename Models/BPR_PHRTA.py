import numpy as np
import torch
import torch.nn as nn

from Models.BPR import BPR


class PNet(nn.Module):
    def __init__(self, input_dim, num_params):
        super(PNet, self).__init__()

        self.mlp = nn.Sequential(
			nn.Linear(input_dim, (input_dim+num_params)//2),
			nn.ReLU(),
			nn.Linear((input_dim+num_params)//2, num_params)
	)
    
    def forward(self, x):
        return self.mlp(x)


class BPR_PHRTA(BPR):
    def __init__(self, user_count, item_count, teacher_user_emb, teacher_item_emb, student_dim, user_train_mat, TAs_dims, gpu):
        """
        Parameters
        ----------
        user_count (int): number of users
        item_count (int): number of items
        teacher_user_emb (2D FloatTensor): teacher user embeddings
        teacher_item_emb (2D FloatTensor): teacher item embeddings
        student_dim (int): dimension of embedding vectors of student model
        user_train_mat (dict): user-item interaction matrix
        TAs_dims (list): list of dimensions of TAs
        gpu: gpu device
        """
        
        BPR.__init__(self, user_count, item_count, student_dim, gpu)

        self.student_dim = student_dim
        self.gpu = gpu

        # teacher embedding
        self.teacher_user_emb = nn.Embedding.from_pretrained(teacher_user_emb) 
        self.teacher_item_emb = nn.Embedding.from_pretrained(teacher_item_emb)

        # fix the teacher embeddings
        self.teacher_user_emb.weight.requires_grad = False
        self.teacher_item_emb.weight.requires_grad = False

        # get the teacher dimension
        self.teacher_dim = self.teacher_user_emb.weight.size(1)

        # get dimensions of TAs
        self.TAs_dims = TAs_dims

        # PHR 
        self.user_train_mat = user_train_mat
        self.item_train_mat = self.gen_item_train_mat(user_train_mat)
        self.mask = self.gen_mask(user_train_mat)

        num_params_pnet = self.student_dim*self.teacher_dim
        self.pnet = PNet(self.teacher_dim, num_params_pnet)

        self.pnet_TAs = nn.ModuleList([PNet(self.TAs_dims[i], self.student_dim*self.TAs_dims[i]) for i in range(len(self.TAs_dims))])

        # hypernetwork
        num_params_hnet = self.teacher_dim*np.sum(TAs_dims)
        self.hnet = nn.Sequential(
            nn.Linear(self.teacher_dim, num_params_hnet)
        )

        # gate network
        self.gnet = nn.Sequential(
            nn.Linear(self.teacher_dim, len(TAs_dims)),
            nn.Softmax(dim=1)
        )

        print('Teacher dim:', self.teacher_dim)
        print('Student dim:', self.student_dim)
        print('TA dims: ', end='')
        for i in TAs_dims:
            print(i, '', end='')
        print()


    def gen_item_train_mat(self, user_train_mat):
        item_train_mat = {i: {} for i in range(self.item_count)}
        mask = np.zeros((self.user_count, self.item_count))
        
        for key, value in user_train_mat.items():
            for observed_item in value.keys():
                mask[key][observed_item] = 1
        
        for key in item_train_mat.keys():
            observed_user_list = np.where(mask[:, key] == 1)[0]
            item_train_mat[key] = {user: 1 for user in observed_user_list}
        
        return item_train_mat

    
    def gen_mask(self, train_mat):
        mask = np.zeros((self.user_count, self.item_count))
        for key, value in train_mat.items():
            for observed_item in value.keys():
                mask[key][observed_item] = 1
        return mask


    def enrich(self, user_batch, item_batch, is_user=True):
        if is_user:
            enr_emb = nn.Embedding.from_pretrained(self.teacher_user_emb(self.user_list))
            for u in user_batch.tolist():
                observed_items = np.where(self.mask[u, :] == 1)[0]
                observed_items_batch = set(observed_items).intersection(item_batch.tolist())
                if len(observed_items_batch) > 0:
                    enr_emb.weight[u] += self.teacher_item_emb(torch.LongTensor(list(observed_items_batch)).to(self.gpu)).sum(0)/len(observed_items_batch)
                    enr_emb.weight[u] /= 2            
        else:
            enr_emb = nn.Embedding.from_pretrained(self.teacher_item_emb(self.item_list))
            for i in item_batch.tolist():
                observed_users = np.where(self.mask[:, i] == 1)[0]
                observed_users_batch = set(observed_users).intersection(user_batch.tolist())
                if len(observed_users_batch) > 0:
                    enr_emb.weight[i] += self.teacher_user_emb(torch.LongTensor(list(observed_users_batch)).to(self.gpu)).sum(0)/len(observed_users_batch)
                    enr_emb.weight[i] /= 2  

        return enr_emb

    
    def enrich_TA(self, user_batch, item_batch, TA_id, is_user=True):
        if is_user:
            enr_emb = nn.Embedding.from_pretrained(self.user_emb_TAs[TA_id](self.user_list))
            for u in user_batch.tolist():
                observed_items = np.where(self.mask[u, :] == 1)[0]
                observed_items_batch = set(observed_items).intersection(item_batch.tolist())
                if len(observed_items_batch) > 0:
                    enr_emb.weight[u] += self.item_emb_TAs[TA_id](torch.LongTensor(list(observed_items_batch)).to(self.gpu)).sum(0)/len(observed_items_batch)
                    enr_emb.weight[u] /= 2            
        else:
            enr_emb = nn.Embedding.from_pretrained(self.item_emb_TAs[TA_id](self.item_list))
            for i in item_batch.tolist():
                observed_users = np.where(self.mask[:, i] == 1)[0]
                observed_users_batch = set(observed_users).intersection(user_batch.tolist())
                if len(observed_users_batch) > 0:
                    enr_emb.weight[i] += self.user_emb_TAs[TA_id](torch.LongTensor(list(observed_users_batch)).to(self.gpu)).sum(0)/len(observed_users_batch)
                    enr_emb.weight[i] /= 2  

        return enr_emb

    
    def reconstruct(self, X, weights):
        # weight = weights[:, :self.student_dim*self.teacher_dim].reshape(-1, self.teacher_dim, self.student_dim)
        # bias = weights[:, self.student_dim*self.teacher_dim:]
        out = torch.matmul(weights, X.unsqueeze(-1))
        # out = torch.add(out, bias.unsqueeze(-1))
        out = out.squeeze(-1)

        return out
        

    def get_PHR_loss(self, user_batch, item_batch, is_user=True): 
        if is_user:
            s = self.user_emb(user_batch) 
            t = self.teacher_user_emb(user_batch)

            # enr_t = self.enrich(user_batch, item_batch, is_user=True)(user_batch)
        else:
            s = self.item_emb(item_batch)
            t = self.teacher_item_emb(item_batch)

            # enr_t = self.enrich(user_batch, item_batch, is_user=False)(item_batch)

        weights = self.pnet(t).reshape(-1, self.teacher_dim, self.student_dim)
        reconstructed_emb = self.reconstruct(s, weights)
        PHR_loss = ((t - reconstructed_emb)**2).sum(-1).sum()
        
        return PHR_loss
    

    def get_PHRTA_loss(self, user_batch, item_batch, emb_weights, TA_id, is_user=True): 
        if is_user:
            s = self.user_emb(user_batch) 

            # enr_t = self.enrich_TA(user_batch, item_batch, TA_id, is_user=True)(user_batch)
        else:
            s = self.item_emb(item_batch)

            # enr_t = self.enrich_TA(user_batch, item_batch, TA_id, is_user=False)(item_batch)

        t = emb_weights
        weights = self.pnet_TAs[TA_id](t).reshape(-1, self.TAs_dims[TA_id], self.student_dim)
        reconstructed_emb = self.reconstruct(s, weights)
        PHR_loss = ((t - reconstructed_emb)**2).sum(-1).sum()
        
        return PHR_loss
import numpy as np
import torch
import torch.nn as nn

from Models.NeuMF import NeuMF


class PNet(nn.Module):
    def __init__(self, input_dim, num_params):
        super(PNet, self).__init__()

        self.mlp = nn.Sequential(
			nn.Linear(input_dim, (input_dim+num_params)),
			nn.ReLU(),
			nn.Linear((input_dim+num_params), num_params)
	)
    
    def forward(self, x):
        return self.mlp(x)


class NeuMF_PHR(NeuMF):
    def __init__(self, user_count, item_count, teacher_user_MF, teacher_item_MF, teacher_user_MLP, teacher_item_MLP, student_dim, num_layers, user_train_mat, gpu):
        """
        Parameters
        ----------
        user_count (int): number of users
        item_count (int): number of items
        teacher_user_MF (2D FloatTensor): teacher user MF embeddings
        teacher_item_MF (2D FloatTensor): teacher item MF embeddings
        teacher_user_MLP (2D FloatTensor): teacher user MLP embeddings
        teacher_item_MLP (2D FloatTensor): teacher item MLP embeddings
        student_dim (int): dimension of embedding vectors of student model
        num_layers (int): number of MLP layers
        user_train_mat: interaction matrix
        gpu: gpu device
        """

        NeuMF.__init__(self, user_count, item_count, student_dim, num_layers, gpu)

        self.student_dim = self.dim
        self.gpu = gpu

        # teacher embeddings
        self.teacher_user_MF = nn.Embedding.from_pretrained(teacher_user_MF)
        self.teacher_item_MF = nn.Embedding.from_pretrained(teacher_item_MF)
        self.teacher_user_MLP = nn.Embedding.from_pretrained(teacher_user_MLP)
        self.teacher_item_MLP = nn.Embedding.from_pretrained(teacher_item_MLP)

        self.teacher_user_MF.weight.requires_grad = False
        self.teacher_item_MF.weight.requires_grad = False
        self.teacher_user_MLP.weight.requires_grad = False
        self.teacher_item_MLP.weight.requires_grad = False

        self.teacher_dim = self.teacher_user_MF.weight.size(1)

        self.user_train_mat = user_train_mat
        self.item_train_mat = self.gen_item_train_mat(user_train_mat)
        self.mask = self.gen_mask(user_train_mat)

        num_params = self.student_dim*self.teacher_dim
        self.pnet = PNet(self.teacher_dim, num_params)


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


    def enrich(self, user_batch, item_batch, is_MF=True, is_user=True):
        if is_MF and is_user:
            enr_emb = nn.Embedding.from_pretrained(self.teacher_user_MF(self.user_list))
            for u in user_batch.tolist():
                observed_items = np.where(self.mask[u, :] == 1)[0]
                observed_items_batch = set(observed_items).intersection(item_batch.tolist())
                if len(observed_items_batch) > 0:
                    enr_emb.weight[u] += self.teacher_item_MF(torch.LongTensor(list(observed_items_batch)).to(self.gpu)).sum(0)/len(observed_items_batch)
                    enr_emb.weight[u] /= 2            
        elif is_MF and not is_user:
            enr_emb = nn.Embedding.from_pretrained(self.teacher_item_MF(self.item_list))
            for i in item_batch.tolist():
                observed_users = np.where(self.mask[:, i] == 1)[0]
                observed_users_batch = set(observed_users).intersection(user_batch.tolist())
                if len(observed_users_batch) > 0:
                    enr_emb.weight[i] += self.teacher_user_MF(torch.LongTensor(list(observed_users_batch)).to(self.gpu)).sum(0)/len(observed_users_batch)
                    enr_emb.weight[i] /= 2  
        if not is_MF and is_user:
            enr_emb = nn.Embedding.from_pretrained(self.teacher_user_MLP(self.user_list))
            for u in user_batch.tolist():
                observed_items = np.where(self.mask[u, :] == 1)[0]
                observed_items_batch = set(observed_items).intersection(item_batch.tolist())
                if len(observed_items_batch) > 0:
                    enr_emb.weight[u] += self.teacher_item_MLP(torch.LongTensor(list(observed_items_batch)).to(self.gpu)).sum(0)/len(observed_items_batch)
                    enr_emb.weight[u] /= 2            
        else:
            enr_emb = nn.Embedding.from_pretrained(self.teacher_item_MLP(self.item_list))
            for i in item_batch.tolist():
                observed_users = np.where(self.mask[:, i] == 1)[0]
                observed_users_batch = set(observed_users).intersection(user_batch.tolist())
                if len(observed_users_batch) > 0:
                    enr_emb.weight[i] += self.teacher_user_MLP(torch.LongTensor(list(observed_users_batch)).to(self.gpu)).sum(0)/len(observed_users_batch)
                    enr_emb.weight[i] /= 2  

        return enr_emb
    

    def reconstruct(self, X, weights):
        # weight = weights[:, :self.student_dim*self.teacher_dim].reshape(-1, self.teacher_dim, self.student_dim)
        # bias = weights[:, self.student_dim*self.teacher_dim:]
        out = torch.matmul(weights, X.unsqueeze(-1))
        # out = torch.add(out, bias.unsqueeze(-1))
        out = out.squeeze(-1)

        return out

    def get_PHR_loss(self, user_batch, item_batch, is_MF=True, is_user=True):
        if is_MF and is_user:
            s = self.user_emb_MF(user_batch)
            t = self.teacher_user_MF(user_batch)

            # enr_t = self.enrich(user_batch, item_batch, is_user=True)(user_batch)
        elif is_MF and not is_user:
            s = self.item_emb_MF(item_batch)
            t = self.teacher_item_MF(item_batch)

            # enr_t = self.enrich(user_batch, item_batch, is_user=True)(item_batch)
        elif not is_MF and is_user:
            s = self.user_emb_MLP(user_batch)
            t = self.teacher_user_MLP(user_batch)

            # enr_t = self.enrich(user_batch, item_batch, is_user=False)(user_batch)
        else:
            s = self.item_emb_MLP(item_batch)
            t = self.teacher_item_MLP(item_batch)

            # enr_t = self.enrich(user_batch, item_batch, is_user=False)(item_batch)

        weights = self.pnet(t).reshape(-1, self.teacher_dim, self.student_dim)
        reconstructed_emb = self.reconstruct(s, weights)
        PHR_loss = ((t - reconstructed_emb)**2).sum(-1).sum()

        return PHR_loss
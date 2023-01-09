import numpy as np
import torch
import torch.nn as nn

from Models.BPR import BPR


class Expert(nn.Module):
    def __init__(self, dims):
        super(Expert, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )
    
    def forward(self, x):
        return self.mlp(x)


class BPR_DETA(BPR):
    def __init__(self, user_count, item_count, teacher_user_emb, teacher_item_emb, student_dim, TAs_dims, num_experts, gpu):
        """
        Parameters
        ----------
        user_count (int): number of users
        item_count (int): number of items
        teacher_user_emb (2D FloatTensor): teacher user embeddings
        teacher_item_emb (2D FloatTensor): teacher item embeddings
        student_dim (int): dimension of embedding vectors of student model
        TAs_dims (list): list of dimensions of TAs
        num_experts (int): number of DEs
        gpu: gpu device
        """

        BPR.__init__(self, user_count, item_count, student_dim, gpu)

        self.student_dim = student_dim
        self.gpu = gpu

        # teacher embeddings
        self.teacher_user_emb = nn.Embedding.from_pretrained(teacher_user_emb)
        self.teacher_item_emb = nn.Embedding.from_pretrained(teacher_item_emb)

        # fix the teacher embeddings
        self.teacher_user_emb.weight.requires_grad = False
        self.teacher_item_emb.weight.requires_grad = False

        # get the teacher dimension
        self.teacher_dim = self.teacher_user_emb.weight.size(1)

        # get dimensions of TAs
        self.TAs_dims = TAs_dims

        # expert configuration
        self.num_experts = num_experts
        # define dimensions of the expert network
        expert_dims = [self.student_dim, (self.teacher_dim + self.student_dim)//2, self.teacher_dim]
        expert_dims_TAs = [[self.student_dim, (TA_dim + self.student_dim)//2, TA_dim] for TA_dim in self.TAs_dims]
        
        # user/item experts
        self.user_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
        self.item_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])

        self.user_experts_TAs = nn.ModuleList([nn.ModuleList([Expert(expert_dims_TA) for i in range(self.num_experts)]) for expert_dims_TA in expert_dims_TAs])
        self.item_experts_TAs = nn.ModuleList([nn.ModuleList([Expert(expert_dims_TA) for i in range(self.num_experts)]) for expert_dims_TA in expert_dims_TAs])

        # user/item selection networks
        self.user_selection_net = nn.Sequential(
            nn.Linear(self.teacher_dim, num_experts),
            nn.Softmax(dim=1)
        )
        self.item_selection_net = nn.Sequential(
            nn.Linear(self.teacher_dim, num_experts),
            nn.Softmax(dim=1)
        )

        self.user_selection_net_TA = nn.ModuleList([nn.Sequential(
            nn.Linear(TA_dim, num_experts),
            nn.Softmax(dim=1)
        ) for TA_dim in self.TAs_dims])
        self.item_selection_net_TA = nn.ModuleList([nn.Sequential(
            nn.Linear(TA_dim, num_experts),
            nn.Softmax(dim=1)
        ) for TA_dim in self.TAs_dims])
        
        # Gumbel-Softmax temperature
        self.T = 0.
        self.sm = nn.Softmax(dim = 1)

        # hypernetwork
        num_params = self.teacher_dim*np.sum(TAs_dims)
        self.hnet = nn.Sequential(
            nn.Linear(self.teacher_dim, num_params)
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
    

    def reconstruct(self, X, weights):
        # weight = weights[:, :self.student_dim*self.teacher_dim].reshape(-1, self.teacher_dim, self.student_dim)
        # bias = weights[:, self.student_dim*self.teacher_dim:]
        out = torch.matmul(weights, X.unsqueeze(-1))
        # out = torch.add(out, bias.unsqueeze(-1))
        out = out.squeeze(-1)

        return out


    def get_DE_loss(self, batch_entity, is_user=True):
        """
        Compute DE loss
        ----------

        Parameters
        ----------
        batch_entity (1-D Long Tensor): batch of users/items
        is_user (Bolean): distilling from user or item embeddings

        Returns
        ----------
        DE_loss (float): DE loss
        """

        if is_user:
            s = self.user_emb(batch_entity)
            t = self.teacher_user_emb(batch_entity) 

            experts = self.user_experts
            selection_net = self.user_selection_net
        else:
            s = self.item_emb(batch_entity)
            t = self.teacher_item_emb(batch_entity)

            experts = self.item_experts
            selection_net = self.item_selection_net
        
        selection_dist = selection_net(t) 

        if self.num_experts == 1:
            selection_result = 1
        else:
            # expert selection
            g = torch.distributions.Gumbel(0, 1).sample(selection_dist.size()).to(self.gpu) 
            eps = 1e-10
            selection_dist = selection_dist + eps
            selection_dist = self.sm((selection_dist.log() + g) / self.T) 

            selection_dist = torch.unsqueeze(selection_dist, 1) 
            selection_result = selection_dist.repeat(1, self.teacher_dim, 1) 

        expert_outputs = [experts[i](s).unsqueeze(-1) for i in range(self.num_experts)]
        expert_outputs = torch.cat(expert_outputs, -1) 

        expert_outputs = expert_outputs*selection_result 

        expert_outputs = expert_outputs.sum(2) 

        DE_loss = ((t - expert_outputs)**2).sum(-1).sum() 

        return DE_loss
    

    def get_DETA_loss(self, batch_entity, emb_weights, TA_id, is_user=True):
        """
        Compute DETA loss
        ----------

        Parameters
        ----------
        batch_entity (1D Long Tensor): batch of users/items
        emb_weights (2D FloatTensor): generated embeddings of TAs
        TA_id (int): TA index 
        is_user (Bolean): distilling from user or item embeddings

        Returns
        ----------
        DETA_loss (2D FloatTensor): DETA loss
        """

        if is_user:
            s = self.user_emb(batch_entity)

            experts = self.user_experts_TAs[TA_id]
            selection_net = self.user_selection_net_TA[TA_id]
        else:
            s = self.item_emb(batch_entity)

            experts = self.item_experts_TAs[TA_id]
            selection_net = self.item_selection_net_TA[TA_id]
        
        t = emb_weights
        selection_dist = selection_net(t) 

        if self.num_experts == 1:
            selection_result = 1
        else:
            # expert selection
            g = torch.distributions.Gumbel(0, 1).sample(selection_dist.size()).to(self.gpu) 
            eps = 1e-10
            selection_dist = selection_dist + eps
            selection_dist = self.sm((selection_dist.log() + g) / self.T) 

            selection_dist = torch.unsqueeze(selection_dist, 1) 
            selection_result = selection_dist.repeat(1, self.TAs_dims[TA_id], 1) 

        expert_outputs = [experts[i](s).unsqueeze(-1) for i in range(self.num_experts)]
        expert_outputs = torch.cat(expert_outputs, -1) 

        expert_outputs = expert_outputs*selection_result 

        expert_outputs = expert_outputs.sum(2) 

        DE_loss = ((t - expert_outputs)**2).sum(-1)

        return DE_loss
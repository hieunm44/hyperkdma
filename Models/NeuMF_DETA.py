import numpy as np
import torch
import torch.nn as nn

from Models.NeuMF import NeuMF


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


class NeuMF_DETA(NeuMF):
    def __init__(self, user_count, item_count, teacher_user_MF, teacher_item_MF, teacher_user_MLP, teacher_item_MLP, \
                student_dim, num_layers, TAs_dims, num_experts, gpu):
        """
        Parameters
        ----------
        user_count (int): number of users
        item_count (int): number of items
        teacher_user_MF (2D FloatTensor): teacher user MF embeddings
        teacher_item_MF (2D FloatTensor): teacher item MF embeddings
        teacher_user_MLP (2D FloatTensor): teacher user MLP embeddings
        teacher_item_MLP (2D FloatTensor): teacher item MLP embeddings
        num_layers (int): number of MLP layers
        student_dim (int): dimension of embedding vectors of student model
        TAs_dims (list): list of dimensions of TAs
        num_experts (int): number of DEs
        gpu: gpu device
        """
        
        NeuMF.__init__(self, user_count, item_count, student_dim, num_layers, gpu)

        self.student_dim = self.dim
        self.gpu = gpu

        # teacher embedding
        self.teacher_user_MF = nn.Embedding.from_pretrained(teacher_user_MF)
        self.teacher_item_MF = nn.Embedding.from_pretrained(teacher_item_MF)
        self.teacher_user_MLP = nn.Embedding.from_pretrained(teacher_user_MLP)
        self.teacher_item_MLP = nn.Embedding.from_pretrained(teacher_item_MLP)

        # fix the teacher embeddings
        self.teacher_user_MF.weight.requires_grad = False
        self.teacher_item_MF.weight.requires_grad = False
        self.teacher_user_MLP.weight.requires_grad = False
        self.teacher_item_MLP.weight.requires_grad = False

        # get the teacher dimension
        self.teacher_dim = self.teacher_user_MF.weight.size(1)
        self.TAs_dims = TAs_dims

        # expert configuration
        self.num_experts = num_experts
        expert_dims = [self.student_dim, (self.teacher_dim + self.student_dim)//2, self.teacher_dim]
        TAs_expert_dims = [[self.student_dim, (TA_dim + self.student_dim)//2, TA_dim] for TA_dim in self.TAs_dims]

        # experts for KD from teacher
        self.user_MF_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
        self.item_MF_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
        self.user_MLP_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
        self.item_MLP_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])

        # experts for KD from TAs
        self.TAs_user_MF_experts = nn.ModuleList([nn.ModuleList([Expert(TA_expert_dims).to(self.gpu) for i in range(self.num_experts)]) for TA_expert_dims in TAs_expert_dims])
        self.TAs_item_MF_experts = nn.ModuleList([nn.ModuleList([Expert(TA_expert_dims).to(self.gpu) for i in range(self.num_experts)]) for TA_expert_dims in TAs_expert_dims])
        self.TAs_user_MLP_experts = nn.ModuleList([nn.ModuleList([Expert(TA_expert_dims).to(self.gpu) for i in range(self.num_experts)]) for TA_expert_dims in TAs_expert_dims])
        self.TAs_item_MLP_experts = nn.ModuleList([nn.ModuleList([Expert(TA_expert_dims).to(self.gpu) for i in range(self.num_experts)]) for TA_expert_dims in TAs_expert_dims])

        # selection networks for KD from teacher
        self.user_MF_selection_net = nn.Sequential(
            nn.Linear(self.teacher_dim, num_experts),
            nn.Softmax(dim=1)
        )
        self.item_MF_selection_net = nn.Sequential(
            nn.Linear(self.teacher_dim, num_experts),
            nn.Softmax(dim=1)
        )
        self.user_MLP_selection_net = nn.Sequential(
            nn.Linear(self.teacher_dim, num_experts),
            nn.Softmax(dim=1)
        )
        self.item_MLP_selection_net = nn.Sequential(
            nn.Linear(self.teacher_dim, num_experts),
            nn.Softmax(dim=1)
        )

        # selection networks for KD from TAs
        self.TAs_user_MF_selection_net = nn.ModuleList([nn.Sequential(
            nn.Linear(self.TAs_dims[i], num_experts),
            nn.Softmax(dim=1)
        ) for i in range(len(self.TAs_dims))])
        self.TAs_item_MF_selection_net = nn.ModuleList([nn.Sequential(
            nn.Linear(self.TAs_dims[i], num_experts),
            nn.Softmax(dim=1)
        ) for i in range(len(self.TAs_dims))])
        self.TAs_user_MLP_selection_net = nn.ModuleList([nn.Sequential(
            nn.Linear(self.TAs_dims[i], num_experts),
            nn.Softmax(dim=1)
        ) for i in range(len(self.TAs_dims))])
        self.TAs_item_MLP_selection_net = nn.ModuleList([nn.Sequential(
            nn.Linear(self.TAs_dims[i], num_experts),
            nn.Softmax(dim=1)
        ) for i in range(len(self.TAs_dims))])
        
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
        for i in range(len(self.TAs_dims)):
            print(self.TAs_dims[i], '', end='')
        print()


    def reconstruct(self, X, weights):
        # weight = weights[:, :self.student_dim*self.teacher_dim].reshape(-1, self.teacher_dim, self.student_dim)
        # bias = weights[:, self.student_dim*self.teacher_dim:]
        out = torch.matmul(weights, X.unsqueeze(-1))
        # out = torch.add(out, bias.unsqueeze(-1))
        out = out.squeeze(-1)

        return out


    def get_DE_loss(self, batch_entity, is_MF=True, is_user=True):
        """
        Compute DE loss
        ----------

        Parameters
        ----------
        batch_entity (1-D Long Tensor): batch of users/items
        is_MF (Bolean): distilling for MF or MLP embeddings
        is_user (Bolean): distilling from user or item embeddings

        Returns
        ----------
        DE_loss (float): DE loss
        """

        if is_MF and is_user:
            s = self.user_emb_MF(batch_entity)
            t = self.teacher_user_MF(batch_entity)

            experts = self.user_MF_experts
            selection_net = self.user_MF_selection_net
        elif is_MF and not is_user:
            s = self.item_emb_MF(batch_entity)
            t = self.teacher_item_MF(batch_entity)

            experts = self.item_MF_experts
            selection_net = self.item_MF_selection_net
        elif not is_MF and is_user:
            s = self.user_emb_MLP(batch_entity)
            t = self.teacher_user_MLP(batch_entity)

            experts = self.user_MLP_experts
            selection_net = self.user_MLP_selection_net
        else:
            s = self.item_emb_MLP(batch_entity)
            t = self.teacher_item_MLP(batch_entity)

            experts = self.item_MLP_experts
            selection_net = self.item_MLP_selection_net

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

    
    def get_DETA_loss(self, batch_entity, emb_weights, TA_id, is_MF=True, is_user=True):
        """
        Compute DETA loss
        ----------

        Parameters
        ----------
        batch_entity (1D Long Tensor): batch of users/items
        emb_weights (2D FloatTensor): generated embeddings of TAs
        TA_id (int): TA index 
        is_MF (Bolean): distilling for MF or MLP embeddings
        is_user (Bolean): distilling from user or item embeddings

        Returns
        ----------
        DETA_loss (2D FloatTensor): DETA loss
        """

        if is_MF and is_user:
            s = self.user_emb_MF(batch_entity)

            experts = self.TAs_user_MF_experts[TA_id]
            selection_net = self.TAs_user_MF_selection_net[TA_id]

        elif is_MF and not is_user:
            s = self.item_emb_MF(batch_entity)

            experts = self.TAs_item_MF_experts[TA_id]
            selection_net = self.TAs_item_MF_selection_net[TA_id]

        elif not is_MF and is_user:
            s = self.user_emb_MLP(batch_entity)

            experts = self.TAs_user_MLP_experts[TA_id]
            selection_net = self.TAs_user_MLP_selection_net[TA_id]

        else:
            s = self.item_emb_MLP(batch_entity)

            experts = self.TAs_item_MLP_experts[TA_id]
            selection_net = self.TAs_item_MLP_selection_net[TA_id]

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

        DETA_loss = ((t - expert_outputs)**2).sum(-1)

        return DETA_loss
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


class NeuMF_DE(NeuMF):
    def __init__(self, user_count, item_count, teacher_user_MF, teacher_item_MF, teacher_user_MLP, teacher_item_MLP, student_dim, num_layers, num_experts, gpu):
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
        num_experts (int): number of DEs
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

        # fix the teacher embeddings
        self.teacher_user_MF.weight.requires_grad = False
        self.teacher_item_MF.weight.requires_grad = False
        self.teacher_user_MLP.weight.requires_grad = False
        self.teacher_item_MLP.weight.requires_grad = False

        # get the teacher dimension
        self.teacher_dim = self.teacher_user_MF.weight.size(1)

        # expert configuration
        self.num_experts = num_experts # e.g: 30
        expert_dims = [self.student_dim, (self.teacher_dim + self.student_dim)//2, self.teacher_dim]

        # for self-distillation
        if self.teacher_dim == self.student_dim:
            expert_dims = [self.student_dim, self.student_dim // 2, self.teacher_dim]
        
        # user/item experts
        self.user_MF_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
        self.item_MF_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
        self.user_MLP_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
        self.item_MLP_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])

        # user/item selection networks
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

        print('Teacher dim:', self.teacher_dim, 'Student dim:', self.student_dim)
        print('Expert dims:', expert_dims)
        
        # Gumbel-Softmax temperature
        self.T = 0.
        self.sm = nn.Softmax(dim = 1)


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
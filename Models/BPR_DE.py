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


class BPR_DE(BPR):
    def __init__(self, user_count, item_count, teacher_user_emb, teacher_item_emb, student_dim, num_experts, gpu):
        """
        Parameters
        ----------
        user_count (int): number of users
        item_count (int): number of items
        teacher_user_emb (2D FloatTensor): teacher user embeddings
        teacher_item_emb (2D FloatTensor): teacher item embeddings
        student_dim (int): dimension of embedding vectors of student model
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

        # expert configuration
        self.num_experts = num_experts
        # define dimensions of the expert network
        expert_dims = [self.student_dim, (self.teacher_dim + self.student_dim)//2, self.teacher_dim]
        # for self-distillation
        if self.teacher_dim == self.student_dim:
            expert_dims = [self.student_dim, self.student_dim // 2, self.teacher_dim]
        
        # user/item experts
        self.user_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])
        self.item_experts = nn.ModuleList([Expert(expert_dims) for i in range(self.num_experts)])

        # user/item selection networks
        self.user_selection_net = nn.Sequential(
            nn.Linear(self.teacher_dim, num_experts),
            nn.Softmax(dim=1)
        )
        self.item_selection_net = nn.Sequential(
            nn.Linear(self.teacher_dim, num_experts),
            nn.Softmax(dim=1)
        )

        print('Teacher dim:', self.teacher_dim)
        print('Student dim:', self.student_dim)
        print('Expert dims:', expert_dims)
        
        # Gumbel-Softmax temperature
        self.T = 0.
        self.sm = nn.Softmax(dim = 1)


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
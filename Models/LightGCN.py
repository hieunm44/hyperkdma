import torch
import torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, user_count, item_count, dim, num_layers, Graph, gpu):
        """
        Parameters
        ----------
        user_count (int): number of users
        item_count (int): number of items
        dim (int): dimension of embedding vectors
        num_layers (int): number of layers in LightGCN
        gpu: gpu device
        """

        super(LightGCN, self).__init__()

        self.gpu = gpu
        self.user_count = user_count
        self.item_count = item_count
        self.dim = dim
        self.num_layers = num_layers

        # user/item list
        self.user_list = torch.LongTensor([i for i in range(user_count)])
        self.item_list = torch.LongTensor([i for i in range(item_count)])

        if gpu != None:
            self.user_list = self.user_list.to(gpu)
            self.item_list = self.item_list.to(gpu)

        # user/item embedding
        self.user_emb = nn.Embedding(self.user_count, dim)
        self.item_emb = nn.Embedding(self.item_count, dim)

        self.type = 'graph'

        # initialize user/item embedding
        nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0, std=0.01)

        self.f = nn.Sigmoid()
        self.Graph = Graph


    def computer(self):
        """
        LightGCN propagation method
        ----------

        Returns
        -------
        users, items: Light GCN user/item embeddings
        """   

        users_emb = self.user_emb.weight
        items_emb = self.item_emb.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        g_droped = self.Graph 
        A_split = False

        for layer in range(self.num_layers): 
            if A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb) 
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1) 
        users, items = torch.split(light_out, [self.user_count, self.item_count])

        return users, items


    def forward(self, batch_user, batch_pos_item, batch_neg_item):
        """
        Forward propagation
        ----------

        Parameters
        ----------
        batch_user (1-D Long Tensor): batch of users
        batch_pos_item (1-D Long Tensor): batch of positive items
        batch_neg_item (1-D Long Tensor): batch of negative items

        Returns
        -------
        output: positive score, negative score and regularization loss after forward propagation;
        """

        # get LightGCN embeddings of users / positive items / negative items
        all_users, all_items = self.computer()
        user = all_users[batch_user]
        pos = all_items[batch_pos_item]
        neg = all_items[batch_neg_item]

        # compute postive and negative scores with element-wise product
        pos_score = torch.sum(torch.mul(user, pos), dim=1)
        neg_score = torch.sum(torch.mul(user, neg), dim=1)

        # get embeddings of users / positive items / negative items of first LightGCN layer
        user0 = self.user_emb(batch_user)
        pos0 = self.item_emb(batch_pos_item)
        neg0 = self.item_emb(batch_neg_item)

        # compute regularization loss
        reg_loss = user0.norm(2).pow(2) + pos0.norm(2).pow(2) + neg0.norm(2).pow(2)

        output = (pos_score, neg_score, reg_loss)

        return output

    
    def get_loss(self, output):
        """
        Compute the model loss
        ----------

        Parameters
        ----------
        output (tuple): model output

        Returns
        -------
        loss (float): model loss
        """

        pos_score, neg_score, reg_loss = output[0], output[1], output[2]

        # BPR loss
        BPR_loss = -(pos_score - neg_score).sigmoid().log().sum()
        # total loss
        loss = BPR_loss + reg_loss*1e-4

        return loss + reg_loss*1e-4


    def get_embedding(self):
        """
        Get all LightGCN embeddings of users and items for KD purpose
        ----------

        Returns
        ----------
        users (2-D FloatTensor): all LightGCN user embeddings
        items (2-D FloatTensor): all LightGCN item embeddings
        """

        users, items = self.computer()

        return users, items


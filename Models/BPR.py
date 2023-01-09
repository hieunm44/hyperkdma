import torch
import torch.nn as nn

class BPR(nn.Module):
    def __init__(self, user_count, item_count, dim, gpu):
        """
        Parameters
        ----------
        user_count (int): number of users
        item_count (int): number of items
        dim (int): dimension of embedding vectors
        gpu: gpu device
        """

        super(BPR, self).__init__()

        self.dim = dim
        self.user_count = user_count
        self.item_count = item_count

        # user/item list
        self.user_list = torch.LongTensor([i for i in range(user_count)]) 
        self.item_list = torch.LongTensor([i for i in range(item_count)])

        if gpu != None:
            self.user_list = self.user_list.to(gpu)
            self.item_list = self.item_list.to(gpu)
        
        # user/item embeddings
        self.user_emb = nn.Embedding(self.user_count, dim) 
        self.item_emb = nn.Embedding(self.item_count, dim) 

        # initialize user/item embeddings
        nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0, std=0.01)

        # model type
        self.type = 'MF'


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
        output: positive score and negative score after forward propagation 
        """

        # get embeddings of users / positive items / negative items
        user = self.user_emb(batch_user) 
        pos = self.item_emb(batch_pos_item) 
        neg = self.item_emb(batch_neg_item) 

        # compute postive and negative scores with element-wise product
        pos_score = (user*pos).sum(dim=1, keepdim=True) 
        neg_score = (user*neg).sum(dim=1, keepdim=True) 

        output = (pos_score, neg_score)

        return output


    def get_loss(self, output):
        """
        Compute the model loss
        ----------

        Parameters
        ----------
        output (tuple): positive score and negative score

        Returns
        -------
        loss (float): model loss
        """

        pos_score, neg_score = output[0], output[1]

        # loss of BPR
        loss = -(pos_score - neg_score).sigmoid().log().sum()

        return loss


    def get_embedding(self):
        """
        Get all embeddings of users and items for KD purpose
        ----------

        Returns
        ----------
        users (2D FloatTensor): all user embeddings
        items (2D FloatTensor): all item embeddings
        """

        users = self.user_emb(self.user_list) 
        items = self.item_emb(self.item_list) 

        return users, items
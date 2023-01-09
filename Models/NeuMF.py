
import torch.nn.functional as F
import torch.nn as nn
import torch

class NeuMF(nn.Module):
	def __init__(self, user_count, item_count, dim, num_layers, gpu):
		"""
        Parameters
        ----------
        user_count (int): number of users
        item_count (int): number of items
        dim (int): dimension of embedding vectors
		num_layers: number of MLP layers
        gpu: gpu device
        """

		super(NeuMF, self).__init__()

		self.dim = dim
		self.user_count = user_count
		self.item_count = item_count

		# user/item list
		self.user_list = torch.LongTensor([i for i in range(user_count)])
		self.item_list = torch.LongTensor([i for i in range(item_count)])

		if gpu != None:
			self.user_list = self.user_list.to(gpu)
			self.item_list = self.item_list.to(gpu)

		# user/item MF/MLP embeddings
		self.user_emb_MF = nn.Embedding(self.user_count, dim)
		self.item_emb_MF = nn.Embedding(self.item_count, dim)
		self.user_emb_MLP = nn.Embedding(self.user_count, dim)
		self.item_emb_MLP = nn.Embedding(self.item_count, dim)

		# initialize user/item MF/MLP embeddings
		nn.init.normal_(self.user_emb_MF.weight, mean=0., std= 0.01)
		nn.init.normal_(self.item_emb_MF.weight, mean=0., std= 0.01)
		nn.init.normal_(self.user_emb_MLP.weight, mean=0., std= 0.01)
		nn.init.normal_(self.item_emb_MLP.weight, mean=0., std= 0.01)

		# model type
		self.type = 'network'

		# MLP layer configuration
		MLP_layers = [] 
		layers_shape = [dim * 2]
		for i in range(num_layers):
			layers_shape.append(layers_shape[-1] // 2)
			MLP_layers.append(nn.Linear(layers_shape[-2], layers_shape[-1]))
			MLP_layers.append(nn.ReLU())
		print("MLP Layer Shape:", layers_shape)
		self.MLP_layers = nn.Sequential(* MLP_layers)
		# final Layer
		self.final_layer  = nn.Linear(layers_shape[-1] + dim, 1) 

		self._init_weights()

		# loss function
		self.BCE_loss = nn.BCEWithLogitsLoss(reduction='sum')


	def _init_weights(self):
		"""
		Layer initialization
		----------
		"""

		for m in self.MLP_layers:
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
		nn.init.kaiming_uniform_(self.final_layer.weight, a=1, nonlinearity='relu')

		for m in self.modules():
			if isinstance(m, nn.Linear) and m.bias is not None:
				m.bias.data.zero_()


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

		pos_score = self.forward_no_neg(batch_user, batch_pos_item)	 
		neg_score = self.forward_no_neg(batch_user, batch_neg_item)	 

		output = (pos_score, neg_score)

		return output


	def forward_no_neg(self, batch_user, batch_item):
		"""
		Forward propagation without negative items
		----------

		Parameters
		----------
		batch_user (1-D LongTensor): batch of users
		batch_item (1-D LongTensor): batch of items
		
		Returns
		-------
		output (2-D LongTensor): prediction score
		"""
		
		# get user/item MF embeddings
		u_mf = self.user_emb_MF(batch_user)			
		i_mf = self.item_emb_MF(batch_item)			
		
		# compute MF vector with element-wise product
		mf_vector = (u_mf * i_mf) 

		# get user/item MLP embeddings
		u_mlp = self.user_emb_MLP(batch_user)		
		i_mlp = self.item_emb_MLP(batch_item)		

		# MLP propagation
		mlp_vector = torch.cat([u_mlp, i_mlp], dim=-1) 
		mlp_vector = self.MLP_layers(mlp_vector) 

		# concatenate MF vector and MLP vector
		predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1) 

		output = self.final_layer(predict_vector) 

		return output


	def get_loss(self, output):
		"""
        Compute the model loss
        ----------

        Parameters
        ----------
        output (): model output

        Returns
        -------
        loss (float): model loss
        """

		pos_score, neg_score = output[0], output[1]

		pred = torch.cat([pos_score, neg_score], dim=0) 
		gt = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0) 
		
		return self.BCE_loss(pred, gt)


	def forward_multi_items(self, batch_user, batch_items):
		"""
		Forward when we have multiple items for a user (for evaluation purpose)
		----------

		Parameters
		----------
		batch_user (1D LongTensor): batch of users
		batch_item (1D LongTensor): batch of items

		Returns
		-------
		score (2D FloatTensor): prediction scores
		"""
		batch_user = batch_user.unsqueeze(-1)
		batch_user = torch.cat(batch_items.size(1) * [batch_user], 1)

		score = self.forward_no_neg(batch_user, batch_items).squeeze(-1)
			
		return score
		

	def get_embedding(self):
		"""
        Get all embeddings of users and items for KD purpose
        ----------

        Returns
        ----------
        user_MF (2D FloatTensor): all user MF embeddings
        item_MF (2D FloatTensor): all item MF embeddings
		user_MLP (2D FloatTensor): all user MLP embeddings
        item_MLP (2D FloatTensor): all item MLP embeddings
        """

		user_MF = self.user_emb_MF(self.user_list)
		item_MF = self.item_emb_MF(self.item_list)

		user_MLP = self.user_emb_MLP(self.user_list)
		item_MLP = self.item_emb_MLP(self.item_list)

		return user_MF, item_MF, user_MLP, item_MLP
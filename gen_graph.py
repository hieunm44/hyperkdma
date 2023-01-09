import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


def getSparseGraph(train_mat, user_count, item_count, gpu):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    traindataSize = 0
    
    for uid, interacted_items in train_mat.items():
        items = interacted_items.keys()
        trainUniqueUsers.append(uid)
        trainUser.extend([uid] * len(items))
        trainItem.extend(items)
        traindataSize += len(items)
    
    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)
    
    # (users,items), bipartite graph
    UserItemNet = csr_matrix((np.ones(len(trainUser)), (trainUser, trainItem)), shape=(user_count, item_count))
    users_D = np.array(UserItemNet.sum(axis=1)).squeeze()
    users_D[users_D == 0.] = 1
    items_D = np.array(UserItemNet.sum(axis=0)).squeeze()
    items_D[items_D == 0.] = 1.

    # Graph
    adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = UserItemNet.tolil()
    adj_mat[:user_count, user_count:] = R
    adj_mat[user_count:, :user_count] = R.T
    adj_mat = adj_mat.todok()
    
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum + 1e-16, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    
    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()

    Graph = convert_sp_mat_to_sp_tensor(norm_adj)
    Graph = Graph.coalesce().to(gpu)

    return Graph
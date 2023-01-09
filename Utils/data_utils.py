import pickle


def to_np(x):
	return x.data.cpu().numpy()


def is_visited(base_dict, user_id, item_id):
    if user_id in base_dict and item_id in base_dict[user_id]:
        return True
    else:
        return False

def load_pickle(path, filename):
    with open(path + filename, 'rb') as f:
        obj = pickle.load(f)
    
    return obj


def dict_to_list(base_dict):
    result = []

    for user_id in base_dict:
        for item_id in base_dict[user_id]:
            result.append((user_id, item_id, 1))
    
    return result


def T_annealing(epoch, max_epoch, initial_T, end_T):
	new_T = initial_T * ((end_T / initial_T) ** (epoch / max_epoch))
	return new_T
    

def read_LOO_settings(path, dataset, seed):
    train_mat = load_pickle(path + dataset, '/train_mat_' + str(seed)) 
    train_interactions = dict_to_list(train_mat)
    test_sample = load_pickle(path + dataset, '/test_sample_' + str(seed))
    valid_sample = load_pickle(path + dataset, '/valid_sample_' + str(seed))
    candidates = load_pickle(path + dataset, '/candidates_' + str(seed)) 
    user_count, item_count = load_pickle(path + dataset, '/counts')

    return user_count, item_count, train_mat, train_interactions, valid_sample, test_sample, candidates

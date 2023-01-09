import numpy as np
import pickle

user_item_dict = {}
new_item_dict = {}
# dataset = 'CiteULike'
dataset = 'Foursquare'
seed = 0

with open('Data sets/' + dataset + '/users.dat') as f:
    contents = f.readlines()
    contents = [line.rstrip().split(" ") for line in contents]

new_user = 0
new_item = 0
for user in contents:
    if int(user[0]) >= 5:
        user_item_dict[new_user] = []
        for item in user[1:]:
            if int(item) not in new_item_dict.keys():
                new_item_dict[int(item)] = new_item
                new_item += 1
            user_item_dict[new_user].append(new_item_dict[int(item)])
        new_user += 1

all_items_list = [i for i in range(new_item)]
train_mat = {}
valid_sample = {}
test_sample = {}
candidates = {}

for key, value in user_item_dict.items():
    value = np.random.permutation(value).tolist()
    neg_items = all_items_list.copy()
    for pos_item in value:
        neg_items.remove(pos_item)
    candidates[key] = {item: 1 for item in np.random.choice(neg_items, 499, replace=False)}
    
    valid_item, test_item = np.random.choice(value, 2, replace=False)
    valid_sample[key] = valid_item
    test_sample[key] = test_item
    value.remove(valid_item)
    value.remove(test_item)
    train_mat[key] = {item: 1 for item in value}

with open('Data sets/' + dataset + '/train_mat_' + str(seed), 'wb') as handle:
    pickle.dump(train_mat, handle)

with open('Data sets/' + dataset + '/valid_sample_' + str(seed), 'wb') as handle:
    pickle.dump(valid_sample, handle)

with open('Data sets/' + dataset + '/test_sample_' + str(seed), 'wb') as handle:
    pickle.dump(test_sample, handle)

with open('Data sets/' + dataset + '/candidates_' + str(seed), 'wb') as handle:
    pickle.dump(candidates, handle)

with open('Data sets/' + dataset + '/counts', 'wb') as handle:
    pickle.dump((new_user, new_item), handle)
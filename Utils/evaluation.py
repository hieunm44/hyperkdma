import time
import copy
import math
import numpy as np
import torch

from Utils.data_utils import to_np


def LOO_check(ranking_list, target_item, topk):
	"""
	Calculate H@N and N@N
	----------

	Parameters
	----------
	ranking_list (1D array): model prediction scores
	target_item (int): ground-truth item
	topk (int): topk recommendation

	Returns
	-------
	H@N (float), N@N (float)
	"""

	k = 0
	for item_id in ranking_list:
		if k == topk: return (0., 0.)
		if target_item == item_id: return (1., math.log(2.) / math.log(k + 2.))
		k += 1


def LOO_print_result(epoch, max_epoch, train_loss, eval_results, is_improved=False, train_time=0., test_time=0.):
	"""
	Print Leave-one-out evaluation results
	----------

	Parameters
	----------
	epoch (int): current epoch
	max_epoch (int): maximum training epoch
	train_loss (float): training loss
	eval_results (dict): summary of evaluation results
	is_improved (boolean): is the result improved compared to the last best results
	train_time (float): elapsed time for training
	test_time (float): elapsed time for test
	"""

	if is_improved:
		print('Epoch [{}/{}], Train Loss: {:.4f}, Elapsed: train {:.2f} test {:.2f} *' .format(epoch , max_epoch, train_loss, train_time, test_time))
	else: 
		print('Epoch [{}/{}], Train Loss: {:.4f}, Elapsed: train {:.2f} test {:.2f}' .format(epoch, max_epoch, train_loss, train_time, test_time))


	for mode in ['valid', 'test']:
		for topk in ['05', '10', '20']:
			h = eval_results[mode]['H' + topk]
			n = eval_results[mode]['N' + topk]

			print('{} H@{}: {:.4f}, N@{}: {:.4f}'.format(mode, topk, h, topk, n))
			

def print_final_result(eval_dict):
	"""
	Print final result after the training
	----------

	Parameters
	----------
	eval_dict : dict
	"""

	res = []
	for mode in ['valid', 'test']:
		print(mode)

		r_dict = {'H05':0, 'N05':0, 'H10':0, 'N10':0, 'H20':0, 'N20':0}

		if mode == 'valid':
			key = 'best_result'
		else:
			key = 'final_result'

		for topk in ['05', '10', '20']:
			r_dict['H' + topk] = eval_dict[topk][key]['H' + topk]
			r_dict['N' + topk] = eval_dict[topk][key]['N' + topk]

		print(r_dict)

		res.append(r_dict)
	return res


def latent_factor_evaluate(model, test_dataset):
	"""
	Evaluation for latent factor model (BPR, LightGCN)
	----------

	Parameters
	----------
	model: model
	test_dataset: test dataset 

	Returns
	-------
	eval_results : dict
		summarizes the evaluation results
	"""
	
	metrics = {'H05':[], 'N05':[], 'H10':[], 'N10':[], 'H20':[], 'N20':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}
	
	# extract score 
	if model.type == 'MF':
		user_emb, item_emb = model.get_embedding()
	elif model.type == 'graph':
		user_emb, item_emb = model.computer()

	score_mat = to_np(-torch.matmul(user_emb, item_emb.T)) 
	test_user_list = to_np(test_dataset.user_list) 
	
	for test_user in test_user_list: 

		test_item = [int(test_dataset.test_item[test_user][0])] 
		valid_item = [int(test_dataset.valid_item[test_user][0])] 
		candidates = to_np(test_dataset.candidates[test_user]).tolist()

		total_items = test_item + valid_item + candidates
		score = score_mat[test_user][total_items] 
		
		result = np.argsort(score).flatten().tolist()
		ranking_list = np.array(total_items)[result]

		for mode in ['test', 'valid']:
			if mode == 'test':
				target_item = test_item[0] 
				ranking_list_tmp = np.delete(ranking_list, np.where(ranking_list == valid_item[0]))
			else:
				target_item = valid_item[0]
				ranking_list_tmp = np.delete(ranking_list, np.where(ranking_list == test_item[0]))
		
			for topk in ['05', '10', '20']:
				(h, n) = LOO_check(ranking_list_tmp, target_item, int(topk))
			
				eval_results[mode]['H' + topk].append(h)
				eval_results[mode]['N' + topk].append(n)

	# valid, test
	for mode in ['test', 'valid']:
		for topk in ['05', '10', '20']:
			eval_results[mode]['H' + topk] = round(np.asarray(eval_results[mode]['H' + topk]).mean(), 4)
			eval_results[mode]['N' + topk] = round(np.asarray(eval_results[mode]['N' + topk]).mean(), 4)	

	return eval_results


def net_evaluate(model, gpu, test_dataset):
	"""
	Leave-one-out evaluation for deep model
	----------

	Parameters
	----------
	model: model
	gpu: gpu device
	test_dataset: test dataset
	----------

	Returns
	-------
	eval_results (dict): summary of the evaluation results
	"""
	metrics = {'H05':[], 'N05':[], 'H10':[], 'N10':[], 'H20':[], 'N20':[]}
	eval_results = {'test': copy.deepcopy(metrics), 'valid':copy.deepcopy(metrics)}

	# for each batch
	while True:
		batch_users, is_last_batch = test_dataset.get_next_batch_users() 
		batch_test_items, batch_valid_items, batch_candidates = test_dataset.get_next_batch(batch_users)

		batch_total_items = torch.cat([batch_test_items, batch_valid_items, batch_candidates], -1) 

		batch_users = batch_users.to(gpu)
		batch_total_items = batch_total_items.to(gpu)

		batch_score_mat = model.forward_multi_items(batch_users, batch_total_items)

		batch_score_mat = to_np(-batch_score_mat) 
		batch_total_items = to_np(batch_total_items) 

		# for each test user in a mini-batch
		for idx, test_user in enumerate(batch_users):
			
			total_items = batch_total_items[idx] 
			score = batch_score_mat[idx] 

			result = np.argsort(score).flatten().tolist()
			ranking_list = np.array(total_items)[result]

			for mode in ['test', 'valid']:
				if mode == 'test':
					target_item = total_items[0]
					ranking_list_tmp = np.delete(ranking_list, np.where(ranking_list == total_items[1]))
				else:
					target_item = total_items[1]
					ranking_list_tmp = np.delete(ranking_list, np.where(ranking_list == total_items[0]))
				
				for topk in ['05', '10', '20']:
					(h, n) = LOO_check(ranking_list_tmp, target_item, int(topk))
					eval_results[mode]['H' + topk].append(h)
					eval_results[mode]['N' + topk].append(n)

		if is_last_batch: break


	# valid, test
	for mode in ['test', 'valid']:
		for topk in ['05', '10', '20']:
			eval_results[mode]['H' + topk] = round(np.asarray(eval_results[mode]['H' + topk]).mean(), 4)
			eval_results[mode]['N' + topk] = round(np.asarray(eval_results[mode]['N' + topk]).mean(), 4)

	return eval_results

	
def evaluation(model, gpu, eval_dict, epoch, test_dataset):
	"""
	Parameters
	----------
	model: model
	gpu: gpu device
	eval_dict (dict): for control the training process
	epoch (int): current epoch
	test_dataset: test dataset

	Returns
	-------
	is_improved: is the result improved compared to the last best results
	eval_results: summary of the evaluation results
	toc-tic: elapsed time for evaluation
	"""

	model.eval()
	with torch.no_grad():
		tic = time.time()

		# NeuMF
		if model.type == 'network':
			eval_results = net_evaluate(model, gpu, test_dataset)

		# BPR, LightGCN
		elif model.type == 'MF' or model.type == 'graph':
			eval_results = latent_factor_evaluate(model, test_dataset)

		else:
			assert 'Unknown model type'	

		toc = time.time()
		is_improved = False

		for topk in ['05', '10', '20']:
			if eval_dict['early_stop'] < eval_dict['early_stop_max']:
				if eval_dict[topk]['best_score'] < eval_results['valid']['H' + topk]:
					eval_dict[topk]['best_score'] = eval_results['valid']['H' + topk]
					eval_dict[topk]['best_result'] = eval_results['valid']
					eval_dict[topk]['final_result'] = eval_results['test']

					is_improved = True
					eval_dict['final_epoch'] = epoch

		if not is_improved:
			eval_dict['early_stop'] +=1
		else:
			eval_dict['early_stop'] = 0

		return is_improved, eval_results, toc - tic

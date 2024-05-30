import argparse
import torch
import torch.utils.data as data
import torch.optim as optim

from Models.BPR import BPR
from Models.NeuMF import NeuMF
from Models.LightGCN import LightGCN

from Utils.dataset import implicit_CF_dataset, implicit_CF_dataset_test
from Utils.data_utils import read_LOO_settings

from run import no_KD_run
import gen_graph


def run():
    # gpu setting
	gpu = torch.device('cuda:' + str(opt.gpu))

	# dataset
	user_count, item_count, train_mat, train_interactions, valid_sample, test_sample, candidates = read_LOO_settings(opt.data_path, opt.dataset, opt.seed)

	train_dataset = implicit_CF_dataset(user_count, item_count, train_mat, train_interactions, opt.num_ns)
	test_dataset = implicit_CF_dataset_test(user_count, test_sample, valid_sample, candidates)
	train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
 
	# setup model
	if opt.model == 'BPR':
		model = BPR(user_count, item_count, opt.dim, gpu)
	elif opt.model == 'NeuMF':
		model = NeuMF(user_count, item_count, opt.dim, opt.num_layers, gpu)
	elif opt.model == 'LightGCN':
		Graph = gen_graph.getSparseGraph(train_mat, user_count, item_count, gpu)
		model = LightGCN(user_count, item_count, opt.dim, opt.num_layers, Graph, gpu)
	else:
		assert False
	model = model.to(gpu)

	# training
	optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.reg)
	# no_KD_run(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path=opt.saved_models + opt.dataset + '/' + opt.model + '_' + str(opt.dim) + '_seed_' + str(opt.seed))
	no_KD_run(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path=None)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# training
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('--reg', type=float, default=0.001, help='weight decay')
	parser.add_argument('--batch_size', type=int, default=1024)
	parser.add_argument('--num_ns', type=int, default=1, help='number of negative samples')

	parser.add_argument('--max_epoch', type=int, default=1000)
	parser.add_argument('--early_stop', type=int, default=20, help='number of epochs for early stopping')
	parser.add_argument('--es_epoch', type=int, default=0)
	parser.add_argument('--saved_models', type=str, default='Saved models/')
	parser.add_argument('--gpu', type=int, default=0, help='gpu id')

	# dataset
	parser.add_argument('--data_path', type=str, default='datasets/')
	parser.add_argument('--dataset', type=str, default='CiteULike')
	parser.add_argument('--seed', type=int, default=0, help='dataset seed')
	
	# model
	parser.add_argument('--model', type=str, default='BPR')
	parser.add_argument('--dim', type=int, default=200)
	parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers (for NeuMF and LightGCN)')

	opt = parser.parse_args()
	# print(opt)

	run()

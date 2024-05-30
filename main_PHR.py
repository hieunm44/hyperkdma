import argparse
import torch
import torch.utils.data as data
import torch.optim as optim

from Models.BPR import BPR
from Models.BPR_PHR import BPR_PHR
from Models.NeuMF import NeuMF
from Models.NeuMF_PHR import NeuMF_PHR
from Models.LightGCN import LightGCN
from Models.LightGCN_PHR import LightGCN_PHR

from Utils.dataset import implicit_CF_dataset, implicit_CF_dataset_test
from Utils.data_utils import read_LOO_settings

from run import PHR_run
import gen_graph


def run():
	# gpu settings
	gpu = torch.device('cuda:' + str(opt.gpu))

	# dataset
	user_count, item_count, train_mat, train_interactions, valid_sample, test_sample, candidates = read_LOO_settings(opt.data_path, opt.dataset, opt.seed)

	train_dataset = implicit_CF_dataset(user_count, item_count, train_mat, train_interactions, opt.num_ns)
	test_dataset = implicit_CF_dataset_test(user_count, test_sample, valid_sample, candidates)
	train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

	# read teacher model
	teacher_dim = opt.teacher_dim
	if opt.model == 'BPR':
		teacher_model = BPR(user_count, item_count, teacher_dim, gpu)
	elif opt.model == 'NeuMF':
		num_layers = opt.num_layers
		teacher_model = NeuMF(user_count, item_count, teacher_dim, num_layers, gpu)
	elif opt.model == 'LightGCN':
		num_layers = opt.num_layers
		Graph = gen_graph.getSparseGraph(train_mat, user_count, item_count, gpu)
		teacher_model = LightGCN(user_count, item_count, teacher_dim, num_layers, Graph, gpu)
	else:
		assert False
		
	with torch.no_grad():
		teacher_model_path = opt.saved_models + opt.dataset +"/" + opt.model + '_' + str(teacher_dim) + '_seed_' + str(opt.seed)
		teacher_model = teacher_model.to(gpu)
		teacher_model.load_state_dict(torch.load(teacher_model_path, map_location='cuda:' + str(opt.gpu)))
		if opt.model == 'BPR' or opt.model == 'LightGCN':		
			teacher_user_emb, teacher_item_emb = teacher_model.get_embedding()
		elif opt.model == 'NeuMF':
			teacher_user_MF, teacher_item_MF, teacher_user_MLP, teacher_item_MLP = teacher_model.get_embedding()
		else:
			assert False
		del teacher_model

	# student model
	student_dim = opt.student_dim
	if opt.model == 'BPR':
		student_model = BPR_PHR(user_count, item_count, teacher_user_emb, teacher_item_emb, student_dim, train_mat, gpu)
	elif opt.model == 'NeuMF':
		student_model = NeuMF_PHR(user_count, item_count, teacher_user_MF, teacher_item_MF, teacher_user_MLP, teacher_item_MLP, student_dim, num_layers, train_mat, gpu)
	elif opt.model == 'LightGCN':
		student_model = LightGCN_PHR(user_count, item_count, teacher_user_emb, teacher_item_emb, student_dim, num_layers, Graph, train_mat, gpu)
	else:
		assert False
	student_model = student_model.to(gpu)
	
	# training
	optimizer = optim.Adam(student_model.parameters(), lr=opt.lr, weight_decay=opt.reg)
	# PHR_run(opt, student_model, gpu, optimizer, train_loader, test_dataset, model_save_path=opt.saved_models + opt.dataset + '/' + opt.model + '_PHR_' + str(teacher_dim) + 'to' + str(student_dim) + '_seed_' + str(seed))
	PHR_run(opt, student_model, gpu, optimizer, train_loader, test_dataset, model_save_path=None)


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
	parser.add_argument('--teacher_dim', type=int, default=200)
	parser.add_argument('--student_dim', type=int, default=20)
	parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers (for NeuMF and LightGCN)')

	# PHR
	parser.add_argument('--lmbda_PHR', type=float, default=0.01)

	opt = parser.parse_args()
	# print(opt)

	run()


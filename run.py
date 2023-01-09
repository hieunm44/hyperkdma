import time
from copy import deepcopy
import torch
import numpy as np

from Utils.evaluation import evaluation, LOO_print_result, print_final_result
from Utils.data_utils import T_annealing


def no_KD_run(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path=None):
    """
    Training without KD
    ----------

    Parameters
    ----------
    opt: parse arguments
    model: model
    gpu: gpu device
    optimizer: optimizer
    train_loader: training dataset
    test_dataset: test dataset
    model_save_path (str): path for saving models
    """

    max_epoch, early_stop, es_epoch = opt.max_epoch, opt.early_stop, opt.es_epoch
    template = {'best_score': -999, 'best_result': -1, 'final_result': -1}
    eval_dict = {'05': deepcopy(template), '10': deepcopy(template), '20': deepcopy(template), 'early_stop': 0, 'early_stop_max': early_stop, 'final_epoch': 0}

    print('\nTraining model with dim =', model.dim, '...\n')

    # begin training
    for epoch in range(max_epoch):
        tic1 = time.time()
        train_loader.dataset.negative_sampling()
        epoch_loss = []

        # mini-batch training
        for batch_user, batch_pos_item, batch_neg_item in train_loader:
            batch_user = batch_user.to(gpu)
            batch_pos_item = batch_pos_item.to(gpu)
            batch_neg_item = batch_neg_item.to(gpu)

            # forward propagation
            model.train()
            output = model(batch_user, batch_pos_item, batch_neg_item)

            # batch loss
            batch_loss = model.get_loss(output)
            epoch_loss.append(batch_loss)

            # update model parameters
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # total loss in an epoch
        epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
        toc1 = time.time()

        # evaluation
        is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
        LOO_print_result(epoch, max_epoch, epoch_loss, eval_results, is_improved=is_improved, train_time = toc1-tic1, test_time = elapsed)

        # save model
        if model_save_path != None and is_improved:
            torch.save(model.state_dict(), model_save_path)

        # early stopping
        if eval_dict['early_stop'] >= eval_dict['early_stop_max']:
            break

    print("BEST EPOCH::", eval_dict['final_epoch'])
    print_final_result(eval_dict)


def DE_run(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path=None):
    """
    Training using DE for KD
    ----------

    Parameters
    ----------
    opt: parse arguments
    model: model
    gpu: gpu device
    optimizer: optimizer
    train_loader: training dataset
    test_dataset: test dataset
    model_save_path (str): path for saving models
    """

    max_epoch, early_stop, es_epoch = opt.max_epoch, opt.early_stop, opt.es_epoch
    template = {'best_score':-999, 'best_result':-1, 'final_result':-1}
    eval_dict = {'05': deepcopy(template), '10': deepcopy(template), '20': deepcopy(template), 'early_stop': 0, 'early_stop_max': early_stop, 'final_epoch': 0}

    print('\nTraining model with dim =', model.dim, 'using DE ...\n')

    current_T = opt.end_T * opt.anneal_size

    # begin training
    for epoch in range(max_epoch):
        tic1 = time.time()
        train_loader.dataset.negative_sampling()
        epoch_loss = []

        model.T = current_T
        
        # mini-batch training
        for batch_user, batch_pos_item, batch_neg_item in train_loader:			
            batch_user = batch_user.to(gpu)
            batch_pos_item = batch_pos_item.to(gpu)
            batch_neg_item = batch_neg_item.to(gpu)
            
            # forward propagation
            model.train()
            output = model(batch_user, batch_pos_item, batch_neg_item)
            base_loss = model.get_loss(output)

            # DE loss
            if opt.model == 'BPR' or opt.model == 'LightGCN':
                DE_loss_user = model.get_DE_loss(batch_user.unique(), is_user=True)
                DE_loss_pos = model.get_DE_loss(batch_pos_item.unique(), is_user=False)
                DE_loss_neg = model.get_DE_loss(batch_neg_item.unique(), is_user=False)
                DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg)*0.5
            elif opt.model == 'NeuMF':
                DE_loss_user_MF = model.get_DE_loss(batch_user.unique(), is_MF=True, is_user=True)
                DE_loss_pos_MF = model.get_DE_loss(batch_pos_item.unique(), is_MF=True, is_user=False)
                DE_loss_neg_MF = model.get_DE_loss(batch_neg_item.unique(), is_MF=True, is_user=False)

                # DE_loss_user_MLP = model.get_DE_loss(batch_user.unique(), is_MF=False, is_user=True)
                # DE_loss_pos_MLP = model.get_DE_loss(batch_pos_item.unique(), is_MF=False, is_user=False)
                # DE_loss_neg_MLP = model.get_DE_loss(batch_neg_item.unique(), is_MF=False, is_user=False)

                # DE_loss = DE_loss_user_MF + DE_loss_user_MLP + (DE_loss_pos_MF + DE_loss_neg_MF + DE_loss_pos_MLP + DE_loss_neg_MLP) * 0.5
                DE_loss = DE_loss_user_MF + (DE_loss_pos_MF + DE_loss_neg_MF)*0.5
                # DE_loss = DE_loss_user_MLP + (DE_loss_pos_MLP + DE_loss_neg_MLP) * 0.5
            
            # batch loss
            batch_loss = base_loss + DE_loss*opt.lmbda_DE

            epoch_loss.append(batch_loss)
            
            # update model parameters
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # compute total loss in an epoch 
        epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
        toc1 = time.time()
        
        # evaluation
        is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
        LOO_print_result(epoch, max_epoch, epoch_loss, eval_results, is_improved=is_improved, train_time = toc1-tic1, test_time = elapsed)
        
        # save model
        if model_save_path != None and is_improved:
            torch.save(model.state_dict(), model_save_path)

        # early stopping
        if (eval_dict['early_stop'] >= eval_dict['early_stop_max']):
            break

        # annealing
        current_T = T_annealing(epoch, max_epoch, opt.end_T * opt.anneal_size, opt.end_T)
        if current_T < opt.end_T:
            current_T = opt.end_T


    print("BEST EPOCH::", eval_dict['final_epoch'])
    print_final_result(eval_dict)


def PHR_run(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path=None):    
    """
    Training using PHR for KD
    ----------

    Parameters
    ----------
    opt: parse arguments
    model: model
    gpu: gpu device
    optimizer: optimizer
    train_loader: training dataset
    test_dataset: test dataset
    model_save_path (str): path for saving models
    """

    max_epoch, early_stop, es_epoch = opt.max_epoch, opt.early_stop, opt.es_epoch

    template = {'best_score':-999, 'best_result':-1, 'final_result':-1}
    eval_dict = {'05': deepcopy(template), '10': deepcopy(template), '20': deepcopy(template), 'early_stop': 0, 'early_stop_max': early_stop, 'final_epoch': 0}

    print('\nTraining model with dim =', model.dim, 'using PHR ...\n')

    # begin training
    for epoch in range(max_epoch):
        tic1 = time.time()
        train_loader.dataset.negative_sampling()
        epoch_loss = []
        
        for batch_user, batch_pos_item, batch_neg_item in train_loader:			
            # Convert numpy arrays to torch tensors
            batch_user = batch_user.to(gpu)
            batch_pos_item = batch_pos_item.to(gpu)
            batch_neg_item = batch_neg_item.to(gpu)

            # Forward Pass
            model.train()
            output = model(batch_user, batch_pos_item, batch_neg_item)
            base_loss = model.get_loss(output)

            batch_user = batch_user.unique()
            batch_pos_item = batch_pos_item.unique()
            batch_neg_item = batch_neg_item.unique()
            
            if opt.model == 'BPR' or opt.model == 'LightGCN':
                PHR_loss_user = model.get_PHR_loss(batch_user, torch.cat((batch_pos_item, batch_neg_item), dim=0), is_user=True)
                PHR_loss_pos = model.get_PHR_loss(batch_user, batch_pos_item, is_user=False)
                PHR_loss_neg = model.get_PHR_loss(batch_user, batch_neg_item, is_user=False)
                PHR_loss = PHR_loss_user + (PHR_loss_pos+PHR_loss_neg)*0.5
                # PHR_loss = PHR_loss_user
            elif opt.model == 'NeuMF':
                PHR_loss_user_MF = model.get_PHR_loss(batch_user, torch.cat((batch_pos_item, batch_neg_item), dim=0), is_MF=True, is_user=True)
                PHR_loss_pos_MF = model.get_PHR_loss(batch_user, batch_pos_item, is_MF=True, is_user=False)
                PHR_loss_neg_MF = model.get_PHR_loss(batch_user, batch_neg_item, is_MF=True, is_user=False)

                # PHR_loss_user_MLP = model.get_PHR_loss(batch_user.unique(), is_MF=False, is_user=True)
                # PHR_loss_pos_MLP = model.get_PHR_loss(batch_pos_item.unique(), is_MF=False, is_user=False)
                # PHR_loss_neg_MLP = model.get_PHR_loss(batch_neg_item.unique(), is_MF=False, is_user=False)

                # PHR_loss = PHR_loss_user_MF + PHR_loss_user_MLP + (PHR_loss_pos_MF + PHR_loss_neg_MF + PHR_loss_pos_MLP + PHR_loss_neg_MLP) * 0.5
                PHR_loss = PHR_loss_user_MF + (PHR_loss_pos_MF + PHR_loss_neg_MF) * 0.5
                # PHR_loss = PHR_loss_user_MLP + (PHR_loss_pos_MLP + PHR_loss_neg_MLP) * 0.5
            
            batch_loss = base_loss + PHR_loss*opt.lmbda_PHR

            epoch_loss.append(batch_loss)
            
            # Backward and optimize
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        
        epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
        toc1 = time.time()
        
        # evaluation
        is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
        LOO_print_result(epoch, max_epoch, epoch_loss, eval_results, is_improved=is_improved, train_time = toc1-tic1, test_time = elapsed)
            
        if model_save_path != None and is_improved:
            torch.save(model.state_dict(), model_save_path)

        if (eval_dict['early_stop'] >= eval_dict['early_stop_max']):
            break

    print("BEST EPOCH::", eval_dict['final_epoch'])
    print_final_result(eval_dict)


def DETA_run(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path=None):
    max_epoch, early_stop, es_epoch = opt.max_epoch, opt.early_stop, opt.es_epoch

    template = {'best_score':-999, 'best_result':-1, 'final_result':-1}
    eval_dict = {'05': deepcopy(template), '10':deepcopy(template), '20':deepcopy(template), 'early_stop':0,  'early_stop_max':early_stop, 'final_epoch':0}

    current_T = opt.end_T * opt.anneal_size

    # begin training
    for epoch in range(max_epoch):
        
        tic1 = time.time()
        train_loader.dataset.negative_sampling()
        epoch_loss = []

        model.T = current_T
        
        for batch_user, batch_pos_item, batch_neg_item in train_loader:			
            # Convert numpy arrays to torch tensors
            batch_user = batch_user.to(gpu)
            batch_pos_item = batch_pos_item.to(gpu)
            batch_neg_item = batch_neg_item.to(gpu)
            
            # Forward Pass
            model.train()
            output = model(batch_user, batch_pos_item, batch_neg_item)
            base_loss = model.get_loss(output)
            
            # DE loss from teacher
            if opt.model == 'BPR' or opt.model == 'LightGCN':
                DE_loss_user = model.get_DE_loss(batch_user.unique(), is_user=True)
                DE_loss_pos = model.get_DE_loss(batch_pos_item.unique(), is_user=False)
                DE_loss_neg = model.get_DE_loss(batch_neg_item.unique(), is_user=False)
                DE_loss = DE_loss_user + (DE_loss_pos + DE_loss_neg)*0.5
            elif opt.model == 'NeuMF':
                DE_loss_user_MF = model.get_DE_loss(batch_user.unique(), is_MF=True, is_user=True)
                DE_loss_pos_MF = model.get_DE_loss(batch_pos_item.unique(), is_MF=True, is_user=False)
                DE_loss_neg_MF = model.get_DE_loss(batch_neg_item.unique(), is_MF=True, is_user=False)

                # DE_loss_user_MLP = model.get_DE_loss(batch_user.unique(), is_MF=False, is_user=True)
                # DE_loss_pos_MLP = model.get_DE_loss(batch_pos_item.unique(), is_MF=False, is_user=False)
                # DE_loss_neg_MLP = model.get_DE_loss(batch_neg_item.unique(), is_MF=False, is_user=False)

                # DE_loss = DE_loss_user_MF + DE_loss_user_MLP + (DE_loss_pos_MF + DE_loss_neg_MF + DE_loss_pos_MLP + DE_loss_neg_MLP) * 0.5
                DE_loss = DE_loss_user_MF + (DE_loss_pos_MF + DE_loss_neg_MF)*0.5
                # DE_loss = DE_loss_user_MLP + (DE_loss_pos_MLP + DE_loss_neg_MLP) * 0.5

            batch_loss = base_loss + DE_loss*opt.lmbda_DE

            # DE loss from TAs
            if opt.model == 'BPR' or opt.model == 'LightGCN':
                user_weights = model.hnet(model.teacher_user_emb(batch_user.unique()))
                pos_weights = model.hnet(model.teacher_item_emb(batch_pos_item.unique()))
                neg_weights = model.hnet(model.teacher_item_emb(batch_neg_item.unique()))

                user_distribution = model.gnet(model.teacher_user_emb(batch_user.unique()))
                pos_distribution = model.gnet(model.teacher_item_emb(batch_pos_item.unique()))
                neg_distribution = model.gnet(model.teacher_item_emb(batch_neg_item.unique()))

                TAs_dims = model.TAs_dims.copy()
                TAs_dims.insert(0, 0)
            
                for i in range(len(model.TAs_dims)):
                    user_weights_TA = user_weights[:, model.teacher_dim*np.sum(TAs_dims[0:i+1]): model.teacher_dim*np.sum(TAs_dims[0:i+2])].reshape(-1, TAs_dims[i+1], model.teacher_dim)
                    pos_weights_TA = pos_weights[:, model.teacher_dim*np.sum(TAs_dims[0:i+1]): model.teacher_dim*np.sum(TAs_dims[0:i+2])].reshape(-1, TAs_dims[i+1], model.teacher_dim)
                    neg_weights_TA = neg_weights[:, model.teacher_dim*np.sum(TAs_dims[0:i+1]): model.teacher_dim*np.sum(TAs_dims[0:i+2])].reshape(-1, TAs_dims[i+1], model.teacher_dim)

                    reconstructed_user_emb = model.reconstruct(model.teacher_user_emb(batch_user.unique()), user_weights_TA)
                    reconstructed_pos_emb = model.reconstruct(model.teacher_item_emb(batch_pos_item.unique()), pos_weights_TA)
                    reconstructed_neg_emb = model.reconstruct(model.teacher_item_emb(batch_neg_item.unique()), neg_weights_TA)   
                    
                    DE_loss_user_TAS = (user_distribution[:, i]*model.get_DETA_loss(batch_user.unique(), reconstructed_user_emb, i, is_user=True)).sum()
                    DE_loss_pos_TAS = (pos_distribution[:, i]*model.get_DETA_loss(batch_pos_item.unique(), reconstructed_pos_emb, i, is_user=False)).sum()
                    DE_loss_neg_TAS = (neg_distribution[:, i]*model.get_DETA_loss(batch_neg_item.unique(), reconstructed_neg_emb, i, is_user=False)).sum()
                    DE_loss_TAS = DE_loss_user_TAS + 0.5*(DE_loss_pos_TAS + DE_loss_neg_TAS)         
            
            elif opt.model == 'NeuMF':
                user_weights = model.hnet(model.teacher_user_MF(batch_user.unique()))
                pos_weights = model.hnet(model.teacher_item_MF(batch_pos_item.unique()))
                neg_weights = model.hnet(model.teacher_item_MF(batch_neg_item.unique()))

                user_distribution = model.gnet(model.teacher_user_MF(batch_user.unique()))
                pos_distribution = model.gnet(model.teacher_item_MF(batch_pos_item.unique()))
                neg_distribution = model.gnet(model.teacher_item_MF(batch_neg_item.unique()))

                TAs_dims = model.TAs_dims.copy()
                TAs_dims.insert(0, 0)
            
                for i in range(len(model.TAs_dims)):
                    user_weights_TA = user_weights[:, model.teacher_dim*np.sum(TAs_dims[0:i+1]): model.teacher_dim*np.sum(TAs_dims[0:i+2])].reshape(-1, TAs_dims[i+1], model.teacher_dim)
                    pos_weights_TA = pos_weights[:, model.teacher_dim*np.sum(TAs_dims[0:i+1]): model.teacher_dim*np.sum(TAs_dims[0:i+2])].reshape(-1, TAs_dims[i+1], model.teacher_dim)
                    neg_weights_TA = neg_weights[:, model.teacher_dim*np.sum(TAs_dims[0:i+1]): model.teacher_dim*np.sum(TAs_dims[0:i+2])].reshape(-1, TAs_dims[i+1], model.teacher_dim)

                    reconstructed_user_emb = model.reconstruct(model.teacher_user_MF(batch_user.unique()), user_weights_TA)
                    reconstructed_pos_emb = model.reconstruct(model.teacher_item_MF(batch_pos_item.unique()), pos_weights_TA)
                    reconstructed_neg_emb = model.reconstruct(model.teacher_item_MF(batch_neg_item.unique()), neg_weights_TA)   

                    DE_loss_user_TAS_MF = (user_distribution[:, i]*model.get_DETA_loss(batch_user.unique(), reconstructed_user_emb, i, is_MF=True, is_user=True)).sum()
                    DE_loss_pos_TAS_MF = (pos_distribution[:, i]*model.get_DETA_loss(batch_pos_item.unique(), reconstructed_pos_emb, i, is_MF=True, is_user=False)).sum()
                    DE_loss_neg_TAS_MF = (neg_distribution[:, i]*model.get_DETA_loss(batch_neg_item.unique(), reconstructed_neg_emb, i, is_MF=True, is_user=False)).sum()
                    DE_loss_TAS = DE_loss_user_TAS_MF + 0.5*(DE_loss_pos_TAS_MF + DE_loss_neg_TAS_MF)

                batch_loss += DE_loss_TAS*opt.lmbda_DE_TAS

            epoch_loss.append(batch_loss)
            
            # Backward and optimize
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            
        epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
        toc1 = time.time()
        
        # evaluation
        is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
        LOO_print_result(epoch, max_epoch, epoch_loss, eval_results, is_improved=is_improved, train_time = toc1-tic1, test_time = elapsed)
            
        if model_save_path != None and is_improved:
            torch.save(model.state_dict(), model_save_path)

        if (eval_dict['early_stop'] >= eval_dict['early_stop_max']):
            break

        # annealing
        current_T = T_annealing(epoch, max_epoch, opt.end_T * opt.anneal_size, opt.end_T)
        if current_T < opt.end_T:
            current_T = opt.end_T

    print("BEST EPOCH::", eval_dict['final_epoch'])
    print_final_result(eval_dict)


def PHRTA_run(opt, model, gpu, optimizer, train_loader, test_dataset, model_save_path=None):
    max_epoch, early_stop, es_epoch = opt.max_epoch, opt.early_stop, opt.es_epoch

    template = {'best_score':-999, 'best_result':-1, 'final_result':-1}
    eval_dict = {'05': deepcopy(template), '10':deepcopy(template), '20':deepcopy(template), 'early_stop':0,  'early_stop_max':early_stop, 'final_epoch':0}

    # begin training
    for epoch in range(max_epoch):
        tic1 = time.time()
        train_loader.dataset.negative_sampling()
        epoch_loss = []
        
        for batch_user, batch_pos_item, batch_neg_item in train_loader:			
            # Convert numpy arrays to torch tensors
            batch_user = batch_user.to(gpu)
            batch_pos_item = batch_pos_item.to(gpu)
            batch_neg_item = batch_neg_item.to(gpu)
            
            # Forward Pass
            model.train()
            output = model(batch_user, batch_pos_item, batch_neg_item)
            base_loss = model.get_loss(output)
            
            # PHR loss from teacher
            if opt.model == 'BPR' or opt.model == 'LightGCN':
                PHR_loss_user = model.get_PHR_loss(batch_user, torch.cat((batch_pos_item, batch_neg_item), dim=0), is_user=True)
                PHR_loss_pos = model.get_PHR_loss(batch_user, batch_pos_item, is_user=False)
                PHR_loss_neg = model.get_PHR_loss(batch_user, batch_neg_item, is_user=False)
                PHR_loss = PHR_loss_user + (PHR_loss_pos+PHR_loss_neg)*0.5
                # PHR_loss = PHR_loss_user
            elif opt.model == 'NeuMF':
                PHR_loss_user_MF = model.get_PHR_loss(batch_user, torch.cat((batch_pos_item, batch_neg_item), dim=0), is_MF=True, is_user=True)
                PHR_loss_pos_MF = model.get_PHR_loss(batch_user, batch_pos_item, is_MF=True, is_user=False)
                PHR_loss_neg_MF = model.get_PHR_loss(batch_user, batch_neg_item, is_MF=True, is_user=False)

                # PHR_loss_user_MLP = model.get_PHR_loss(batch_user.unique(), is_MF=False, is_user=True)
                # PHR_loss_pos_MLP = model.get_PHR_loss(batch_pos_item.unique(), is_MF=False, is_user=False)
                # PHR_loss_neg_MLP = model.get_PHR_loss(batch_neg_item.unique(), is_MF=False, is_user=False)

                # PHR_loss = PHR_loss_user_MF + PHR_loss_user_MLP + (PHR_loss_pos_MF + PHR_loss_neg_MF + PHR_loss_pos_MLP + PHR_loss_neg_MLP) * 0.5
                PHR_loss = PHR_loss_user_MF + (PHR_loss_pos_MF + PHR_loss_neg_MF) * 0.5
                # PHR_loss = PHR_loss_user_MLP + (PHR_loss_pos_MLP + PHR_loss_neg_MLP) * 0.5

            batch_loss = base_loss + PHR_loss*opt.lmbda_PHR

            # PHR loss from TAs
            if opt.model == 'BPR' or opt.model == 'LightGCN':
                user_weights = model.hnet(model.teacher_user_emb(batch_user.unique()))
                pos_weights = model.hnet(model.teacher_item_emb(batch_pos_item.unique()))
                neg_weights = model.hnet(model.teacher_item_emb(batch_neg_item.unique()))

                user_distribution = model.gnet(model.teacher_user_emb(batch_user.unique()))
                pos_distribution = model.gnet(model.teacher_item_emb(batch_pos_item.unique()))
                neg_distribution = model.gnet(model.teacher_item_emb(batch_neg_item.unique()))

                TAs_dims = model.TAs_dims.copy()
                TAs_dims.insert(0, 0)
            
                for i in range(len(model.TAs_dims)):
                    user_weights_TA = user_weights[:, model.teacher_dim*np.sum(TAs_dims[0:i+1]): model.teacher_dim*np.sum(TAs_dims[0:i+2])].reshape(-1, TAs_dims[i+1], model.teacher_dim)
                    pos_weights_TA = pos_weights[:, model.teacher_dim*np.sum(TAs_dims[0:i+1]): model.teacher_dim*np.sum(TAs_dims[0:i+2])].reshape(-1, TAs_dims[i+1], model.teacher_dim)
                    neg_weights_TA = neg_weights[:, model.teacher_dim*np.sum(TAs_dims[0:i+1]): model.teacher_dim*np.sum(TAs_dims[0:i+2])].reshape(-1, TAs_dims[i+1], model.teacher_dim)

                    reconstructed_user_emb = model.reconstruct(model.teacher_user_emb(batch_user.unique()), user_weights_TA)
                    reconstructed_pos_emb = model.reconstruct(model.teacher_item_emb(batch_pos_item.unique()), pos_weights_TA)
                    reconstructed_neg_emb = model.reconstruct(model.teacher_item_emb(batch_neg_item.unique()), neg_weights_TA)

                    PHR_loss_user_TAS = (user_distribution[:, i]*model.get_PHRTA_loss(batch_user.unique(), torch.cat((batch_pos_item.unique(), batch_neg_item.unique()), dim=0), reconstructed_user_emb, i, is_user=True)).sum()
                    PHR_loss_pos_TAS = (pos_distribution[:, i]*model.get_PHRTA_loss(batch_user.unique(), batch_pos_item.unique(), reconstructed_pos_emb, i,  is_user=False)).sum()
                    PHR_loss_neg_TAS = (neg_distribution[:, i]*model.get_PHRTA_loss(batch_user.unique(), batch_neg_item.unique(), reconstructed_neg_emb, i,  is_user=False)).sum()
                    PHR_loss_TAS = PHR_loss_user_TAS + 0.5*(PHR_loss_pos_TAS + PHR_loss_neg_TAS)

                    batch_loss += PHR_loss_TAS*opt.lmbda_PHR_TAS
            
            elif opt.model == 'NeuMF':
                user_weights = model.hnet(model.teacher_user_MF(batch_user.unique()))
                pos_weights = model.hnet(model.teacher_item_MF(batch_pos_item.unique()))
                neg_weights = model.hnet(model.teacher_item_MF(batch_neg_item.unique()))

                user_distribution = model.gnet(model.teacher_user_MF(batch_user.unique()))
                pos_distribution = model.gnet(model.teacher_item_MF(batch_pos_item.unique()))
                neg_distribution = model.gnet(model.teacher_item_MF(batch_neg_item.unique()))

                TAs_dims = model.TAs_dims.copy()
                TAs_dims.insert(0, 0)
            
                for i in range(len(model.TAs_dims)):
                    user_weights_TA = user_weights[:, model.teacher_dim*np.sum(TAs_dims[0:i+1]): model.teacher_dim*np.sum(TAs_dims[0:i+2])].reshape(-1, TAs_dims[i+1], model.teacher_dim)
                    pos_weights_TA = pos_weights[:, model.teacher_dim*np.sum(TAs_dims[0:i+1]): model.teacher_dim*np.sum(TAs_dims[0:i+2])].reshape(-1, TAs_dims[i+1], model.teacher_dim)
                    neg_weights_TA = neg_weights[:, model.teacher_dim*np.sum(TAs_dims[0:i+1]): model.teacher_dim*np.sum(TAs_dims[0:i+2])].reshape(-1, TAs_dims[i+1], model.teacher_dim)

                    reconstructed_user_emb = model.reconstruct(model.teacher_user_MF(batch_user.unique()), user_weights_TA)
                    reconstructed_pos_emb = model.reconstruct(model.teacher_item_MF(batch_pos_item.unique()), pos_weights_TA)
                    reconstructed_neg_emb = model.reconstruct(model.teacher_item_MF(batch_neg_item.unique()), neg_weights_TA)

                    PHR_loss_user_TAS_MF = (user_distribution[:, i]*model.get_PHRTA_loss(batch_user.unique(), torch.cat((batch_pos_item.unique(), batch_neg_item.unique()), dim=0), reconstructed_user_emb, i, is_user=True)).sum()
                    PHR_loss_pos_TAS_MF = (pos_distribution[:, i]*model.get_PHRTA_loss(batch_user.unique(), batch_pos_item.unique(), reconstructed_pos_emb, i,  is_user=False)).sum()
                    PHR_loss_neg_TAS_MF = (neg_distribution[:, i]*model.get_PHRTA_loss(batch_user.unique(), batch_neg_item.unique(), reconstructed_neg_emb, i,  is_user=False)).sum()
                    PHR_loss_TAS = PHR_loss_user_TAS_MF + 0.5*(PHR_loss_pos_TAS_MF + PHR_loss_neg_TAS_MF)

                    batch_loss += PHR_loss_TAS*opt.lmbda_PHR_TAS

            epoch_loss.append(batch_loss)
            
            # Backward and optimize
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            
        epoch_loss = float(torch.mean(torch.stack(epoch_loss)))
        toc1 = time.time()
        
        # evaluation
        is_improved, eval_results, elapsed = evaluation(model, gpu, eval_dict, epoch, test_dataset)
        LOO_print_result(epoch, max_epoch, epoch_loss, eval_results, is_improved=is_improved, train_time = toc1-tic1, test_time = elapsed)
            
        if model_save_path != None and is_improved:
            torch.save(model.state_dict(), model_save_path)

        if (eval_dict['early_stop'] >= eval_dict['early_stop_max']):
            break

    print("BEST EPOCH::", eval_dict['final_epoch'])
    print_final_result(eval_dict)
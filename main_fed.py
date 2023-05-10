#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import os.path
import wget
from zipfile import ZipFile
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import pickle
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import vgg16, CNNCifar, MobileNetV2
from models.Fed import FedAvg
from models.Test import test_img
from utils.util import setup_seed, exp_details
from utils.dataset_reader import TinyImageNetDataset
from datetime import datetime
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    setup_seed(args.seed)
    exp_details(args)

    # log
    current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
    TAG = 'exp/fed/{}_{}_{}_C{}_iid{}_{}_user{}_{}'.format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                                                           args.alpha, args.num_users, current_time)
    # TAG = f'alpha_{alpha}/data_distribution'
    logdir = f'runs/{TAG}' if not args.debug else f'runs2/{TAG}'
    writer = SummaryWriter(logdir)

    # load dataset and split users
    # test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    if args.dataset == 'imagenet':
        TINY_IMAGENET_ROOT = 'data/tiny-imagenet-200/'
        if os.path.exists('tiny-imagenet-200.zip') == False:
            print('\nDownload dataset\n')
            url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
            tiny_imgdataset = wget.download(url, out = os.getcwd())
            with ZipFile('tiny-imagenet-200.zip', 'r') as zip_ref:
                zip_ref.extractall('data/')
        else:
            print('\nDataset is already downloaded\n')


        dataset_train = datasets.ImageFolder(
            os.path.join('data/', 'tiny-imagenet-200', 'train'),
            transform=transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )
        )
        dataset_test = TinyImageNetDataset(
            img_path=os.path.join('data/', 'tiny-imagenet-200', 'val', 'images'), 
            gt_path=os.path.join('data/', 'tiny-imagenet-200', 'val', 'val_annotations.txt'),
            class_to_idx=dataset_train.class_to_idx.copy(),
            transform=transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            )
        )
        
        data_dist = []
        x_client = []
        if args.iid:
            print('start separate dataset for iid')
            dict_users = cifar_iid(dataset_train, args.num_users)
            print('end')
        else:
            print('start separate dataset for non-iid')
            dict_users, _ = cifar_noniid(dataset_train, args.num_users, args.alpha)

            for k, v in dict_users.items():
                data_dist.append(len(np.array(dataset_train.targets)[v]))
                x_client.append(f'client{k}')
    
                writer.add_histogram(f'user_{k}/data_distribution',
                                    np.array(dataset_train.targets)[v])
                writer.add_histogram(f'all_user/data_distribution',
                                    np.array(dataset_train.targets)[v], global_step=k)
            
            plt.title("data distribution model")
            plt.bar(x_client, data_dist, color ='maroon', width = 0.3)
            plt.savefig(f"imgs/data_dist_num_users{args.num_users}_iid_{args.iid}_epochs_{args.epochs}.png") 
            
            print('end')
        
        
        
    else:
        exit('Error: unrecognized dataset')


    # build model
    if args.model == 'mobilenet' and args.dataset == 'imagenet':
        # net_glob = MobileNetV2().to(args.device)
        net_glob = models.mobilenet_v3_large(pretrained=True, classes_num=200, input_size=224, width_multiplier=1).to(args.device)
    else:
        exit('Error: unrecognized model')
    # print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    test_best_acc = 0.0
    
    test_loss_ar = []
    test_loss_peer_batch = []
    tmp = 0
    test_acc_graph = []
    train_local_loss = {}
    

    for iter in range(1, args.epochs + 1):
        print(f'\nGlobal epoch {iter}\n')
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            print(f'\nclient {idx}\n')
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(w)
            loss_locals.append(loss)
            if idx in train_local_loss.keys():
                train_local_loss[idx].append(loss)
            else:
                train_local_loss[idx] = []
                train_local_loss[idx].append(loss)
            # writer.add_scalar('train loss', {f'client_{idx}': loss, iter)
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('==============================')
        print('Round {:3d}, Train loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        writer.add_scalar('train_loss', loss_avg, iter)
        test_acc, test_loss, tmp = test_img(net_glob, dataset_test, args)
        test_loss_peer_batch.append(tmp)
        test_loss_ar.append(test_loss)
        test_acc_graph.append(test_acc)
        
        writer.add_scalar('test_loss', test_loss, iter)
        writer.add_scalar('test_acc', test_acc, iter)
        with open(f'save/model_{iter}.pkl', 'wb') as fin:
            pickle.dump(net_glob, fin)
        print('==============================')
        save_info = {
            "model": net_glob.state_dict(),
            "epoch": iter
        }
        # save model weights
        if (iter+1) % 500 == 0:
            save_path = f'./save2/{TAG}_{iter+1}es' if args.debug else f'./save/{TAG}_{iter+1}es'
            torch.save(save_info, save_path)
        if iter > 100 and test_acc > test_best_acc:
            test_best_acc = test_acc
            save_path = f'./save2/{TAG}_bst' if args.debug else f'./save/{TAG}_bst'
            torch.save(save_info, save_path)
        
        
    with open('save/model_lust.pkl', 'wb') as fin:
        pickle.dump(net_glob, fin)
    
    # plot loss curve

    
    plt.title("global model")
    plt.plot(range(1, len(loss_train)+1, 1), loss_train, label='train loss')
    plt.plot(range(1, len(loss_train)+1, 1), test_loss_ar, label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(fontsize=12)
    plt.savefig("imgs/global model {}_{}_{}_C_{}_iid_{}.png".format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    plt.close()
    
    
    k = 0
    print(train_local_loss)
    for i in train_local_loss.keys():
        plt.plot(range(1, len(train_local_loss[i]) + 1, 1), train_local_loss[i], label=f'client{i}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("imgs/Train loss by each client {}_{}_{}_C{}_iid{}.png".format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    plt.close()
    
    
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('imgs/fed_numUsers_{}_{}_{}_{}_C{}_iid{}.png'.format(args.num_users, args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    writer.close()

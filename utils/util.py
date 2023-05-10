import torch.backends.cudnn
import torch.cuda
import numpy as np
import random
import torch.nn as nn


def setup_seed(seed):
    torch.manual_seed(seed+1)
    torch.cuda.manual_seed_all(seed+123)
    np.random.seed(seed+1234)
    random.seed(seed+12345)
    torch.backends.cudnn.deterministic = True


def add_scalar(writer, user_num, test_result, epoch):
    test_loss, test_acc, user_loss, user_acc = test_result
    writer.add_scalar(f'user_{user_num}/global/test_loss', test_loss, epoch)
    writer.add_scalar(f'user_{user_num}/global/test_acc', test_acc, epoch)
    writer.add_scalar(f'user_{user_num}/local/test_loss', user_loss, epoch)
    writer.add_scalar(f'user_{user_num}/local/test_acc', user_acc, epoch)


def exp_details(args):
    print('\n---------------Experimental details:-------------\n')
    print(f'\tModel           : {args.model}')
    print(f'\ttLearning       : {args.lr}')
    print(f'\tGlobal Rounds   : {args.epochs}\n')

    print('\n----------------Federated parameters:------------\n')
    if args.iid:
        print('\tIID')
    else:
        print('\tNon-IID')
    print(f'\tFraction of users  : {args.frac}')
    print(f'\tLocal Batch size   : {args.local_bs}')
    print(f'\tLocal Epochs       : {args.local_ep}\n')
    return
from model import MLP
import argparse
from tqdm import tqdm
from dataset import MNIST_Dataset
import cfg
import random
import numpy as np
from utils import AverageMeter,compute_acc,Accuracy_averagemeter
from tensor import Tensor
import logging
import os
import matplotlib.pyplot as plt
import pickle

LOG_FMT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"

def augment_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',type=str)
    parser.add_argument('--test-log-dir',type=str,default='./test_log/')
    parser.add_argument('--nl',type=int,default=2,help='num of layers')
    parser.add_argument('--dim-list',type=int,nargs='+',default=[28*28,512])

    return parser

def split_batch(index_list,batch_size):
    batch_index = random.sample(index_list, batch_size)
    for each in batch_index:
        index_list.remove(each)

    return np.array(batch_index)

def main():
    parser = augment_parser()
    args = parser.parse_args()
    dataset = MNIST_Dataset()
    train_data,test_data,train_label,test_label = dataset.train_data,dataset.test_data,dataset.train_label,dataset.test_label

    if not os.path.exists(args.test_log_dir):
        os.makedirs(args.test_log_dir)

    mlp = MLP(n_layers=args.nl,n_class=10,dim_list=args.dim_list)
    with open(args.model_path,'rb') as f:
        state_dict = pickle.load(f)
    mlp.load_state_dict(state_dict)

    mlp.eval()
    test_loss = AverageMeter()
    test_acc = Accuracy_averagemeter()

    log_file_name = args.test_log_dir + 'test.log'
    if not os.path.exists(log_file_name):
        os.system(f'touch {log_file_name}')

    logging.basicConfig(
        level=logging.INFO,
        filename=log_file_name,
        filemode="a",
        format=LOG_FMT
    )

    for i in tqdm(range(test_data.shape[0])):
        data, label = test_data[i].reshape(1,-1),test_label[i].reshape(1,-1)
        data = Tensor(value=data)
        pred, loss = mlp(data,label)
        test_loss.update(val=float(loss))
        test_acc.update(val=int(compute_acc(pred.value,label)))

    log_info = 'Test Model Path: {0} --- test loss : {1} --- test accuracy:{2}'.format(args.model_path, test_loss.avg,test_acc.avg)
    logging.info(log_info)

if __name__ == "__main__":
    main()
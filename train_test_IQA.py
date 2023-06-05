import os
import argparse
import random
import numpy as np
from engine import IQASolver
import torch 
import math

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(config):
    print(config.netFile)
    set_random_seed(10)
    options = {
        'dataset':config.dataset,
        'batch_size': config.batch_size,
        'epochs': config.epochs,
    }
    
    path = {
        'QADS': '/home/jfu/datasets/sriqa/QADS/',
        'CVIU': '/home/jfu/datasets/sriqa/CVIU/',
        'Waterloo': '/home/jfu/datasets/sriqa/Waterloo/',
        'RealSRQ': '/home/jfu/datasets/sriqa/RealSRQ/',
        'SISAR': '/home/jfu/datasets/sriqa/SISAR/'
    }
      
    if config.dataset == 'QADS':          
        imgIndex = list(range(1,21))
        srIndex = list(range(1,21))
    elif config.dataset == 'CVIU':
        imgIndex = list(range(30))
        srIndex = list(range(7))
    elif config.dataset == 'Waterloo':   
        imgIndex = list(range(13))
        srIndex = list(range(8))
    elif config.dataset == 'RealSRQ':   
        imgIndex = list(range(6))
        srIndex = list(range(10))
    elif config.dataset == 'SISAR':   
        imgIndex = list(range(100))
        srIndex = list(range(6))

    srcc_all = np.zeros(config.train_test_num, dtype=np.float)
    plcc_all = np.zeros(config.train_test_num, dtype=np.float)
    krcc_all = np.zeros(config.train_test_num, dtype=np.float)


    if not os.path.exists(os.path.join('./save/', config.netFile, config.dataset)):
        os.makedirs(os.path.join('./save/', config.netFile, config.dataset))
    if not os.path.exists('./log/'):
        os.makedirs('./log/')

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    log = open(os.path.join('./save/', config.netFile, config.dataset, 'log.txt'), 'w')
    for i in range(config.train_test_num):
        print('Round %d' % (i+1))
        log.write('Round %d\n' % (i+1))
        # Randomly select 80% images for training and the rest for testing
        random.shuffle(imgIndex)
        random.shuffle(srIndex)

        if config.dataset == 'SISAR':
            train_imgIndex = imgIndex[0:math.floor(0.8*len(imgIndex))]
            test_imgIndex = imgIndex[math.floor(0.8*len(imgIndex)):len(imgIndex)]

            train_srIndex = srIndex[0:4]
            test_srIndex = srIndex[4:len(srIndex)]
        else:
            train_imgIndex = imgIndex[0:math.floor(0.8*len(imgIndex))]
            test_imgIndex = imgIndex[math.floor(0.8*len(imgIndex)):len(imgIndex)]

            train_srIndex = srIndex[0:math.floor(0.8*len(srIndex))]
            test_srIndex = srIndex[math.floor(0.8*len(srIndex)):len(srIndex)]
    
        options['train_imgIndex'] = train_imgIndex
        options['test_imgIndex'] = test_imgIndex
        options['train_srIndex'] = train_srIndex
        options['test_srIndex'] = test_srIndex

        savePath = os.path.join('./save/', config.netFile, config.dataset, 'round%d'%i)
        options['savePath'] = savePath
        solver = IQASolver(config, path, options)
        srcc_all[i], plcc_all[i], krcc_all[i] = solver.train(log)

    print(srcc_all)
    print(plcc_all)
    print(krcc_all)
    log.write(' '.join([str(i) for i in srcc_all])+'\n')
    log.write(' '.join([str(i) for i in plcc_all])+'\n')
    log.write(' '.join([str(i) for i in krcc_all])+'\n')

    srcc_mean = np.mean(srcc_all)
    plcc_mean = np.mean(plcc_all)
    krcc_mean = np.mean(krcc_all)
    print('Testing mean SRCC %4.4f,\tmean PLCC %4.4f,mean KRCC %4.4f' % (srcc_mean, plcc_mean, krcc_mean))
    log.write('Testing mean SRCC %4.4f,\tmean PLCC %4.4f,mean KRCC %4.4f\n' % (srcc_mean, plcc_mean, krcc_mean))
    log.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', dest='dataset', type=str, default='Waterloo', help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=10, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=1, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=96, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='Train-test times')
    parser.add_argument('--use_scale', dest='use_scale', type=int, default=0, help='enable scale embedding')
    parser.add_argument('--gpuid', dest='gpuid', type=int, default=0, help='enable scale embedding')
    parser.add_argument('--netFile', dest='netFile', type=str, default="CNNIQA", help='iqa model')
 
    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%config.gpuid
    main(config)



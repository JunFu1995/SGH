import torch
from scipy import stats
import numpy as np
import torchvision
# from datasets import * 
import os 
import torch.nn as nn 
from importlib import import_module
from six.moves import cPickle as pickle #for performance
# from scipy.optimize import curve_fit
# import copy 
# import scipy.io as scio
import nets
def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

class IQASolver(object):
    """Solver for training and testing hyperIQA"""
    def __init__(self, config, path, options):
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num

        self.model = nets.buildModel(config.netFile).cuda()
        self.model.train(True)

        self.l1_loss = torch.nn.L1Loss().cuda()
        self.netFile = config.netFile

        #backbone_params = list(map(id, self.model_hyper.res.parameters()))
        #
        if config.netFile == 'HyperIQA':
            backbone_params = list(map(id, self.model.res.parameters()))
            self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model.parameters())
            self.lr = config.lr
            self.lrratio = config.lr_ratio
            self.weight_decay = config.weight_decay
            paras = [{'params': self.hypernet_params, 'lr': self.lr}, 
                     {'params': self.model.res.parameters(), 'lr': self.lr}
                     ]
            ## * self.lrratio},
            self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)
            #self.scheduler = torch.optim.lr_scheduler.StepLR(self.solver, 25, 0.5)
        elif config.netFile == 'maniqa':
            self.l1_loss = torch.nn.MSELoss().cuda()
            self.solver = torch.optim.Adam(
                self.model.parameters(),
                lr= 2e-5,
                weight_decay=config.weight_decay
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.solver, T_max=50, eta_min=0)

        else:
            self.hypernet_params = self.model.parameters() 
            self.lr = config.lr
            self.lrratio = config.lr_ratio
            self.weight_decay = config.weight_decay
            paras = [
                     {'params': self.model.parameters(), 'lr': self.lr}
                     ]
            self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.solver, 25, 0.5)



        self._options = options
        self._path = path 


        if config.netFile == 'DeepSRQ':
            train_transforms = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
            #                                  std=(0.229, 0.224, 0.225))
            ])             
            # 224
            #crop_size = 32
            test_transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                #                                  std=(0.229, 0.224, 0.225))
            ])
        else:
            crop_size = 224
            train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
            ])             
            # 224
            crop_size = 32
            test_transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                 std=(0.229, 0.224, 0.225))
            ])
                       
        if self._options['dataset'] in ['QADS', 'Waterloo', 'CVIU']:
            ds = import_module('datasets') if config.netFile != 'DeepSRQ' else import_module('datasets_deepsrq')
            dsname = self._options['dataset']
            train_data = getattr(ds, dsname)(self._path[dsname], self._options['train_imgIndex'], self._options['train_srIndex'], train_transforms, config.train_patch_num)
            test_data = getattr(ds, dsname)(self._path[dsname], self._options['test_imgIndex'], self._options['test_srIndex'], test_transforms, config.test_patch_num, False)
        else:
            raise AttributeError('Only support QADS, Waterloo, CVIU, RealSRQ right now!')


        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=4, pin_memory=False)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1,
            shuffle=False, num_workers=0, pin_memory=False)

        self.train_data = self._train_loader 
        self.test_data = self._test_loader 

        # save 
        if not os.path.exists(self._options['savePath']):
            os.makedirs(self._options['savePath'])
        self.unfold =  nn.Unfold(kernel_size=(224, 224), stride=64) 
        self.use_scale = config.use_scale
        #self.print_step = len(train_data) // 25 // self._options['batch_size'] 
        self.totoal_num = len(self._train_loader)  * self.epochs 
        self.print_step = self.totoal_num // 50 

    def train(self, log):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0
        best_krcc = 0.0 
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTest_KROCC')
        log.write('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTest_KROCC\n')
        ts = 0
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []
            for imgPath, img, label, z in self.train_data:
                # Data.
                img = img.cuda() #torch.tensor(img.cuda())
                label = label.cuda().float() #torch.tensor(label.cuda()).float()
                z = z.unsqueeze(-1).cuda().float() / 10.0 
                self.solver.zero_grad()

                # Quality prediction
                pred = self.model((img,z)) 
                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                loss = self.l1_loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

                ts += 1 
                # if ts % self.print_step == 0:
            with torch.no_grad():
                train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
                test_srcc, test_plcc, test_krocc, epoch_data = self.test(self.test_data)
                if test_srcc > best_srcc:
                    best_srcc = test_srcc
                    best_plcc = test_plcc
                    best_krcc = test_krocc
                    #scio.savemat(os.path.join(self._options['savePath'], 'result.mat'), {'score': epoch_data['pred_scores'], 'label': epoch_data['gt_scores'], 'name': epoch_data['img_names'], 'scale': epoch_data['scale']})
                    #torch.save(self.model.state_dict(), os.path.join(self._options['savePath'], 'model.pkl'))
                    save_dict(epoch_data, os.path.join(self._options['savePath'], 'result.pkl'))
                print('%3d \t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                      (t, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc, test_krocc))
                log.write('%d \t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f\n' %
                      (t, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc, test_krocc))
                    # epoch_loss = []
                    # pred_scores = []
                    # gt_scores = []
            # Update optimizer
            # if self.netFile == 'HyperIQA':
            #     lr = self.lr / pow(10, (t // 6))
            #     if t > 8:
            #         self.lrratio = 1
            #     if self.netFile == 'HyperIQA':
            #         self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
            #                       {'params': self.model.res.parameters(), 'lr': self.lr}
            #                       ]
            # else:
            #     self.paras = [{'params': self.model.parameters(), 'lr': lr * self.lrratio}]
            #     self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)                
            # else:
            #     self.scheduler.step()


        print('Best test SRCC %f, PLCC %f, KRCC %f' % (best_srcc, best_plcc, best_krcc))
        log.write('Best test SRCC %f, PLCC %f, KRCC %f\n' % (best_srcc, best_plcc, best_krcc))
        return best_srcc, best_plcc, best_krcc

    def test(self, data):
        """Testing"""
        self.model.train(False)
        pred_scores = []
        gt_scores = []
        img_names = []
        epoch_data = {}
        z_l = []
        batch_size = 96 
        for imgname, img, label, z in data:
            # Data.
            label = label.cuda().float() #torch.tensor(label.cuda()).float()
            z_l += [z.item()]
            img = img.cuda() #torch.tensor(img.cuda())
            z = z.unsqueeze(-1).cuda().float() / 10.0 
            img = self.unfold(img).view(1, img.shape[1], 224, 224, -1)[0]
            img = img.permute(3,0,1,2)
            img = torch.split(img, batch_size, dim=0)
            pred_s = []
            for i in img:
                pred = self.model((i,z))
                pred_s += pred.detach().cpu().tolist()

            pred_s = np.mean(pred_s) 
            pred_scores += [pred_s] #append(float(pred.item()))
            gt_scores = gt_scores + label.cpu().tolist()
            img_names.append(imgname[0])

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 1)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, 1)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        test_krocc, _ = stats.kendalltau(pred_scores, gt_scores)

        epoch_data['pred_scores'] = pred_scores.tolist()
        epoch_data['gt_scores'] = gt_scores.tolist() 
        epoch_data['img_names'] = img_names
        epoch_data['scale'] = z_l
        epoch_data['plcc'] = test_plcc
        epoch_data['srcc'] = test_srcc             
        epoch_data['krocc'] = test_krocc

        self.model.train(True)
        return test_srcc, test_plcc, test_krocc, epoch_data
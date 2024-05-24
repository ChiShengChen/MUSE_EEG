"""
Object recognition Things-EEG2 dataset
use 250 Hz data

The code is modified from https://github.com/eeyhsong/NICE-EEG

MUSE*: Use Enc_muse_eeg as EEG Encoder
MUSE-GA: Need to modifify STConvEEGModel
MUSE-Nerv-*: Use Enc_nervformer_eeg as EEG Encoder
MUSE-Nerv-GA: Need to modifify NervFormerEEGModel

Default is using SK-InfoNCE loss
"""

import os
import argparse
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from torch.autograd import Variable
from einops.layers.torch import Rearrange
from einops import rearrange

from muse_eeg_model import Proj_eeg, Proj_img, Enc_nervformer_eeg, Enc_muse_eeg


gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
# result_path = '/home/NICE/results/' 
result_path = './THINGS-EEG/MUSE_EEG/results/' 
model_path = './THINGS-EEG/MUSE_EEG/model/' 

model_idx = 'test0'
 
parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--epoch', default='200', type=int)
parser.add_argument('--num_sub', default=10, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=1000, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training. ')


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


# Image2EEG
class IE():
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 200
        self.batch_size = args.batch_size
        self.batch_size_test = 400
        self.batch_size_img = 500 
        self.n_epochs = args.epoch
        self.lambda_cen = 0.003
        self.alpha = 0.5
        self.proj_dim = 256
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub

        self.model_idx = 'test0_' + str(self.nSub) + '_'

        local_path = './THINGS-EEG/MUSE_EEG/'

        self.start_epoch = 0
        self.eeg_data_path = local_path + 'Data/Things-EEG2/Preprocessed_data_250Hz/'
        self.img_data_path = local_path + 'Data/Things-EEG2/DNN_feature_maps/pca_feature_maps/' + args.dnn + '/pretrained-True/'
        self.test_center_path = local_path + 'Data/Things-EEG2/Image_set/'

        self.log_write = open(result_path + "log_subject%d.txt" % self.nSub, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.Proj_eeg = Proj_eeg().cuda()
        self.Proj_img = Proj_img().cuda()
        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        self.Enc_nervformer_eeg = Enc_nervformer_eeg().cuda()
        self.Enc_nervformer_eeg = nn.DataParallel(self.Enc_nervformer_eeg, device_ids=[i for i in range(len(gpus))])

        self.Enc_muse_eeg = Enc_muse_eeg().cuda()
        self.Enc_muse_eeg = nn.DataParallel(self.Enc_muse_eeg, device_ids=[i for i in range(len(gpus))])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print('initial define done.')


    def get_eeg_data(self):
        train_data = []
        train_label = []
        test_data = []
        test_label = np.arange(200)
        
        print("self.nSub: ", self.nSub)
        train_data = np.load(self.eeg_data_path + 'sub-' + format(self.nSub, '02') + '/preprocessed_eeg_training.npy', allow_pickle=True)
        train_data = train_data['preprocessed_eeg_data']
        train_data = np.mean(train_data, axis=1)
        train_data = np.expand_dims(train_data, axis=1)

        test_data = np.load(self.eeg_data_path + 'sub-' + format(self.nSub, '02') + '/preprocessed_eeg_test.npy', allow_pickle=True)
        test_data = test_data['preprocessed_eeg_data']
        test_data = np.mean(test_data, axis=1)
        test_data = np.expand_dims(test_data, axis=1)

        return train_data, train_label, test_data, test_label

    def get_image_data(self):
        train_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_training.npy', allow_pickle=True)
        test_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_test.npy', allow_pickle=True)

        train_img_feature = np.squeeze(train_img_feature)
        test_img_feature = np.squeeze(test_img_feature)

        return train_img_feature, test_img_feature
        
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(self):
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)

        train_eeg, _, test_eeg, test_label = self.get_eeg_data()
        # The training set includes 1654concepts × 10images × 4repetitions, 63 channels.
        # print("train_eeg1: ", train_eeg.shape) #  (16540, 1, 63, 250)
        train_img_feature, _ = self.get_image_data() 
        # Images were resized to 224×224 and normalized before being processed by the image encoder
        # print('train_img_feature1: ', train_img_feature.shape) # (16540, 768)
        test_center = np.load(self.test_center_path + 'center_' + self.args.dnn + '.npy', allow_pickle=True)

        # shuffle the training data
        train_shuffle = np.random.permutation(len(train_eeg))
        train_eeg = train_eeg[train_shuffle]
        train_img_feature = train_img_feature[train_shuffle]

        val_eeg = torch.from_numpy(train_eeg[:740])
        val_image = torch.from_numpy(train_img_feature[:740])

        train_eeg = torch.from_numpy(train_eeg[740:])
        print('train_eeg: ', train_eeg.shape) # torch.Size([15800, 1, 63, 250])
        train_image = torch.from_numpy(train_img_feature[740:])
        print('train_image: ', train_image.shape) # torch.Size([15800, 768])

        dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = torch.utils.data.TensorDataset(val_eeg, val_image)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False)

        test_eeg = torch.from_numpy(test_eeg)

        test_center = torch.from_numpy(test_center)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)

        # Optimizers
        self.optimizer = torch.optim.AdamW(itertools.chain(self.Enc_nervformer_eeg.parameters(), self.Proj_eeg.parameters(), self.Proj_img.parameters()), lr=self.lr, betas=(self.b1, self.b2))

        num = 0
        best_loss_val = np.inf
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        for e in range(self.n_epochs):
            in_epoch = time.time()

            self.Enc_nervformer_eeg.train()
            self.Proj_eeg.train()
            self.Proj_img.train()

            # starttime_epoch = datetime.datetime.now()

            for i, (eeg, img) in enumerate(self.dataloader):

                eeg = Variable(eeg.cuda().type(self.Tensor))
                # print("eeg: ", eeg.shape) # torch.Size([1000, 1, 63, 250])
                # img = Variable(img.cuda().type(self.Tensor))
                img_features = Variable(img.cuda().type(self.Tensor))
                # print("img_features: ", img_features.shape) # torch.Size([1000, 768])
                # label = Variable(label.cuda().type(self.LongTensor))
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = Variable(labels.cuda().type(self.LongTensor))

                # eeg cor
                flattened_eeg_data_tensor = eeg.view(eeg.shape[0], -1)  # [1000, 63*250]

                # calculate L2 norm
                eeg_data_norms = torch.norm(flattened_eeg_data_tensor, p=2, dim=1, keepdim=True)
                normalized_eeg_tensor = flattened_eeg_data_tensor / eeg_data_norms
                eeg_cos_similarity_matrix = torch.mm(normalized_eeg_tensor, normalized_eeg_tensor.transpose(0, 1)) # [1000, 1000]

                # image cor
                img_features_norms = torch.norm(img_features, p=2, dim=1, keepdim=True) # torch.Size([1000, 768])
                normalized_tensor = img_features / img_features_norms
                img_cos_similarity_matrix = torch.mm(normalized_tensor, normalized_tensor.transpose(0, 1)) # [1000, 1000]
                eeg_img_cos_similarity = F.cosine_similarity(eeg_cos_similarity_matrix, img_cos_similarity_matrix)
                eeg_img_cos_sim_loss = 1 - eeg_img_cos_similarity.mean()


                # obtain the features
                eeg_features = self.Enc_nervformer_eeg(eeg)
                # print('eeg_features1: ', eeg_features.shape) # torch.Size([1000, 1440])

                # project the features to a multimodal embedding space
                eeg_features = self.Proj_eeg(eeg_features)
                # print('eeg_features_proj: ', eeg_features.shape) #  torch.Size([1000, 768])
                img_features = self.Proj_img(img_features)
                # print("img_features_proj: ", img_features.shape) #  torch.Size([1000, 768])


                # normalize the features
                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)

                # cosine similarity as the logits
                logit_scale = self.logit_scale.exp()
                logits_per_eeg = logit_scale * eeg_features @ img_features.t()
                logits_per_img = logits_per_eeg.t()

                loss_eeg = self.criterion_cls(logits_per_eeg, labels)
                loss_img = self.criterion_cls(logits_per_img, labels)

                loss_cos = (loss_eeg + loss_img) / 2

                # total loss SK-InfoNCE loss
                loss = loss_cos + eeg_img_cos_sim_loss

                # INfoNCE loss
                # loss = loss_cos 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            if (e + 1) % 1 == 0:
                self.Enc_nervformer_eeg.eval()
                self.Proj_eeg.eval()
                self.Proj_img.eval()
                with torch.no_grad():
                    # * validation part
                    for i, (veeg, vimg) in enumerate(self.val_dataloader):

                        veeg = Variable(veeg.cuda().type(self.Tensor))
                        vimg_features = Variable(vimg.cuda().type(self.Tensor))
                        vlabels = torch.arange(veeg.shape[0])
                        vlabels = Variable(vlabels.cuda().type(self.LongTensor))

                        veeg_features = self.Enc_nervformer_eeg(veeg)
                        veeg_features = self.Proj_eeg(veeg_features)
                        vimg_features = self.Proj_img(vimg_features)

                        veeg_features = veeg_features / veeg_features.norm(dim=1, keepdim=True)
                        vimg_features = vimg_features / vimg_features.norm(dim=1, keepdim=True)

                        logit_scale = self.logit_scale.exp()
                        vlogits_per_eeg = logit_scale * veeg_features @ vimg_features.t()
                        vlogits_per_img = vlogits_per_eeg.t()

                        vloss_eeg = self.criterion_cls(vlogits_per_eeg, vlabels)
                        vloss_img = self.criterion_cls(vlogits_per_img, vlabels)

                        vloss = (vloss_eeg + vloss_img) / 2

                        if vloss <= best_loss_val:
                            best_loss_val = vloss
                            best_epoch = e + 1

                            torch.save(self.Enc_nervformer_eeg.module.state_dict(), model_path + self.model_idx + 'Enc_custom_eeg_cls.pth')
                            torch.save(self.Proj_eeg.module.state_dict(), model_path + self.model_idx + 'Proj_eeg_cls.pth')
                            torch.save(self.Proj_img.module.state_dict(), model_path + self.model_idx + 'Proj_img_cls.pth')

                print('Epoch:', e,
                      '  Cos eeg: %.4f' % loss_eeg.detach().cpu().numpy(),
                      '  Cos img: %.4f' % loss_img.detach().cpu().numpy(),
                      '  loss val: %.4f' % vloss.detach().cpu().numpy(),
                      )
                self.log_write.write('Epoch %d: Cos eeg: %.4f, Cos img: %.4f, loss val: %.4f\n'%(e, loss_eeg.detach().cpu().numpy(), loss_img.detach().cpu().numpy(), vloss.detach().cpu().numpy()))


        # * test part
        all_center = test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        self.Enc_nervformer_eeg.load_state_dict(torch.load(model_path + self.model_idx + 'Enc_custom_eeg_cls.pth'), strict=False)
        self.Proj_eeg.load_state_dict(torch.load(model_path + self.model_idx + 'Proj_eeg_cls.pth'), strict=False)
        self.Proj_img.load_state_dict(torch.load(model_path + self.model_idx + 'Proj_img_cls.pth'), strict=False)

        self.Enc_nervformer_eeg.eval()
        self.Proj_eeg.eval()
        self.Proj_img.eval()

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.type(self.Tensor))
                tlabel = Variable(tlabel.type(self.LongTensor))
                all_center = Variable(all_center.type(self.Tensor))            

                tfea = self.Proj_eeg(self.Enc_nervformer_eeg(teeg))

                tfea = tfea / tfea.norm(dim=1, keepdim=True)
                similarity = (100.0 * tfea @ all_center.t()).softmax(dim=-1)  # no use 100?
                _, indices = similarity.topk(5)

                tt_label = tlabel.view(-1, 1)
                total += tlabel.size(0)
                top1 += (tt_label == indices[:, :1]).sum().item()
                top3 += (tt_label == indices[:, :3]).sum().item()
                top5 += (tt_label == indices).sum().item()

            
            top1_acc = float(top1) / float(total)
            top3_acc = float(top3) / float(total)
            top5_acc = float(top5) / float(total)
        
        print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))
        self.log_write.write('The best epoch is: %d\n' % best_epoch)
        self.log_write.write('The test Top1-%.6f, Top3-%.6f, Top5-%.6f\n' % (top1_acc, top3_acc, top5_acc))
        
        return top1_acc, top3_acc, top5_acc


def main():
    args = parser.parse_args()

    num_sub = args.num_sub   
    
    cal_num = 0
    aver = []
    aver3 = []
    aver5 = []
    
    for i in range(num_sub):

        cal_num += 1
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(args.seed)

        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print('Subject %d' % (i+1))
        ie = IE(args, i + 1)

        Acc, Acc3, Acc5 = ie.train()
        print('THE BEST ACCURACY IS ' + str(Acc))

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))

        aver.append(Acc)
        aver3.append(Acc3)
        aver5.append(Acc5)

    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))

    column = np.arange(1, cal_num+1).tolist()
    column.append('ave')
    pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    pd_all.to_csv(result_path + 'result.csv')


class Print_model_info():
    def __init__(self):
        model_idx = 'test0'
        self.Proj_eeg = Proj_eeg().cuda()
        self.Proj_img = Proj_img().cuda()
        self.Proj_eeg.load_state_dict(torch.load(model_path + model_idx + 'Proj_eeg_cls.pth'), strict=False)
        self.Proj_img.load_state_dict(torch.load(model_path + model_idx + 'Proj_img_cls.pth'), strict=False)

        print(self.Proj_img[2].weight.shape)


def print_mdl_info():
    Print_model_info()



if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    # print_mdl_info()

    print(time.asctime(time.localtime(time.time())))
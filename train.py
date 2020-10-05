import os
import numpy as np
import torch
print(torch.__version__)
from torch.utils.data import DataLoader
from dataloader.dataloader import Image_Loader
from models.GhostNet_bap import ghost_net
from losses.losses import First_Loss, Recur_Loss
from utils.parameter import get_parameters
import time
from matplotlib import pyplot as plt
import torch.nn.functional as F
from utils.attention import calculate_pooling_center_loss, attention_crop_drop
from scheduler import GradualWarmupScheduler
from evaluation.eval import get_tpr_from_threshold, get_thresholdtable_from_fpr

classes = ['fake', 'real']

class Trainer(object):
    def __init__(self, data_loader, eval_data_loader, config, device = 'cpu'):

        self.device = device
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.data_loader = data_loader
        self.eval_data_loader = eval_data_loader
        self.model_save_step = config.model_save_step
        self.log_step = config.log_step
        self.total_epoch = config.total_epoch
        self.model_save_path = config.model_save_path
        self.parallel = config.parallel
        self.pretrained_model = config.pretrained_model
        self.w_loss = config.w_loss
        self.plot_loss_step = config.plot_loss_step
        self.alpha = config.alpha
        self.build_model()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def train(self):

        # Data iterator
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        # Start with trained model
        if self.pretrained_model:
            step = self.pretrained_model + 1
        else:
            step = 1
        # start = 0
        # Start time
        start_time = time.time()
        cros_loss_values = []
        pc_loss_values = []

        eval_cross_values = []
        eval_pc_values = []
        precision_values = []
        lr_values = []

        loss_eva_total = 0
        log_file = open('training_log.txt', 'w')
        num_iters = self.total_epoch*len(self.data_loader)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_iters)
        scheduler_warmup = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=1000, after_scheduler=scheduler)

        # start = 0
        iteration = 0
        for epoch in range(0, self.total_epoch):
            step=0
            for img_1, atr_label, spoof_type_label, illum_label, env_label, spoof_label in self.data_loader:

                step = step+1
                iteration=epoch*len(self.data_loader) + step
                log_file = open('training_log.txt', 'a')

                # if (step+1)%100 !=0:
                #     continue

                # ================== Train ================== #
                self.model.train()
                # self.adjust_learning_rate(self.optimizer, step, self.lr)
                img_1 = img_1.to(self.device)

                atr_label = atr_label.to(self.device)
                atr_label = atr_label.squeeze()
                # atr_label = atr_label.long()

                illum_label = illum_label.to(self.device)
                illum_label = illum_label.squeeze()
                illum_label = illum_label.long()

                spoof_type_label = spoof_type_label.to(self.device)
                spoof_type_label = spoof_type_label.squeeze()
                spoof_type_label = spoof_type_label.long()

                env_label = env_label.to(self.device)
                env_label = env_label.squeeze()
                env_label = env_label.long()

                spoof_label = spoof_label.to(self.device)
                spoof_label = spoof_label.squeeze()
                spoof_label = spoof_label.long()


                # for param_group in self.optimizer.param_groups:
                lr_values.append(self.optimizer.param_groups[0]['lr'])
                # print(self.optimizer.param_groups[0]['lr'])

                # Forward
                attention_maps, raw_features, atr_pred, spoof_type_pred, illum_pred, env_pred, spoof_pred1= self.model(img_1)
                
                features = raw_features.reshape(raw_features.shape[0], -1)
                # print(features.size())

                feature_center_loss, center_diff = calculate_pooling_center_loss(
                    features, self.feature_center, spoof_label, alfa=self.alpha)

                # update model.centers
                # print(self.feature_center.size(), center_diff.size())
                self.feature_center[spoof_label] += center_diff

                # compute refined loss
                img_crop, img_drop = attention_crop_drop(attention_maps, img_1)
                _, _, _, _, _, _, spoof_pred2 = self.model(img_drop)
                _, _, _, _, _, _, spoof_pred3 = self.model(img_crop)

                loss1, spoof_loss, atr_loss, spoof_type_loss, illum_loss = self.first_loss(atr_pred, spoof_type_pred, illum_pred, env_pred, spoof_pred1, atr_label, spoof_type_label, illum_label, env_label, spoof_label)
                loss2 = self.recur_loss(spoof_pred2, spoof_label)/2
                loss3 = self.recur_loss(spoof_pred3, spoof_label)/2

                overall_loss = loss1 + loss2 + loss3 + feature_center_loss
                    
                # Backward + Optimize
                self.reset_grad()
                overall_loss.backward()
                self.optimizer.step()
                scheduler_warmup.step(iteration)
                # self.scheduler.step()

                eva_loss = 0
                correct = 0
                # total = len(self.eval_data_loader)
                total = 296

                if (iteration+1) % self.model_save_step==0:
                    torch.save(self.model.state_dict(),
                               os.path.join(self.model_save_path, '{}_EfficientNet.pth'.format(iteration + 1)))

                    loss_eva_total = 0

                # if (step+1) % 5000==0:
                #     FA_nums=0
                #     FR_nums=0
                #     threshold = 0.6
                #     self.model.eval()
                #     with torch.no_grad():
                #         for eval_data in self.eval_data_loader:

                #             eval_img, eval_label, _ = eval_data
                #             eval_img = eval_img.to(self.device)
                #             # eval_label = eval_label.to(self.device)
                #             eval_label = int(eval_label.item())

                #             # Forward
                #             _, _, preds = self.model(eval_img)
                #             # print(eval_pre.size())-
                #             # preds = torch.sigmoid(preds)
                #             preds = preds.cpu().data.numpy()
                #             preds = np.squeeze(preds)
                #             # print(prediction)----------
                #             if preds>threshold:
                #                 prediction='real'
                #                 if classes[eval_label]=='fake':
                #                     FA_nums=FA_nums+1
                #             else:
                #                 prediction='fake'
                #                 if classes[eval_label]=='real':
                #                     FR_nums=FR_nums+1

                #     self.model.train()
                #     # loss_eva_total = loss_eva_total/total
                #     print('===================EVALUATION========================')
                #     print('FAR: {}/{},  FRR: {}/{},  Error of Evalution: {}/{}'.format(FA_nums, 191, FR_nums, 105, FA_nums+FR_nums, total))
                #     log_file.write('FAR: {}/{},  FRR: {}/{},  Error of Evalution: {}/{} \n'.format(FA_nums, 191, FR_nums, 105, FA_nums+FR_nums, total))

                if (iteration+1) % 100==0:

                    scores = []
                    test_labels = []
                    self.model.eval()
                    with torch.no_grad():
                        tem = 0
                        for img_test, _, _, _, _, spoof_label_test in self.eval_data_loader:
                            img_test = img_test.to(self.device)
                            # _, _, preds = self.model(img_test)
                            _, _, _, _, _, _, preds = self.model(img_test)
                            preds = preds.cpu().data.numpy().squeeze()
                            spoof_label_test = spoof_label_test.cpu().data.numpy().squeeze()

                            spoof_label_test = list(spoof_label_test.squeeze())
                            preds = list(preds)

                            scores = scores + preds
                            test_labels = test_labels + spoof_label_test

                            tem = tem + 1
                            if tem ==10:
                                break

                        # calculate tpr
                        fpr_list = [0.01, 0.005, 0.001]
                        threshold_list = get_thresholdtable_from_fpr(scores,test_labels, fpr_list)
                        tpr_list = get_tpr_from_threshold(scores,test_labels, threshold_list)

                        # show results
                        print('=========================================================================')
                        print('TPR@FPR=10E-3: {}\n'.format(tpr_list[0]))
                        print('TPR@FPR=5E-3: {}\n'.format(tpr_list[1]))
                        print('TPR@FPR=10E-4: {}\n'.format(tpr_list[2]))
                        print('=========================================================================')

                # Print out loss info
                if (iteration + 1) % self.log_step == 0:               
                    print("epoch: {}/{}, interation: {}, overall_loss: {:.5f}".format(epoch+1, self.total_epoch, step+1, overall_loss.item()))
                    print("loss_1: {:.5}, spoof_loss: {:.5}, atr_loss: {:.5}, spoof_type_loss: {:.5}, illum_loss: {:.5}".format(loss1, spoof_loss, atr_loss, spoof_type_loss, illum_loss))
                    print("loss_2: {:.5}, loss_3: {:.5}, feature_center_loss: {:.5}".format(loss2, loss3, feature_center_loss))
                    log_file.write("epoch: {}/{}, interation: {}, overall_loss: {:.5f} \n".format(epoch+1, self.total_epoch, step+1, overall_loss.item()))
                    log_file.write("loss_1: {:.5}, atr_loss: {:.5}, spoof_type_loss: {:.5}, illum_loss: {:.5}".format(loss1, atr_loss, spoof_type_loss, illum_loss))
                    log_file.write("loss_2: {:.5}, loss_3: {:.5}, feature_center_loss: {:.5}".format(loss2, loss3, feature_center_loss))
                    log_file.close()
                if (iteration+1) % self.plot_loss_step==0:
                    cros_loss_values.append(overall_loss.item())
                    # pc_loss_values.append(pc_loss)

                    # eval_cross_values.append(eval_entropy_loss)
                    # eval_pc_values.append(eva_pc_loss)

                    # precision_values.append(correct/float(total))
                    self.plot_loss(cros_loss_values)
                    self.plot_lr(lr_values)
                    # self.plot_accuracy(correct/float(total))

    def build_model(self):

        self.model = ghost_net()
        net_dict = self.model.state_dict()
        if True:
            pretrained_model = torch.load('Pre_trained_GhostNet.pth')
            # model.load_state_dict(state_dict)
            # 1. filter out unnecessary keys
            pretrained_model = {k: v for k, v in pretrained_model.items() if k in net_dict}
            # 2. overwrite entries in the existing state dict
            net_dict.update(pretrained_model)
            # 3. load the new state dict
            self.model.load_state_dict(net_dict)

        self.model = self.model.to(self.device)
        if self.parallel:
            self.model = nn.DataParallel(self.model)

        self.feature_center = torch.zeros(2, 32*160)
        self.feature_center = self.feature_center.to(self.device)
        
        # Loss and optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum = 0.9, weight_decay = 1e-4)

        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.lr/5, max_lr=self.lr, step_size_up=2000, step_size_down=2000, mode='triangular2')

        self.first_loss = First_Loss()
        self.recur_loss = Recur_Loss()
        # print networks
        # print(self.model)

    def load_pretrained_model(self):
        self.model.load_state_dict(torch.load(os.path.join(
            'saved_models', '{}_GhostNet_Logictech.pth'.format(self.pretrained_model))), strict=False)
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))
        self.model = self.model.to(self.device)

    def reset_grad(self):
        self.optimizer.zero_grad()

    def plot_loss(self, overall_loss):

        plt.figure(1)
        plt.plot(overall_loss, color = 'b') # plotting cross-entropy loss
        # plt.plot(pc_loss_values, color = 'r') # plotting evaluation loss

        # plt.plot(eval_cross_values, color = 'g')
        # plt.plot(eval_pc_values, color = 'yellow')

        # plt.plot(precision_values, color = 'yellow') # plotting evaluation loss

        plt.legend(['overall_loss'], loc='upper left')
        # plt.show()
        plt.savefig('plot_loss.png')

    def plot_lr(self, lr_values):
        
        plt.figure(2)
        plt.plot(lr_values, color = 'b') #

        plt.legend(['learning rate scheduler'], loc='upper left')
        # plt.show()
        plt.savefig('plot_lr.png')

    def plot_accuracy(self, precision):

        plt.plot(precision, 'b') # precision

        # plt.show()
        plt.savefig('plot_accuracy.png')

    def adjust_learning_rate(self, optimizer_net, step, learning_rate):
        lr = learning_rate * (0.2 ** (step //10000))

        for param_group in optimizer_net.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)

    config = get_parameters()
    dataset = Image_Loader(root_path='./dataloader/data_train_all.csv', image_size=[config.imsize, config.imsize], transforms_data=True, aug = False, phase = 'train')
    data_loader = DataLoader(dataset = dataset, batch_size = config.batch_size, shuffle = True, num_workers=4, pin_memory=False)

    eval_dataset = Image_Loader(root_path='./dataloader/data_test_all.csv', image_size=[config.imsize, config.imsize], transforms_data=True, aug = False, phase = 'test')
    eval_data_loader = DataLoader(dataset = eval_dataset, batch_size = 32, shuffle = False, num_workers=2, pin_memory=False)

    if config.train:
        trainer = Trainer(data_loader, eval_data_loader, config, device)
        trainer.train()
    else:
        tester = Tester(data_loader, config, device)
        tester.test()
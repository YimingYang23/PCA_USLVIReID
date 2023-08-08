from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F





def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx 
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class ClusterContrastTrainer_pretrain(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data(inputs_ir)
            inputs_rgb, labels_rgb, indexes_rgb = self._parse_data(inputs_rgb)
            # forward
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            # f_out_rgb = self._forward(inputs_rgb)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)

            # loss_tri_rgb, batch_acc = self.tri(f_out_rgb, labels_rgb)
            # loss_tri_ir, batch_acc = self.tri(f_out_ir, labels_ir)
            # loss_tri = loss_tri_rgb+loss_tri_ir
            loss_ir = self.memory_ir(f_out_ir, labels_ir)# + loss_tri
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss = loss_ir+loss_rgb#+loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)


class ClusterContrastTrainer_intra_modal(object):
    def __init__(self, encoder, net_modal_classifer,memory=None):
        super(ClusterContrastTrainer_intra_modal, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.memory_all = memory
        self.net_modal_classifer = net_modal_classifer
        self.criterion1=nn.CrossEntropyLoss()
        self.criterion1.cuda()

    def train(self, epoch, data_loader_ir,data_loader_rgb, data_loader_all_ir, data_loader_all_rgb, optimizer, modal_classifier_optimizer, print_freq=10, train_iters=400, adv_flag=True):
        self.encoder.train()
        self.net_modal_classifer.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        modal_i_labels = Variable(torch.ones(64).long().cuda())
        modal_v_labels = Variable(torch.zeros(64).long().cuda())
        modal_g_labels = Variable(2 * torch.ones(64).long().cuda())
        modal_adv_loss=0
        true_label = torch.cat([modal_v_labels, modal_g_labels, modal_i_labels], dim=0)
        loss_all_ir=0
        loss_all_rgb=0
        loss = 0
        loss_ir=0
        loss_rgb=0
        modal_classifier_acc=0
        modal_loss=0

        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()


            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)

            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,pool_rgb,pool_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)

            if epoch<25:
                loss_ir = self.memory_ir(f_out_ir, labels_ir)
                loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
                loss = loss_ir+loss_rgb

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if adv_flag==True:
                    f_modal_in = torch.cat([f_out_rgb,f_out_ir], dim=0)
                    predit_modal_label=self.net_modal_classifer(f_modal_in.detach())


                    _,predict =predit_modal_label.max(1)
                    modal_classifier_acc= predict.eq(true_label).sum().item()/true_label.shape[0]


                    modal_loss=self.criterion1(predit_modal_label[:f_out_ir.shape[0]],modal_v_labels)\
                    +self.criterion1(predit_modal_label[f_out_ir.shape[0]:f_out_ir.shape[0]*2],modal_g_labels)\
                    +self.criterion1(predit_modal_label[f_out_ir.shape[0]*2:],modal_i_labels)

                    modal_classifier_optimizer.zero_grad()
                    modal_loss.backward()
                    modal_classifier_optimizer.step()

            elif epoch>=25:
                if adv_flag==True:
                    f_modal_in = torch.cat([f_out_rgb, f_out_ir], dim=0)
                    predit_modal_label = self.net_modal_classifer(f_modal_in.detach())

                    _,predict =predit_modal_label.max(1)
                    modal_classifier_acc= predict.eq(true_label).sum().item()/true_label.shape[0]

                    modal_loss = self.criterion1(predit_modal_label[:f_out_ir.shape[0]], modal_v_labels) \
                                 + self.criterion1(predit_modal_label[f_out_ir.shape[0]:f_out_ir.shape[0] * 2],
                                                   modal_g_labels) \
                                 + self.criterion1(predit_modal_label[f_out_ir.shape[0] * 2:], modal_i_labels)
                    #
                    modal_classifier_optimizer.zero_grad()
                    modal_loss.backward()
                    modal_classifier_optimizer.step()

                    predit_modal_label2 = self.net_modal_classifer(f_modal_in)
                    modal_adv_loss = self.criterion1(predit_modal_label2[:f_out_ir.shape[0]], modal_g_labels) \
                                 + self.criterion1(predit_modal_label2[f_out_ir.shape[0]:f_out_ir.shape[0] * 2],
                                                   modal_g_labels) \
                                 + self.criterion1(predit_modal_label2[f_out_ir.shape[0] * 2:], modal_g_labels)

                    loss_ir = self.memory_ir(f_out_ir, labels_ir)
                    loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
                    loss = loss_ir + loss_rgb + modal_adv_loss
                else:
                    loss_ir = self.memory_ir(f_out_ir, labels_ir)
                    loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
                    loss = loss_ir + loss_rgb

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.update(loss.item())
            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                        'Loss ir all {:.3f}\t'
                      'Loss rgb all {:.3f}\t'
                      'Loss modal {:.3f}\t'
                      'Loss modal adv {:.3f}\t'
                      ' modal class acc {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb,loss_all_ir,loss_all_rgb,modal_loss,modal_adv_loss,modal_classifier_acc))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)

class ClusterContrastTrainer_inter_modal(object):
    def __init__(self, encoder, net_modal_classifer,memory=None):
        super(ClusterContrastTrainer_inter_modal, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.memory_all = memory
        self.net_modal_classifer = net_modal_classifer
        self.criterion1=nn.CrossEntropyLoss()
        self.criterion1.cuda()
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
    def train(self, epoch, data_loader_ir,data_loader_rgb, data_loader_all_ir, data_loader_all_rgb, optimizer, modal_classifier_optimizer, print_freq=10, train_iters=400, adv_flag=True):
        self.encoder.train()
        self.net_modal_classifer.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        modal_i_labels = Variable(torch.ones(64).long().cuda())
        modal_v_labels = Variable(torch.zeros(64).long().cuda())
        modal_g_labels = Variable(2 * torch.ones(64).long().cuda())
        modal_adv_loss=0
        true_label = torch.cat([modal_v_labels, modal_g_labels, modal_i_labels], dim=0)
        loss_all_ir=0
        loss_all_rgb=0
        loss = 0
        loss_ir=0
        loss_rgb=0
        modal_classifier_acc=0
        modal_loss=0

        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()


            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)




            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,pool_rgb,pool_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)

            if epoch < 100:
                if adv_flag==True:
                    f_modal_in = torch.cat([f_out_rgb, f_out_ir], dim=0)
                    predit_modal_label = self.net_modal_classifer(f_modal_in.detach())

                    _,predict =predit_modal_label.max(1)
                    modal_classifier_acc= predict.eq(true_label).sum().item()/true_label.shape[0]

                    modal_loss = self.criterion1(predit_modal_label[:f_out_ir.shape[0]], modal_v_labels) \
                                 + self.criterion1(predit_modal_label[f_out_ir.shape[0]:f_out_ir.shape[0] * 2],
                                                   modal_g_labels) \
                                 + self.criterion1(predit_modal_label[f_out_ir.shape[0] * 2:], modal_i_labels)

                    modal_classifier_optimizer.zero_grad()
                    modal_loss.backward()
                    modal_classifier_optimizer.step()

                    predit_modal_label2 = self.net_modal_classifer(f_modal_in)
                    modal_adv_loss = self.criterion1(predit_modal_label2[:f_out_ir.shape[0]], modal_g_labels) \
                                 + self.criterion1(predit_modal_label2[f_out_ir.shape[0]:f_out_ir.shape[0] * 2],
                                                   modal_g_labels) \
                                 + self.criterion1(predit_modal_label2[f_out_ir.shape[0] * 2:], modal_g_labels)
                    #
                    loss_ir = self.memory_ir(f_out_ir, labels_ir)
                    loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)


                    loss = loss_ir + loss_rgb + 0.1*modal_adv_loss
                else:
                    loss_ir = self.memory_ir(f_out_ir, labels_ir)
                    loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)

                    loss = loss_ir + loss_rgb

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.update(loss.item())

                inputs_all_ir = data_loader_all_ir.next()
                inputs_all_rgb = data_loader_all_rgb.next()
                # process inputs
                inputs_all_ir, labels_all_ir, indexes_all_ir = self._parse_data_ir(inputs_all_ir)
                inputs_all_rgb, inputs_all_rgb1, labels_all_rgb, indexes_all_rgb = self._parse_data_rgb(inputs_all_rgb)
                # forward
                inputs_all_rgb = torch.cat((inputs_all_rgb, inputs_all_rgb1), 0)
                labels_all_rgb = torch.cat((labels_all_rgb, labels_all_rgb), -1)

                _, f_out_all_rgb, f_out_all_ir, labels_all_rgb, labels_all_ir, pool_all_rgb, pool_all_ir = self._forward(inputs_all_rgb, inputs_all_ir,
                                                                                                 label_1=labels_all_rgb,
                                                                                                 label_2=labels_all_ir, modal=0)

                loss_all_ir = self.memory_all(f_out_all_ir, labels_all_ir)
                loss_all_rgb = self.memory_all(f_out_all_rgb, labels_all_rgb)

                loss2 = loss_all_ir + loss_all_rgb

                optimizer.zero_grad()
                loss2.backward()
                optimizer.step()

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                        'Loss ir all {:.3f}\t'
                      'Loss rgb all {:.3f}\t'
                      'Loss modal {:.3f}\t'
                      'Loss modal adv {:.3f}\t'
                      ' modal class acc {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb,loss_all_ir,loss_all_rgb,modal_loss,modal_adv_loss,modal_classifier_acc))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)




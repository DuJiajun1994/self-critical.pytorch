from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, log_prob_y, log_prob_s, seq, reward):
        log_prob_y = to_contiguous(log_prob_y).view(-1)
        log_prob_s = to_contiguous(log_prob_s).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - (log_prob_y + log_prob_s) * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.baseline = 0

    def forward(self, log_prob_y, log_prob_s, target, mask):
        # truncate to the same size
        sequence_length = log_prob_y.size(1)
        target = target[:, :sequence_length]
        mask = mask[:, :sequence_length]

        log_prob_y = log_prob_y.gather(2, target.unsqueeze(2)).squeeze(2)
        reward = (log_prob_y * mask).sum(1) / mask.sum(1)

        loss = -(log_prob_s * (reward.detach() - self.baseline).unsqueeze(1) + log_prob_y)
        loss = (loss * mask).sum() / mask.sum()
        update = reward.mean().data.item()
        self.baseline = self.baseline * 0.9 + update * 0.1
        print('update {}, baseline {}'.format(update, self.baseline))

        cross_entropy_loss = (- log_prob_y * mask).sum() / mask.sum()
        return loss, cross_entropy_loss

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))
    
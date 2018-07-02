import torch
from torch.autograd import Variable
import scipy.io as sio
import numpy as np
import spd_net_util as util


class SPDNetwork(torch.nn.Module):

    def __init__(self):
        super(SPDNetwork, self).__init__()
        tmp = sio.loadmat('./tmp/afew/w_1.mat')['w_1']
        self.w_1_p = Variable(torch.from_numpy(tmp), requires_grad=True)

        tmp = sio.loadmat('./tmp/afew/w_2.mat')['w_2']
        self.w_2_p = Variable(torch.from_numpy(tmp), requires_grad=True)

        tmp = sio.loadmat('./tmp/afew/w_3.mat')['w_3']
        self.w_3_p = Variable(torch.from_numpy(tmp), requires_grad=True)

        tmp = sio.loadmat('./tmp/afew/fc.mat')['theta']
        self.fc_w = Variable(torch.from_numpy(tmp.astype(np.float64)), requires_grad=True)

    def forward(self, input):
        batch_size = input.shape[0]
        w_1_pc = self.w_1_p.contiguous()
        w_1 = w_1_pc.view([1, w_1_pc.shape[0], w_1_pc.shape[1]])

        w_2_pc = self.w_2_p.contiguous()
        w_2 = w_2_pc.view([1,w_2_pc.shape[0], w_2_pc.shape[1]])

        w_3_pc = self.w_3_p.contiguous()
        w_3 = w_3_pc.view([1, w_3_pc.shape[0], w_3_pc.shape[1]])

        w_tX = torch.matmul(torch.transpose(w_1, dim0=1, dim1=2), input)
        w_tXw = torch.matmul(w_tX, w_1)
        X_1 = util.rec_mat_v2(w_tXw)

        w_tX = torch.matmul(torch.transpose(w_2, dim0=1, dim1=2), X_1)
        w_tXw = torch.matmul(w_tX, w_2)
        X_2 = util.rec_mat_v2(w_tXw)

        w_tX = torch.matmul(torch.transpose(w_3, dim0=1, dim1=2), X_2)
        w_tXw = torch.matmul(w_tX, w_3)
        X_3 = util.log_mat_v2(w_tXw)

        feat = X_3.view([batch_size, -1])  # [batch_size, d]
        logits = torch.matmul(feat, self.fc_w)  # [batch_size, num_class]

        return logits

    def update_para(self, lr):

        egrad_w1 = self.w_1_p.grad.data.numpy()
        egrad_w2 = self.w_2_p.grad.data.numpy()
        egrad_w3 = self.w_3_p.grad.data.numpy()
        w_1_np = self.w_1_p.data.numpy()
        w_2_np = self.w_2_p.data.numpy()
        w_3_np = self.w_3_p.data.numpy()

        new_w_3 = util.update_para_riemann(w_3_np, egrad_w3, lr)
        new_w_2 = util.update_para_riemann(w_2_np, egrad_w2, lr)
        new_w_1 = util.update_para_riemann(w_1_np, egrad_w1, lr)

        # print(np.sum(w_1_np))
        # print(np.sum(np.square(w_3_np - new_w_3)))
        # print(np.sum(np.square(w_2_np - new_w_2)))
        # print(np.sum(np.square(w_1_np - new_w_1)))

        self.w_1_p.data.copy_(torch.DoubleTensor(new_w_1))
        self.w_2_p.data.copy_(torch.DoubleTensor(new_w_2))
        self.w_3_p.data.copy_(torch.DoubleTensor(new_w_3))

        self.fc_w.data -= lr * self.fc_w.grad.data
        # Manually zero the gradients after updating weights
        self.w_1_p.grad.data.zero_()
        self.w_2_p.grad.data.zero_()
        self.w_3_p.grad.data.zero_()
        self.fc_w.grad.data.zero_()
        # print('finished')
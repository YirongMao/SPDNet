import numpy as np
import random
import h5py
import os
import model
from torch.autograd import Variable
import torch
import datetime
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio


batch_size = 30
lr = 0.01
num_epoch = 200

data_dir = './data/afew/afew_400'
train_path = './data/afew/train.txt'
val_path = './data/afew/val.txt'
fid = open(train_path, 'r')
train_file = []
train_label = []
for line in fid.readlines():
    file, label = line.strip('\n').split(' ')
    file = file.replace('\\', '/')
    train_file.append(file)
    train_label.append(label)

fid = open(val_path, 'r')
val_file = []
val_label = []
for line in fid.readlines():
    file, label = line.strip('\n').split(' ')
    file = file.replace('\\', '/')
    val_file.append(file)
    val_label.append(label)

model = model.SPDNetwork()
hist_loss = []
for epoch in range(num_epoch):
    shuffled_index = list(range(len(train_file)))
    random.seed(6666)
    random.shuffle(shuffled_index)

    train_file = [train_file[i] for i in shuffled_index]
    train_label = [train_label[i] for i in shuffled_index]

    for idx_train in range(0, len(train_file) // batch_size):
        idx = idx_train
        b_file = train_file[idx * batch_size:(idx + 1) * batch_size]
        b_label = train_label[idx * batch_size:(idx + 1) * batch_size]
        batch_data = np.zeros([batch_size, 400, 400], dtype=np.float32)
        batch_label = np.zeros([batch_size], dtype=np.int32)
        i = 0
        for file in b_file:
            # f = h5py.File(os.path.join(data_dir, file), 'r')
            spd = sio.loadmat(os.path.join(data_dir, file))['Y1']
            batch_data[i, :, :] = spd
            batch_label[i] = int(b_label[i]) - 1
            i += 1
        input = Variable(torch.from_numpy(batch_data)).double()
        target = Variable(torch.LongTensor(batch_label))

        stime = datetime.datetime.now()
        logits = model(input)
        output = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(output, target)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
        loss.backward()
        model.update_para(lr)
        etime = datetime.datetime.now()
        dtime = etime.second - stime.second

        hist_loss.append(loss.data[0])
        print('[epoch %d/%d] [iter %d/%d] loss %f acc %f %f/batch' % (epoch, num_epoch,
                                                        idx_train, len(train_file) // batch_size, loss.data[0],
                                                         correct / batch_size, dtime))
    # end of one epoch

    if not os.path.exists('./tmp/afew'):
        os.makedirs('./tmp/afew')
    plt.plot(list(range(len(hist_loss))), hist_loss)
    torch.save(model, './tmp/afew/spdnet_' + str(epoch + 1) + 'c.model')
    plt.savefig('./tmp/afew/loss_c.jpg')
    plt.close()





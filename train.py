import torch
import torch.optim as optim
import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import dataLoader
import numpy as np
import argparse

## Pytorch MNIST CLASSIFICATION ##
## Implemented By Zhengqin Li ##
## You can do whatever you want to the code ##


parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', default = './', help = 'The path to the dataset')
parser.add_argument('--trainRoot', default=None, help = 'The path to save the model')
parser.add_argument('--cuda', action = 'store_true', help = 'Use gpu for training')
parser.add_argument('--gpuId', type=int, default = 0, help = 'The device you want to use for training')
parser.add_argument('--batchSize', type=int, default = 100, help = 'Decide the batch size')
parser.add_argument('--nepoch', type=int, default = 15, help = 'The number of epochs you are going to train the network')
parser.add_argument('--learning_rate', type=float, default =0.001, help = 'The learning rate of the optimizer')
parser.add_argument('--momentum', type=float, default=0.99, help='The momentum for SGD optimizer')
parser.add_argument('--isDropOut', action = 'store_true', help = 'Whether to use drop out layer or not')
parser.add_argument('--optimizer', default = 'ADAM', help ='The type of optimizer used for training network, you are required to try ADAM and SGD in this homework')
opt = parser.parse_args()

if opt.trainRoot is None:
    opt.trainRoot = "check_" + opt.optimizer
    if opt.isDropOut:
        opt.trainRoot += '_dropout'
os.system('mkdir {0}'.format(opt.trainRoot) )

# Hyper parameter
imSize = 28

# data batch
imBatch = Variable(torch.FloatTensor(opt.batchSize, 1, imSize, imSize) )
labelBatch = Variable(torch.LongTensor(opt.batchSize, 1) )

# Network
classifier = models.classifier(isDropOut = opt.isDropOut)

# Optimizer
if opt.optimizer == 'ADAM':
    optimizer = optim.Adam(classifier.parameters(), lr = opt.learning_rate)
elif opt.optimizer == 'SGD':
    optimizer = optim.SGD(classifier.parameters(), lr = opt.learning_rate, momentum = 0.99)
else:
    raise ValueError('You are only required to consider SGD and ADAM optimizer')

# DataLoader
mnistDataset = dataLoader.BatchLoader(dataRoot = opt.dataRoot, phase = 'TRAIN')
mnistLoader = DataLoader(mnistDataset, batch_size = opt.batchSize, num_workers = 4, shuffle = False)

# Move data and network to gpu
if opt.cuda:
    imBatch = imBatch.cuda(opt.gpuId)
    labelBatch = labelBatch.cuda(opt.gpuId)
    classifier = classifier.cuda(opt.gpuId)

# train
j = 0
errorList = []
for epoch in list(range(0, opt.nepoch) ):
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.trainRoot, epoch), 'w')
    for i, dataBatch in enumerate(mnistLoader):
        j += 1

        # Load the data
        im_cpu = dataBatch['img']
        imBatch.data.resize_(im_cpu.size() )
        imBatch.data.copy_(im_cpu)
        label_cpu = dataBatch['label']
        labelBatch.data.resize_(label_cpu.size() )
        labelBatch.data.copy_(label_cpu)

        # Train the network
        optimizer.zero_grad()
        predict = classifier(imBatch )
        ## WRITE YOUR CODE HERE, Define the loss function ##
        error = F.cross_entropy(predict, optimizer)

        optimizer.step()

        errorList.append(error.cpu().data[0] )
        print('[%d][%d] Error: %.6f' % (epoch, j, error.cpu().data[0] ))
        trainingLog.write('[%d][%d] Error: %.6f \n' % (epoch, j, error.cpu().data[0] ))
        if j < 1000:
            print('[%d][%d] Accumulated Error: %.6f' % (epoch, j, sum(errorList) / j ) )
            trainingLog.write('[%d][%d] Accumulated Error: %.6f \n' % (epoch, j, sum(errorList) / j ) )
        else:
            print('[%d][%d] Accumulated Error: %.6f' % (epoch, j, sum(errorList[j-1000:j]) / 1000 ) )
            trainingLog.write('[%d][%d] Accumulated Error: %.6f \n' % (epoch, j, sum(errorList[j-1000:j]) / 1000 ) )

    trainingLog.close()

    if (epoch+1) % 5 == 0:
        for parameters in optimizer.param_groups:
            parameters['lr'] /= 2

    np.save('{0}/error_{1}.npy'.format(opt.trainRoot, epoch), np.array(errorList) )
    torch.save(classifier.state_dict(), '{0}/classifier_{1}.pth'.format(opt.trainRoot, epoch) )

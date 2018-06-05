import torch
import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import dataLoader
import argparse

## Pytorch MNIST CLASSIFICATION ##
## Implemented By Zhengqin Li ##
## You can do whatever you want to the code ##


parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', default = './', help = 'The path to the dataset')
parser.add_argument('--trainRoot', default=None, help = 'The path to save the model')
parser.add_argument('--testRoot', default=None, help = 'The path to load the model')
parser.add_argument('--cuda', action = 'store_true', help = 'Use gpu for training')
parser.add_argument('--gpuId', type=int, default = 0, help = 'The device you want to use for training')
parser.add_argument('--batchSize', type=int, default = 1, help = 'Decide the batch size')
parser.add_argument('--nepoch', type=int, default = 15, help = 'The number of epochs you are going to train the network')
parser.add_argument('--isDropOut', action = 'store_true', help = 'Whether to use drop out layer or not')
parser.add_argument('--optimizer', default = 'ADAM', help ='The type of optimizer used for training network, you are required to try ADAM and SGD in this homework')
opt = parser.parse_args()

if opt.trainRoot is None:
    opt.trainRoot = "check_" + opt.optimizer
    if opt.isDropOut:
        opt.trainRoot += '_dropout'
if opt.testRoot is None:
    opt.testRoot = opt.trainRoot.replace('check', 'test')
os.system('mkdir {0}'.format(opt.testRoot) )

# Hyper parameter
imSize = 28

# data batch
imBatch = Variable(torch.FloatTensor(opt.batchSize, 1, imSize, imSize) )
labelBatch = Variable(torch.LongTensor(opt.batchSize, 1) )

# Network
classifier = models.classifier(isDropOut = opt.isDropOut).eval()
classifier.load_state_dict(torch.load('{0}/classifier_{1}.pth'.format(opt.trainRoot, opt.nepoch-1) ) )


# DataLoader
mnistDataset = dataLoader.BatchLoader(dataRoot = opt.dataRoot, phase = 'TEST')
mnistLoader = DataLoader(mnistDataset, batch_size = opt.batchSize, num_workers = 4, shuffle = False)

# Move data and network to gpu
if opt.cuda:
    imBatch = imBatch.cuda(opt.gpuId)
    labelBatch = labelBatch.cuda(opt.gpuId)
    classifier = classifier.cuda(opt.gpuId)

# train
j = 0
errorList = []
testingLog = open('{0}/testingLog_{1}.txt'.format(opt.testRoot, opt.nepoch-1), 'w')
correctNum = 0
epoch = opt.nepoch - 1
for i, dataBatch in enumerate(mnistLoader):
    j += 1

    # Load the data
    im_cpu = dataBatch['img']
    imBatch.data.resize_(im_cpu.size() )
    imBatch.data.copy_(im_cpu)
    label_cpu = dataBatch['label']
    labelBatch.data.resize_(label_cpu.size() )
    labelBatch.data.copy_(label_cpu)

    predict = classifier(imBatch )
    _, predictLabel = torch.max(predict, dim=1)
    ## WRITE YOU CODE HERE, Define the loss function##

    correctNum += torch.sum(predictLabel == labelBatch).cpu().data[0]

    errorList.append(error.cpu().data[0] )
    print('[%d][%d] Error: %.6f' % (epoch, j, error.cpu().data[0] ))
    print('[%d][%d] Accuracy: %.6f' % (epoch, j, float(correctNum) / float(j*opt.batchSize) ) )
    testingLog.write('[%d][%d] Error: %.6f \n' % (epoch, j, error.cpu().data[0] ))
    testingLog.write('[%d][%d] Accuracy: %.6f \n' % (epoch, j, float(correctNum) / float(j*opt.batchSize) ) )
    print('[%d][%d] Accumulated Error: %.6f' % (epoch, j, sum(errorList) / j ) )
    testingLog.write('[%d][%d] Accumulated Error: %.6f \n' % (epoch, j, sum(errorList) / j ) )

testingLog.close()

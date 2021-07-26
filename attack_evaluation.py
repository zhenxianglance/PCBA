from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch.utils.data
from dataset.dataset import ModelNetDataset
from model.pointnet import PointNetCls
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=1024, help='number of points')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--attack_dir', type=str, default='attack', help='attack folder')
parser.add_argument(
    '--model', type=str, default='./model_attacked/model.pth', help='model path')
parser.add_argument(
    '--dataset', type=str, default='modelnet40', help="dataset path")
parser.add_argument(
    '--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

testset = ModelNetDataset(
    root=opt.dataset,
    split='test',
    sub_sampling=True,
    npoints=opt.num_points,
    data_augmentation=False)

testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

num_classes = len(testset.classes)
print('classes: {}'.format(num_classes))

# Load backdoor test images
attack_data_test = np.load(os.path.join(opt.attack_dir, 'attack_data_test.npy'))
attack_labels_test = np.load(os.path.join(opt.attack_dir, 'attack_labels_test.npy'))
attack_testset = ModelNetDataset(
    root=opt.dataset,
    split='test',
    sub_sampling=True,
    npoints=opt.num_points,
    data_augmentation=False)
attack_testset.data = attack_data_test
attack_testset.labels = attack_labels_test


attack_testloader = torch.utils.data.DataLoader(
        attack_testset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)
classifier.load_state_dict(torch.load(opt.model))
classifier.to(device)

total_correct = 0
total_testset = 0
with torch.no_grad():
    for i, (points, targets) in tqdm(enumerate(testloader)):
        points = points.transpose(2, 1)
        points, targets = points.to(device), targets.to(device)
        classifier = classifier.eval()
        pred, _, _, _, _, _, = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]
    print("accuracy {}".format(total_correct / float(total_testset)))

total_correct = 0
total_testset = 0
with torch.no_grad():
    for i, (points, targets) in tqdm(enumerate(attack_testloader)):
        points = points.transpose(2, 1)
        points, targets = points.to(device), targets.to(device)
        classifier = classifier.eval()
        pred, _, _, _, _, _, = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]
    print("attack success rate {}".format(total_correct / float(total_testset)))

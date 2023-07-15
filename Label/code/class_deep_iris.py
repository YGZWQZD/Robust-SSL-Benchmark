from xgboost import XGBClassifier
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Transform.ToTensor import ToTensor
from LAMDA_SSL.Dataloader.LabeledDataloader import LabeledDataLoader
from LAMDA_SSL.Dataloader.UnlabeledDataloader import UnlabeledDataLoader
from LAMDA_SSL.Sampler.RandomSampler import RandomSampler
from LAMDA_SSL.Sampler.SequentialSampler import SequentialSampler
from LAMDA_SSL.Augmentation.Tabular.Noise import Noise
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Algorithm.Classification.Supervised import Supervised
from LAMDA_SSL.Network.FT_Transformer import FT_Transformer
from LAMDA_SSL.Algorithm.Classification.PiModel import PiModel
from LAMDA_SSL.Algorithm.Classification.TemporalEnsembling import TemporalEnsembling
from LAMDA_SSL.Algorithm.Classification.UDA import UDA
from LAMDA_SSL.Algorithm.Classification.PseudoLabel import PseudoLabel
from LAMDA_SSL.Algorithm.Classification.ImprovedGAN import ImprovedGAN
from LAMDA_SSL.Algorithm.Classification.ICT import ICT
from LAMDA_SSL.Algorithm.Classification.MixMatch import MixMatch
from LAMDA_SSL.Algorithm.Classification.FixMatch import FixMatch
from LAMDA_SSL.Algorithm.Classification.FlexMatch import FlexMatch
from LAMDA_SSL.Opitimizer.Adam import Adam
from sklearn.datasets import load_iris
import torch
import random
import os
import argparse
import csv
import pickle
from LAMDA_SSL.Split.DataSplit import DataSplit
import copy
from sklearn.preprocessing import StandardScaler
import numpy as np
from math import ceil
from LAMDA_SSL.Algorithm.Classification.LabelPropagation import LabelPropagation
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
parser = argparse.ArgumentParser()
parser.add_argument('--labels', type=int, default=50)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()
labels=args.labels
device=args.device

def distribution_selection_condition(X,y,p=0.2,interval=0.2,num_classes=3):
    source_X_list=[]
    target_X_list=[]
    source_y_list=[]
    target_y_list=[]
    for _ in range(num_classes):
        _X = X[y==_]
        _y = y[y==_]
        mean_X=np.mean(_X,axis=0)
        dis=[]
        for _ in range(_X.shape[0]):
            s = np.linalg.norm(mean_X - _X[_])
            dis.append(s)
        dis=np.array(dis)
        index=dis.argsort()
        source_index, target_index = index[:ceil(index.shape[0] * 0.5)], index[ceil(index.shape[0] * 0.5):]
        source_X_list.append(_X[source_index])
        source_y_list.append(_y[source_index])
        target_X_list.append(_X[target_index[ceil(target_index.shape[0] * (p-interval)):ceil(target_index.shape[0] * p)]])
        target_y_list.append(_y[target_index[ceil(target_index.shape[0] * (p-interval)):ceil(target_index.shape[0] * p)]])
    source_X = np.concatenate((source_X_list))
    source_y = np.concatenate((source_y_list))
    target_X = np.concatenate((target_X_list))
    target_y = np.concatenate((target_y_list))
    return source_X, source_y,target_X,target_y

evaluation= Accuracy()

labeled_dataset=LabeledDataset(transform=ToTensor())
unlabeled_dataset=UnlabeledDataset(transform=ToTensor())
test_dataset=UnlabeledDataset(transform=ToTensor())

labeled_dataloader=LabeledDataLoader(batch_size=64,num_workers=0,drop_last=True)
unlabeled_dataloader=UnlabeledDataLoader(num_workers=0,drop_last=True)
test_dataloader=UnlabeledDataLoader(batch_size=64,num_workers=0,drop_last=False)

labeled_sampler=RandomSampler(replacement=True,num_samples=64*(10000))
unlabeled_sampler=RandomSampler(replacement=True)
test_sampler=SequentialSampler()

weak_augmentation=Noise(0.1)
strong_augmentation=Noise(0.2)
augmentation={
    'weak_augmentation':weak_augmentation,
    'strong_augmentation':strong_augmentation
}

rate_list=[0,0.2,0.4,0.6,0.8,1.0]
f=open("iris"+"_deep"+'_class'+'_labels_'+str(labels)+'.csv', "w", encoding="utf-8")
r = csv.DictWriter(f,['algorithm','rate','mean','std'])
X,y=load_iris(return_X_y=True)
num_classes=len(np.unique(y))

algorithms = {
    'FT_Transformer': Supervised(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
                                 test_dataset=test_dataset, device=device, augmentation=None,
                                 network=FT_Transformer(dim_in=X.shape[1], num_classes=(num_classes+1)//2),
                                 # network=MLPCLS(num_classes=2, dim_in=X.shape[1]),
                                 num_it_epoch=1000, labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
                                 scheduler=None, weight_decay=1e-5),

    'ImprovedGAN': ImprovedGAN(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
                               test_dataset=test_dataset, device=device, num_it_epoch=100, epoch=10,
                               labeled_sampler=RandomSampler(replacement=True, num_samples=64 * (100)),
                               optimizer=Adam(lr=1e-4),
                               scheduler=None, weight_decay=1e-5),

    'PiModel': PiModel(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
                       test_dataset=test_dataset, device=device, augmentation=weak_augmentation,
                       network=FT_Transformer(dim_in=X.shape[1], num_classes=(num_classes+1)//2),
                       # network=MLPCLS(num_classes=2, dim_in=X.shape[1]),
                       num_it_epoch=1000, labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
                       scheduler=None, weight_decay=1e-5),
    'TemporalEnsembling': TemporalEnsembling(labeled_dataset=labeled_dataset,
                                             unlabeled_dataset=unlabeled_dataset, test_dataset=test_dataset,
                                             device=device, augmentation=weak_augmentation,
                                             network=FT_Transformer(dim_in=X.shape[1], num_classes=(num_classes+1)//2),
                                             # network=MLPCLS(num_classes=2,dim_in=X.shape[1]),
                                             num_it_epoch=100, epoch=10,
                                             labeled_sampler=RandomSampler(replacement=True, num_samples=64 * (100)),
                                             optimizer=Adam(lr=1e-4),
                                             scheduler=None, weight_decay=1e-5),
    'UDA': UDA(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset, test_dataset=test_dataset,
               device=device, augmentation=augmentation, network=FT_Transformer(dim_in=X.shape[1], num_classes=(num_classes+1)//2),
               num_it_epoch=1000, labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
               scheduler=None, weight_decay=1e-5),
    'PseudoLabel': PseudoLabel(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
                               test_dataset=test_dataset, device=device, augmentation=weak_augmentation,
                               network=FT_Transformer(dim_in=X.shape[1], num_classes=(num_classes+1)//2),
                               num_it_epoch=1000, labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
                               scheduler=None, weight_decay=1e-5),
    'ICT': ICT(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset, test_dataset=test_dataset,
               device=device, augmentation=weak_augmentation, network=FT_Transformer(dim_in=X.shape[1], num_classes=(num_classes+1)//2),
               num_it_epoch=1000, labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
               scheduler=None, weight_decay=1e-5),
    'MixMatch': MixMatch(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
                         test_dataset=test_dataset, device=device, augmentation=weak_augmentation,
                         network=FT_Transformer(dim_in=X.shape[1], num_classes=(num_classes+1)//2), num_it_epoch=1000,
                         labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
                         scheduler=None, weight_decay=1e-5),
    'FixMatch': FixMatch(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
                         test_dataset=test_dataset, device=device, augmentation=augmentation,
                         network=FT_Transformer(dim_in=X.shape[1], num_classes=(num_classes+1)//2), num_it_epoch=1000,
                         labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
                         scheduler=None, weight_decay=1e-5),
    'FlexMatch': FlexMatch(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
                           test_dataset=test_dataset, device=device, augmentation=augmentation,
                           network=FT_Transformer(dim_in=X.shape[1], num_classes=(num_classes+1)//2), num_it_epoch=1000,
                           labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
                           scheduler=None, weight_decay=1e-5)
}

for name,algorithm in algorithms.items():
    for rate in rate_list:
        performance_list = []
        performance_list_r = []
        for _ in range(5):
            set_seed(_)
            source_X, source_y, target_X, target_y = DataSplit(stratified=True, shuffle=True, random_state=_, X=X, y=y, size_split=0.5)
            source_X,source_y = source_X[source_y<num_classes/2],source_y[source_y<num_classes/2]
            labeled_X, labeled_y, test_X, test_y = DataSplit(stratified=True, shuffle=True, random_state=_, X=source_X, y=source_y, size_split=labels)
            target_X_in, target_y_in = target_X[target_y<num_classes/2], target_y[target_y<num_classes/2]
            target_X_out, target_y_out = target_X[target_y>=num_classes/2], target_y[target_y>=num_classes/2]
            if rate == 0:
                unlabels = target_X_in.shape[0]
            else:
                unlabels = min(target_X_in.shape[0] / (1 - rate), target_X_out.shape[0] / rate) if rate != 1 else \
                target_X_out.shape[0]
            target_X_in, target_y_in, target_X_in_r, target_y_in_r = DataSplit(stratified=True, shuffle=True, random_state=_, X=target_X_in, y=target_y_in, size_split=int(unlabels*(1-rate)))
            target_X_out, target_y_out, target_X_out_r, target_y_out_r  = DataSplit(stratified=True, shuffle=True, random_state=_, X=target_X_out, y=target_y_out, size_split=int(unlabels * rate))
            unlabeled_X=np.concatenate((target_X_in, target_X_out),axis=0)
            unlabeled_y=np.concatenate((target_y_in, target_y_out),axis=0)
            trans = StandardScaler().fit(labeled_X)
            test_X = trans.transform(test_X)
            labeled_X=trans.transform(labeled_X)
            trans = StandardScaler().fit(unlabeled_X)
            unlabeled_X = trans.transform(unlabeled_X)
            algorithm_1=copy.deepcopy(algorithm)
            pred_y = algorithm_1.fit(labeled_X, labeled_y, unlabeled_X).predict(test_X)
            performance = Accuracy().scoring(test_y, pred_y)
            performance_list.append(performance)
        performance_list=np.array(performance_list)
        mean=performance_list.mean()
        std=performance_list.std()
        d={}
        d['algorithm']=name
        d['mean']=mean
        d['std']=std
        d['rate']=rate
        r.writerow(d)
f.close()

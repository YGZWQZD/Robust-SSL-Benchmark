# import sys; print('Python %s on %s' % (sys.version, sys.platform))
# sys.path.extend(['C:\\JiaLH\\project\\LAMDA-SSL\\LAMDA-SSL', 'C:/JiaLH/project/LAMDA-SSL/LAMDA-SSL'])
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
# from LAMDA_SSL.Algorithm.Classification.FlexMatch import FlexMatch
from LAMDA_SSL.Algorithm.Classification.SoftMatch import SoftMatch
from LAMDA_SSL.Algorithm.Classification.ReMixMatch import ReMixMatch
from LAMDA_SSL.Algorithm.Classification.SoftMatch import SoftMatch
from LAMDA_SSL.Opitimizer.Adam import Adam
from sklearn.datasets import load_wine,load_iris
import torch
import random
from sklearn.utils import check_random_state
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
# p_y=[0.9,0,1]
# p_y=[0.95,0.85,0.75,0.65,0.55,0.45,0.35,0.25,0.15,0.05]
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
parser.add_argument('--labels', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

labels=args.labels
device=args.device

def distribution_selection_label(X,y,p=0.2,random_state=None):
    num_classes=len(np.unique(y))
    X_list=[]
    y_list=[]
    for i in range(num_classes):
        p_y=1-i*p/num_classes
        _X = X[y==i]
        _y = y[y==i]
        rng = check_random_state(seed=random_state)
        permutation = rng.permutation(_X.shape[0])
        X_list.append(_X[permutation[:ceil(_X.shape[0]*p_y)]])
        y_list.append(_y[permutation[:ceil(_y.shape[0] * p_y)]])

    source_X = np.concatenate((X_list))
    source_y = np.concatenate((y_list))
    # r_s=random.sample(list(range(X.shape[0])),s_X.shape[0])
    return source_X,source_y

def distribution_selection_condition(X,y,p=0.2,interval=0.2,random_state=None):
    num_classes=len(np.unique(y))
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
        if p==0:
            rng = check_random_state(seed=random_state)
            permutation = rng.permutation(source_index.shape[0])
            target_X_list.append(_X[source_index[permutation[:ceil(target_index.shape[0] * interval)]]])
            target_y_list.append(_y[source_index[permutation[:ceil(target_index.shape[0] * interval)]]])
        else:
            target_X_list.append(_X[target_index[ceil(target_index.shape[0] * (p-interval)):ceil(target_index.shape[0] * p)]])
            target_y_list.append(_y[target_index[ceil(target_index.shape[0] * (p-interval)):ceil(target_index.shape[0] * p)]])
    source_X = np.concatenate((source_X_list))
    source_y = np.concatenate((source_y_list))
    target_X = np.concatenate((target_X_list))
    target_y = np.concatenate((target_y_list))
    return source_X, source_y,target_X,target_y

def distribution_selection_both(X,y,p=0.2,interval=0.2,random_state=None):
    source_X, source_y, target_X, target_y=distribution_selection_condition(X,y,p,interval,random_state)
    target_X,target_y=distribution_selection_label(target_X,target_y,p,random_state)
    return source_X,source_y,target_X,target_y
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

path='../../data/numerical_only/balanced/'
rate_list=[0,0.2,0.4,0.6,0.8,1.0]
f=open("iris"+"_deep"+'_distribution'+'_labels_'+str(labels)+'_1.csv', "w", encoding="utf-8")
r = csv.DictWriter(f,['algorithm','rate','mean','std'])
X,y=load_iris(return_X_y=True)
num_classes=len(np.unique(y))
labels=labels*num_classes
algorithms = {
    # 'FT_Transformer': Supervised(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
    #                              test_dataset=test_dataset, device=device, augmentation=None,
    #                              network=FT_Transformer(dim_in=X.shape[1], num_classes=num_classes),
    #                              num_it_epoch=1000, labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
    #                              scheduler=None, weight_decay=1e-5),

    # 'ImprovedGAN': ImprovedGAN(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
    #                            test_dataset=test_dataset, device=device, num_it_epoch=100, epoch=10,
    #                            labeled_sampler=RandomSampler(replacement=True, num_samples=64 * (100)),
    #                            optimizer=Adam(lr=1e-4),
    #                            scheduler=None, weight_decay=1e-5),

    # 'PiModel': PiModel(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
    #                    test_dataset=test_dataset, device=device, augmentation=weak_augmentation,
    #                    network=FT_Transformer(dim_in=X.shape[1], num_classes=num_classes),
    #                    # network=MLPCLS(num_classes=2, dim_in=X.shape[1]),
    #                    num_it_epoch=1000, labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
    #                    scheduler=None, weight_decay=1e-5),
    # 'TemporalEnsembling': TemporalEnsembling(labeled_dataset=labeled_dataset,
    #                                          unlabeled_dataset=unlabeled_dataset, test_dataset=test_dataset,
    #                                          device=device, augmentation=weak_augmentation,
    #                                          network=FT_Transformer(dim_in=X.shape[1], num_classes=num_classes),
    #                                          # network=MLPCLS(num_classes=2,dim_in=X.shape[1]),
    #                                          num_it_epoch=100, epoch=10,
    #                                          labeled_sampler=RandomSampler(replacement=True, num_samples=64 * (100)),
    #                                          optimizer=Adam(lr=1e-4),
    #                                          scheduler=None, weight_decay=1e-5),
    # 'UDA': UDA(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset, test_dataset=test_dataset,
    #            device=device, augmentation=augmentation, network=FT_Transformer(dim_in=X.shape[1], num_classes=num_classes),
    #            num_it_epoch=1000, labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
    #            scheduler=None, weight_decay=1e-5),
    # 'PseudoLabel': PseudoLabel(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
    #                            test_dataset=test_dataset, device=device, augmentation=weak_augmentation,
    #                            network=FT_Transformer(dim_in=X.shape[1], num_classes=num_classes),
    #                            num_it_epoch=1000, labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
    #                            scheduler=None, weight_decay=1e-5),
    # 'ICT': ICT(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset, test_dataset=test_dataset,
    #            device=device, augmentation=weak_augmentation, network=FT_Transformer(dim_in=X.shape[1], num_classes=num_classes),
    #            num_it_epoch=1000, labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
    #            scheduler=None, weight_decay=1e-5),
    # 'MixMatch': MixMatch(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
    #                      test_dataset=test_dataset, device=device, augmentation=weak_augmentation,
    #                      network=FT_Transformer(dim_in=X.shape[1], num_classes=num_classes), num_it_epoch=1000,
    #                      labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
    #                      scheduler=None, weight_decay=1e-5),
    # 'FixMatch': FixMatch(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
    #                      test_dataset=test_dataset, device=device, augmentation=augmentation,
    #                      network=FT_Transformer(dim_in=X.shape[1], num_classes=num_classes), num_it_epoch=1000,
    #                      labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
    #                      scheduler=None, weight_decay=1e-5),
    # 'FlexMatch': FlexMatch(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
    #                        test_dataset=test_dataset, device=device, augmentation=augmentation,
    #                        network=FT_Transformer(dim_in=X.shape[1], num_classes=num_classes), num_it_epoch=1000,
    #                        labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
    #                        scheduler=None, weight_decay=1e-5),
        'SoftMatch': SoftMatch(labeled_dataset=labeled_dataset, unlabeled_dataset=unlabeled_dataset,
                           test_dataset=test_dataset, use_DA=True,device=device, augmentation=augmentation,
                           network=FT_Transformer(dim_in=X.shape[1], num_classes=num_classes), num_it_epoch=1000,
                           labeled_sampler=labeled_sampler, optimizer=Adam(lr=1e-4),
                           scheduler=None, weight_decay=1e-5)
}
for name,algorithm in algorithms.items():
    print(name)
    for rate in rate_list:
        print(rate)
        performance_list = []
        performance_list_r = []
        for _ in range(5):
            set_seed(_)
            # _labeled_X, _labeled_y, unlabeled_X, unlabeled_y=distribution_selection(X,y,0.8,_)
            # test_X, test_y, labeled_X, labeled_y = DataSplit(stratified=True, shuffle=True,
            #                                              random_state=_, X=_labeled_X, y=_labeled_y, size_split=0.4)
            # trans = StandardScaler().fit(np.concatenate((labeled_X,unlabeled_X)))
            # test_X = trans.transform(test_X)
            # labeled_X=trans.transform(labeled_X)
            # unlabeled_X = trans.transform(unlabeled_X)
            source_X, source_y, target_X, target_y = distribution_selection_condition(X,y,p=rate,random_state=_)
            labeled_X, labeled_y, test_X, test_y = DataSplit(stratified=True, shuffle=True, random_state=_, X=source_X, y=source_y, size_split=labels)
            unlabeled_X,unlabeled_y=target_X,target_y
            # trans = StandardScaler().fit(train_X)
            # test_X = trans.transform(test_X)
            # train_X=trans.transform(train_X)
            #
            # labeled_X, labeled_y, unlabeled_X, unlabeled_y = DataSplit(stratified=True, shuffle=True,
            #                                                            random_state=_, X=_train_X,
            #                                                            y=_train_y, size_split=labels)
            # unlabeled_X, unlabeled_y = distribution_selection_condition(unlabeled_X, unlabeled_y, p=rate, seed=_)
            # unlabeled_X, unlabeled_y = random_selection(unlabeled_X, unlabeled_y)
            # print(unlabeled_X.shape)

            # print(unlabeled_X.shape)
            trans = StandardScaler().fit(labeled_X)
            test_X = trans.transform(test_X)
            labeled_X=trans.transform(labeled_X)
            trans = StandardScaler().fit(unlabeled_X)
            unlabeled_X = trans.transform(unlabeled_X)
            print(unlabeled_X.shape)
            algorithm_1=copy.deepcopy(algorithm)
            pred_y = algorithm_1.fit(labeled_X, labeled_y, unlabeled_X).predict(test_X)
            performance = Accuracy().scoring(test_y, pred_y)
            performance_list.append(performance)
            #performance_r = Accuracy().scoring(test_y, pred_y_1)
            #performance_list_r.append(performance_r)
            print(performance)
            #print(performance_r)
        performance_list=np.array(performance_list)
        mean=performance_list.mean()
        std=performance_list.std()
        d={}
        d['algorithm']=name
        d['mean']=mean
        d['std']=std
        d['rate']=rate
        print(d)
        r.writerow(d)
        f.flush()
        #performance_list_r=np.array(performance_list_r)
        #mean_r=performance_list_r.mean()
        #std_r=performance_list_r.std()
        #d={}
        #d['algorithm']=name+'random'
        #d['mean']=mean_r
        #d['std']=std_r
        #d['rate']=rate
        #print(d)
        #r.writerow(d)
f.close()

# print(dataset)
# print('end!')

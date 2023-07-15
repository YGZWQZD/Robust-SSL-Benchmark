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
from LAMDA_SSL.Algorithm.Classification.Tri_Training import Tri_Training
from LAMDA_SSL.Algorithm.Classification.Assemble import Assemble
from LAMDA_SSL.Algorithm.Classification.Co_Training import Co_Training
from LAMDA_SSL.Algorithm.Classification.LabelSpreading import LabelSpreading
from LAMDA_SSL.Algorithm.Classification.TSVM import TSVM
from LAMDA_SSL.Algorithm.Classification.SSGMM import SSGMM
from sklearn.datasets import load_wine
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
from sklearn.utils import check_random_state
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
parser.add_argument('--labels', type=int, default=30)
args = parser.parse_args()
labels=args.labels

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
f=open("wine"+"_statistical"+'_distribution'+'_labels_'+str(labels)+'.csv', "w", encoding="utf-8")
r = csv.DictWriter(f,['algorithm','rate','mean','std'])
X,y=load_wine(return_X_y=True)
num_classes=len(np.unique(y))
algorithms = {
              'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
              'SSGMM': SSGMM(num_classes=num_classes),
              'TSVM': TSVM(),
              'LabelPropagation': LabelPropagation(),
              'LabelSpreading': LabelSpreading(),
              'Co_Training': Co_Training(k=30,s=10, base_estimator=XGBClassifier(use_label_encoder=False,eval_metric='logloss')),
              'Tri_Training': Tri_Training(base_estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
              'Assemble': Assemble(T=30,base_estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
              }
Transductive = ['TSVM', 'LabelSpreading','LabelPropagation']
for name,algorithm in algorithms.items():
    for rate in rate_list:
        performance_list = []
        for _ in range(5):
            set_seed(_)
            source_X, source_y, target_X, target_y = distribution_selection_condition(X,y,p=rate,random_state=_)
            labeled_X, labeled_y, test_X, test_y = DataSplit(stratified=True, shuffle=True, random_state=_, X=source_X, y=source_y, size_split=labels)
            unlabeled_X,unlabeled_y=target_X,target_y
            trans = StandardScaler().fit(labeled_X)
            test_X = trans.transform(test_X)
            labeled_X=trans.transform(labeled_X)
            trans = StandardScaler().fit(unlabeled_X)
            unlabeled_X = trans.transform(unlabeled_X)

            if name is 'XGBClassifier':
                algorithm = algorithm.fit(labeled_X, labeled_y)
                pred_y = algorithm.predict(test_X)
            elif name in Transductive:
                algorithm_1=copy.deepcopy(algorithm)
                algorithm_1 = algorithm_1.fit(labeled_X, labeled_y, unlabeled_X)
                pred_y = algorithm_1.predict(test_X, Transductive=False)
            else:
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

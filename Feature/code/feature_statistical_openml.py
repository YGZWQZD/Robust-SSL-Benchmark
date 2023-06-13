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
from LAMDA_SSL.Algorithm.Classification.Tri_Training import Tri_Training
from LAMDA_SSL.Algorithm.Classification.Assemble import Assemble
from LAMDA_SSL.Algorithm.Classification.Co_Training import Co_Training
from LAMDA_SSL.Algorithm.Classification.LabelSpreading import LabelSpreading
from LAMDA_SSL.Algorithm.Classification.TSVM import TSVM
from sklearn.utils import check_random_state
from LAMDA_SSL.Algorithm.Classification.SSGMM import SSGMM
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine,fetch_openml
from sklearn.preprocessing import LabelEncoder
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
parser.add_argument('--labels', type=int, default=10)
# parser.add_argument('--unlabels', type=int, default=100)
args = parser.parse_args()


labels=args.labels
# unlabels=args.unlabels
# def distribution_selection_y(X,y,p):
#     r=np.random.random(X.shape[0])
#     s = []
#     # print(X.shape)
#     for _ in range(X.shape[0]):
#         if r[_]<=p_y[int(y[_])]:
#             s.append(1)
#         else:
#             s.append(0)
#     s=np.array(s)
#     s_X=X[s==1]
#     # print(s_X.shape)
#     s_y=y[s==1]
#     # r_s=random.sample(list(range(X.shape[0])),s_X.shape[0])
#     r_X=X[s==0]
#     r_y=y[s==0]
#     return s_X,s_y,r_X,r_y

# def distribution_selection_condition(X,y,p=0.2,interval=0.2,num_classes=3):
#     source_X_list=[]
#     target_X_list=[]
#     source_y_list=[]
#     target_y_list=[]
#     for _ in range(num_classes):
#         _X = X[y==_]
#         _y = y[y==_]
#         mean_X=np.mean(_X,axis=0)
#         dis=[]
#         for _ in range(_X.shape[0]):
#             s = np.linalg.norm(mean_X - _X[_])
#             dis.append(s)
#         dis=np.array(dis)
#         index=dis.argsort()
#         source_index, target_index = index[:ceil(index.shape[0] * 0.5)], index[ceil(index.shape[0] * 0.5):]
#         # print(source_index.shape)
#         # print(target_index.shape)
#         # print(target_index[ceil(target_index.shape[0] * (p-interval)):ceil(target_index.shape[0] * p)].shape)
#         # print(ceil(target_index.shape[0] * (p-interval)))
#         # print(ceil(target_index.shape[0] * p))
#         source_X_list.append(_X[source_index])
#         source_y_list.append(_y[source_index])
#         target_X_list.append(_X[target_index[ceil(target_index.shape[0] * (p-interval)):ceil(target_index.shape[0] * p)]])
#         target_y_list.append(_y[target_index[ceil(target_index.shape[0] * (p-interval)):ceil(target_index.shape[0] * p)]])
#     source_X = np.concatenate((source_X_list))
#     source_y = np.concatenate((source_y_list))
#     target_X = np.concatenate((target_X_list))
#     target_y = np.concatenate((target_y_list))
#     return source_X, source_y,target_X,target_y

# def distribution_selection_condition(X,y,p=0.2,seed=None,num_classes=2):
#     labeled_X_list=[]
#     unlabeled_X_list=[]
#     labeled_y_list=[]
#     unlabeled_y_list=[]
#     for _ in range(num_classes):
#         _X = X[y==_]
#         _y = y[y==_]
#         mean_X=np.mean(_X,axis=0)
#         dis=[]
#         for _ in range(_X.shape[0]):
#             s = np.linalg.norm(mean_X - _X[_])
#             dis.append(s)
#         dis=np.array(dis)
#         index=dis.argsort()
#         index_domain_1,index_domain_2=index[:ceil(index.shape[0]*0.5)],index[ceil(index.shape[0]*0.5):]
#         rng = check_random_state(seed=seed)
#         permutation = rng.permutation(ceil(index_domain_1.shape[0]))
#         index_domain_1_labeled, index_domain_1_unlabeled = index_domain_1[permutation[:ceil(index_domain_1.shape[0]*p)]],\
#                                                            index_domain_1[permutation[ceil(index_domain_1.shape[0]*p):]]
#         permutation = rng.permutation(ceil(index_domain_2.shape[0]))
#         index_domain_2_unlabeled, index_domain_2_labeled = index_domain_2[permutation[:ceil(index_domain_2.shape[0] * p)]], \
#                                                            index_domain_2[permutation[ceil(index_domain_2.shape[0] * p):]]
#         labeled_index= np.concatenate((index_domain_1_labeled,index_domain_2_labeled))
#         unlabeled_index=np.concatenate((index_domain_1_unlabeled, index_domain_2_unlabeled))
#         labeled_X_list.append(_X[labeled_index])
#         unlabeled_X_list.append(_X[unlabeled_index])
#         labeled_y_list.append(_y[labeled_index])
#         unlabeled_y_list.append(_y[unlabeled_index])
#     unlabeled_X = np.concatenate((unlabeled_X_list))
#     unlabeled_y = np.concatenate((unlabeled_y_list))
#     return unlabeled_X, unlabeled_y

# def distribution_selection_class(X,y,p=0.2,seed=None):
#     p_y=[p,1-p]
#     r=np.random.random(X.shape[0])
#     s = []
#     # print(X.shape)
#     for _ in range(X.shape[0]):
#         if r[_]<=p_y[int(y[_])]:
#             s.append(1)
#         else:
#             s.append(0)
#     s=np.array(s)
#     s_X=X[s==1]
#     # print(s_X.shape)
#     s_y=y[s==1]
#     # r_s=random.sample(list(range(X.shape[0])),s_X.shape[0])
#     return s_X,s_y

# def distribution_selection_both(X,y,p=0.2,seed=None):
#     X,y=distribution_selection_class(X,y,p=p)
#     X,y=distribution_selection_condition(X,y,p=p,seed=seed)
#     return X,y
#     # print(X.shape)
    # print(y.shape)
    # mean_X=np.mean(X,axis=0)
    # dis=[]
    # for _ in range(X.shape[0]):
    #     s = np.linalg.norm(mean_X - X[_])
    #     dis.append(s)
    # dis=np.array(dis)
    # index=dis.argsort()
    # index_domain_1,index_domain_2=index[:ceil(index.shape[0]*0.5)],index[ceil(index.shape[0]*0.5):]
    # rng = check_random_state(seed=random_state)
    # permutation = rng.permutation(ceil(index_domain_1.shape[0]))
    # index_domain_1_labeled, index_domain_1_unlabeled = index_domain_1[permutation[:ceil(index_domain_1.shape[0]*p)]],\
    #                                                    index_domain_1[permutation[ceil(index_domain_1.shape[0]*p):]]
    # permutation = rng.permutation(ceil(index_domain_2.shape[0]))
    # index_domain_2_unlabeled, index_domain_2_labeled = index_domain_2[permutation[:ceil(index_domain_2.shape[0] * p)]], \
    #                                                    index_domain_2[permutation[ceil(index_domain_2.shape[0] * p):]]
    # labeled_index = np.concatenate((index_domain_1_labeled,index_domain_2_labeled))
    # unlabeled_index = np.concatenate((index_domain_1_unlabeled, index_domain_2_unlabeled))
    # labeled_X, unlabeled_X = X[labeled_index], X[unlabeled_index]
    # labeled_y, unlabeled_y = y[labeled_index], y[unlabeled_index]
    # return labeled_X, labeled_y, unlabeled_X, unlabeled_y

def feature_selection(labeled_X,unlabeled_X,p=0.5,random_state=None):
    # print(X.shape)
    # print(y.shape)
    # mean_X=np.mean(X,axis=0)
    # dis=[]
    # for _ in range(X.shape[0]):
    #     s = np.linalg.norm(mean_X - X[_])
    #     dis.append(s)
    # dis=np.array(dis)
    # index=dis.argsort()
    # index_domain_1,index_domain_2=index[:ceil(index.shape[0]*0.5)],index[ceil(index.shape[0]*0.5):]
    # rng = check_random_state(seed=random_state)
    # permutation = rng.permutation(ceil(index_domain_1.shape[0]*0.5))
    # index_domain_1_labeled, index_domain_1_unlabeled = index_domain_1[permutation[:ceil(index_domain_1.shape[0]*p)]],\
    #                                                    index_domain_1[permutation[ceil(index_domain_1.shape[0]*p):]]
    # permutation = rng.permutation(ceil(index_domain_2.shape[0] * 0.5))
    # index_domain_2_unlabeled, index_domain_2_labeled = index_domain_2[permutation[:ceil(index_domain_2.shape[0] * p)]], \
    #                                                    index_domain_2[permutation[ceil(index_domain_2.shape[0] * p):]]
    # labeled_index = np.concatenate((index_domain_1_labeled,index_domain_2_labeled))
    # unlabeled_index = np.concatenate((index_domain_1_unlabeled, index_domain_2_unlabeled))
    # labeled_X, unlabeled_X = X[labeled_index], X[unlabeled_index]
    # labeled_y, unlabeled_y = y[labeled_index], y[unlabeled_index]
    rng = check_random_state(seed=random_state)
    permutation = rng.permutation(unlabeled_X.shape[1])
    mask_feature = permutation[:ceil(unlabeled_X.shape[1]*p)]
    unlabeled_X[:,mask_feature]=labeled_X.mean(axis=0)[mask_feature]
    return  unlabeled_X
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
f=open("letter"+"_statistical"+'_feature'+'_labels_89797989_'+str(labels)+'.csv', "w", encoding="utf-8")
r = csv.DictWriter(f,['algorithm','rate','mean','std'])
X,y=fetch_openml(data_id=6,return_X_y=True)
X=np.array(X).astype(np.float32)
y=LabelEncoder().fit_transform(np.array(y))
num_classes=len(np.unique(y))
labels=labels*num_classes

algorithms = {
              'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
              'SSGMM': SSGMM(num_classes=num_classes),
              'TSVM': TSVM(),
              'LabelPropagation': LabelPropagation(),
              'LabelSpreading': LabelSpreading(),
              'Co_Training': Co_Training(k=30,s=10, base_estimator=XGBClassifier(use_label_encoder=False,eval_metric='logloss')),
              'Tri_Training': Tri_Training(base_estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
              'Assemble': Assemble(T=30,base_estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
              }
Transductive = ['TSVM', 'LabelSpreading','LabelPropagation']

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
            source_X, source_y, target_X, target_y = DataSplit(stratified=True, shuffle=True, random_state=_, X=X, y=y, size_split=0.5)
            labeled_X, labeled_y, test_X, test_y = DataSplit(stratified=True, shuffle=True, random_state=_, X=source_X, y=source_y, size_split=labels)
            # source_X,source_y = source_X[source_y<num_classes/2],source_y[source_y<num_classes/2]
            unlabeled_X = feature_selection(labeled_X,target_X,rate,random_state=_)
            unlabeled_y=target_y
            # print(target_X.shape)
            # print(target_y.shape)
            # labeled_X, labeled_y, test_X, test_y = DataSplit(stratified=True, shuffle=True, random_state=_, X=source_X, y=source_y, size_split=labels)
            # target_X_in, target_y_in,target_X_out, target_y_out=DataSplit(stratified=True, shuffle=True, random_state=_, X=target_X, y=target_y, size_split=0.5)
            # # target_X_in, target_y_in = target_X[target_y<num_classes/2], target_y[target_y<num_classes/2]
            # # target_X_out, target_y_out = target_X[target_y>=num_classes/2], target_y[target_y>=num_classes/2]
            # target_X_out=feature_selection(labeled_X=labeled_X,unlabeled_X=target_X_out,random_state=_)
            # if rate==0:
            #     unlabels=target_X_in.shape[0]
            # else:
            #     unlabels=min(target_X_in.shape[0]/(1-rate),target_X_out.shape[0]/rate) if rate !=1 else target_X_out.shape[0]
            # target_X_in, target_y_in, target_X_in_r, target_y_in_r = DataSplit(stratified=True, shuffle=True, random_state=_, X=target_X_in, y=target_y_in, size_split=int(unlabels*(1-rate)))
            # target_X_out, target_y_out, target_X_out_r, target_y_out_r  = DataSplit(stratified=True, shuffle=True, random_state=_, X=target_X_out, y=target_y_out, size_split=int(unlabels * rate))
            # unlabeled_X=np.concatenate((target_X_in, target_X_out),axis=0)
            # unlabeled_y=np.concatenate((target_y_in, target_y_out),axis=0)
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

            if name is 'XGBClassifier':
                # l=labeled_X.shape[0]
                # u=unlabeled_X.shape[0]
                # sample_weight = np.zeros(l + u)
                # for i in range(l):
                #         sample_weight[i] = 0.9*(l+u)/l
                # for i in range(u):
                #         sample_weight[i + l] = 0.1*(l+u)/u
                # ulb_y=KNeighborsClassifier().fit(labeled_X,labeled_y).predict(unlabeled_X)
                # algorithm = algorithm.fit(np.concatenate((labeled_X,unlabeled_X)), np.concatenate((labeled_y,ulb_y)),sample_weight=sample_weight)
                algorithm = algorithm.fit(labeled_X, labeled_y)
                pred_y = algorithm.predict(test_X)
            elif name in Transductive:
                algorithm_1 = copy.deepcopy(algorithm)
                #algorithm_2=copy.deepcopy(algorithm)
                algorithm_1 = algorithm_1.fit(labeled_X, labeled_y, unlabeled_X)
                pred_y = algorithm_1.predict(test_X, Transductive=False)
                #algorithm_2 = algorithm_2.fit(labeled_X, labeled_y, np.random.rand(unlabeled_X.shape[0], unlabeled_X.shape[1]))
                #pred_y_1 = algorithm_2.predict(test_X, Transductive=False)
            else:
                algorithm_1 = copy.deepcopy(algorithm)
                pred_y = algorithm_1.fit(labeled_X, labeled_y, unlabeled_X).predict(test_X)
                #algorithm_2=copy.deepcopy(algorithm)
                #algorithm_2 = algorithm_2.fit(labeled_X, labeled_y, np.random.rand(unlabeled_X.shape[0], unlabeled_X.shape[1]))
                #pred_y_1 = algorithm_2.predict(test_X)
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
f.close()

# print(dataset)
# print('end!')

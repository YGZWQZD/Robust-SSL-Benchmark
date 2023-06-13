from datasets import load_dataset

# 加载Amazon商品评论数据集
dataset = load_dataset('imdb')

# 获取训练集和测试集
source_x, source_y = dataset['train']['text'],dataset['train']['label']
test_x,test_y= dataset['test']['text'],dataset['test']['label']

sum_len=0
num=0
for text in source_x:
    sum_len+=len(text)
    num+=1
print(sum_len/num)

dataset = load_dataset('amazon_polarity')
target_x, target_y = dataset['content']['text'],dataset['train']['label']
for text in target_x:
    sum_len+=len(text)
    num+=1
print(sum_len/num)

import re
import os
from LAMDA_SSL.Augmentation.Tabular.Noise import Noise
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Algorithm.Classification.Supervised import Supervised
from LAMDA_SSL.Algorithm.Classification.PiModel import PiModel
from LAMDA_SSL.Algorithm.Classification.UDA import UDA
from LAMDA_SSL.Algorithm.Classification.PseudoLabel import PseudoLabel
from LAMDA_SSL.Algorithm.Classification.ICT import ICT
from LAMDA_SSL.Algorithm.Classification.MixMatch import MixMatch
from LAMDA_SSL.Algorithm.Classification.SoftMatch import SoftMatch
from LAMDA_SSL.Algorithm.Classification.FlexMatch import FlexMatch
from LAMDA_SSL.Transform.ToTensor import ToTensor
from LAMDA_SSL.Transform.Vision.Normalization import Normalization
from sklearn.pipeline import Pipeline
import torchvision.transforms as transforms
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Augmentation.Vision.RandomHorizontalFlip import RandomHorizontalFlip
from LAMDA_SSL.Augmentation.Vision.RandAugment import RandAugment
from LAMDA_SSL.Augmentation.Vision.Cutout import Cutout
from LAMDA_SSL.Sampler.RandomSampler import RandomSampler
from LAMDA_SSL.Sampler.SequentialSampler import SequentialSampler
from LAMDA_SSL.Dataloader.UnlabeledDataloader import UnlabeledDataLoader
from LAMDA_SSL.Dataloader.LabeledDataloader import LabeledDataLoader
from LAMDA_SSL.Opitimizer.SGD import SGD
from LAMDA_SSL.Scheduler.CosineWarmup import CosineWarmup
from transformers import AutoModelForSequenceClassification
from LAMDA_SSL.Algorithm.Classification.FixMatch import FixMatch
from LAMDA_SSL.Transform.Text.SynonymsReplacement import SynonymsReplacement
from LAMDA_SSL.Transform.Text.Split import Split
from LAMDA_SSL.Transform.Text.AutoTokenizer import AutoTokenizer
from math import ceil
import torch
import random
import os
import argparse
from LAMDA_SSL.Transform.ToImage import ToImage
import csv
from LAMDA_SSL.Split.DataSplit import DataSplit
import copy
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--iteration', type=int, default=5000)
parser.add_argument('--labels', type=int, default=100)
args = parser.parse_args()
batch_size = args.batch_size
iteration= args.iteration
labels=args.labels
domain=['books','dvd']#,'electronics']#,'kitchen_&_housewares']
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


device='cuda:0'
evaluation= Accuracy()
rate_list=[1.0]
f=open('amazon_yelp'+'_distribution'+'_labels_'+str(labels)+'_17.csv', "w", encoding="utf-8")
r = csv.DictWriter(f,['algorithm','rate','mean','std'])

def get_mena_std(imgs):
    tmp_imgs = []
    for _ in range(len(imgs)):
        tmp_imgs.append(np.array(imgs[_]).transpose(2,0,1).reshape(3,-1))
    tmp_imgs=np.hstack(tmp_imgs)
    mean = np.mean(tmp_imgs / 255, axis=1)
    std = np.std(tmp_imgs / 255, axis=1)
    return mean,std


def worker_init(worked_id):
    worker_seed = seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# with open(os.path.join('./amazon',source,'positive.review'), 'r') as file:
#     source_positive_data = file.read()
# with open(os.path.join('./amazon',source,'negative.review'), 'r') as file:
#     source_negative_data = file.read()
# with open(os.path.join('./amazon',target,'positive.review'), 'r') as file:
#     target_positive_data = file.read()
# with open(os.path.join('./amazon',target,'negative.review'), 'r') as file:
#     target_negative_data = file.read()

# source_positive_data = re.findall(r'<review_text>(.*?)</review_text>', source_positive_data, re.DOTALL)
# source_negative_data = re.findall(r'<review_text>(.*?)</review_text>', source_negative_data, re.DOTALL)
# target_positive_data = re.findall(r'<review_text>(.*?)</review_text>', target_positive_data, re.DOTALL)
# target_negative_data = re.findall(r'<review_text>(.*?)</review_text>', target_negative_data, re.DOTALL)        
# source_x=[]
# source_y=[]
# target_x=[]
# target_y=[]
# for review in source_positive_data:
#     source_x.append(review.strip())
#     source_y.append(1)
# for review in source_negative_data:
#     source_x.append(review.strip())
#     source_y.append(0)
# for review in target_positive_data:
#     target_x.append(review.strip())
#     target_y.append(1)
# for review in target_negative_data:
#     target_x.append(review.strip())
#     target_y.append(0)
# print(len(source_x))
# print(len(target_x))
for rate in rate_list:
    train_pre_transform = None
    valid_pre_transform = None
    test_pre_transform = None
    transform = AutoTokenizer('roberta-base')

    weak_augmentation = Pipeline([('Split', Split()),
            ('SynonymsReplacement', SynonymsReplacement(n=1))])

    strong_augmentation = Pipeline([('Split', Split()),
            ('SynonymsReplacement', SynonymsReplacement(n=5))])
    augmentation = {
            'weak_augmentation': weak_augmentation,
            'strong_augmentation': strong_augmentation
    }
    labeled_sampler = RandomSampler(replacement=True, num_samples=batch_size * iteration)
    unlabeled_sampler = RandomSampler(replacement=True)
    valid_sampler = SequentialSampler()
    test_sampler = SequentialSampler()

    labeled_dataloader = LabeledDataLoader(batch_size=batch_size, num_workers=0, drop_last=True,
                                           worker_init_fn=worker_init)
    unlabeled_dataloader = UnlabeledDataLoader(num_workers=0, drop_last=True, worker_init_fn=worker_init)
    valid_dataloader = UnlabeledDataLoader(batch_size=batch_size, num_workers=0, drop_last=False,
                                           worker_init_fn=worker_init)
    test_dataloader = UnlabeledDataLoader(batch_size=batch_size, num_workers=0, drop_last=False,
                                          worker_init_fn=worker_init)

    network = AutoModelForSequenceClassification.from_pretrained('roberta-base')

    optimizer = SGD(lr=5e-4, momentum=0.9)

    scheduler = CosineWarmup(num_cycles=7. / 16, num_training_steps=iteration)
    algorithms = {
        'Supervised': Supervised(lambda_u=1.0,
                    mu=1, weight_decay=5e-4,
                    eval_it=None,
                    epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                    device=device,
                    labeled_sampler=copy.deepcopy(labeled_sampler),
                    unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                    valid_sampler=copy.deepcopy(valid_sampler),
                    test_sampler=copy.deepcopy(test_sampler),
                    labeled_dataloader=copy.deepcopy(labeled_dataloader),
                    unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                    valid_dataloader=copy.deepcopy(valid_dataloader),
                    test_dataloader=copy.deepcopy(test_dataloader),
                    augmentation=copy.deepcopy(augmentation),
                    network=copy.deepcopy(network),
                    optimizer=copy.deepcopy(optimizer),
                    scheduler=copy.deepcopy(scheduler),
                    verbose=False
                    ),
        'PiModel': PiModel(lambda_u=1.0,
                    mu=1, weight_decay=5e-4,
                    eval_it=None,
                    epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                    device=device,
                    labeled_sampler=copy.deepcopy(labeled_sampler),
                    unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                    valid_sampler=copy.deepcopy(valid_sampler),
                    test_sampler=copy.deepcopy(test_sampler),
                    labeled_dataloader=copy.deepcopy(labeled_dataloader),
                    unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                    valid_dataloader=copy.deepcopy(valid_dataloader),
                    test_dataloader=copy.deepcopy(test_dataloader),
                    augmentation=copy.deepcopy(augmentation),
                    network=copy.deepcopy(network),
                    optimizer=copy.deepcopy(optimizer),
                    scheduler=copy.deepcopy(scheduler),
                    verbose=False
                    ),
        'UDA': UDA(lambda_u=1.0,
                    mu=1, weight_decay=5e-4,
                    eval_it=None,
                    epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                    device=device,
                    labeled_sampler=copy.deepcopy(labeled_sampler),
                    unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                    valid_sampler=copy.deepcopy(valid_sampler),
                    test_sampler=copy.deepcopy(test_sampler),
                    labeled_dataloader=copy.deepcopy(labeled_dataloader),
                    unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                    valid_dataloader=copy.deepcopy(valid_dataloader),
                    test_dataloader=copy.deepcopy(test_dataloader),
                    augmentation=copy.deepcopy(augmentation),
                    network=copy.deepcopy(network),
                    optimizer=copy.deepcopy(optimizer),
                    scheduler=copy.deepcopy(scheduler),
                    verbose=False
                    ),
        'PseudoLabel': PseudoLabel(lambda_u=1.0,
                    mu=1, weight_decay=5e-4,
                    eval_it=None,
                    epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                    device=device,
                    labeled_sampler=copy.deepcopy(labeled_sampler),
                    unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                    valid_sampler=copy.deepcopy(valid_sampler),
                    test_sampler=copy.deepcopy(test_sampler),
                    labeled_dataloader=copy.deepcopy(labeled_dataloader),
                    unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                    valid_dataloader=copy.deepcopy(valid_dataloader),
                    test_dataloader=copy.deepcopy(test_dataloader),
                    augmentation=copy.deepcopy(augmentation),
                    network=copy.deepcopy(network),
                    optimizer=copy.deepcopy(optimizer),
                    scheduler=copy.deepcopy(scheduler),
                    verbose=False
                    ),
        'FixMatch': FixMatch(
                            lambda_u=1.0,
                            mu=1, weight_decay=5e-4,
                            eval_it=None,
                            epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                            device=device,
                            labeled_sampler=copy.deepcopy(labeled_sampler),
                            unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                            valid_sampler=copy.deepcopy(valid_sampler),
                            test_sampler=copy.deepcopy(test_sampler),
                            labeled_dataloader=copy.deepcopy(labeled_dataloader),
                            unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                            valid_dataloader=copy.deepcopy(valid_dataloader),
                            test_dataloader=copy.deepcopy(test_dataloader),
                            augmentation=copy.deepcopy(augmentation),
                            network=copy.deepcopy(network),
                            optimizer=copy.deepcopy(optimizer),
                            scheduler=copy.deepcopy(scheduler),
                            verbose=False
                        ),
        'FlexMatch': FlexMatch(
                            lambda_u=1.0,
                            mu=1, weight_decay=5e-4,
                            eval_it=None,
                            epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                            device=device,
                            labeled_sampler=copy.deepcopy(labeled_sampler),
                            unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                            valid_sampler=copy.deepcopy(valid_sampler),
                            test_sampler=copy.deepcopy(test_sampler),
                            labeled_dataloader=copy.deepcopy(labeled_dataloader),
                            unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                            valid_dataloader=copy.deepcopy(valid_dataloader),
                            test_dataloader=copy.deepcopy(test_dataloader),
                            augmentation=copy.deepcopy(augmentation),
                            network=copy.deepcopy(network),
                            optimizer=copy.deepcopy(optimizer),
                            scheduler=copy.deepcopy(scheduler),verbose=False
                        ),
                'SoftMatch': SoftMatch(
                            lambda_u=1.0,
                            mu=1, weight_decay=5e-4,
                            eval_it=None,
                            epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                            device=device,
                            labeled_sampler=copy.deepcopy(labeled_sampler),
                            unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                            valid_sampler=copy.deepcopy(valid_sampler),
                            test_sampler=copy.deepcopy(test_sampler),
                            labeled_dataloader=copy.deepcopy(labeled_dataloader),
                            unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                            valid_dataloader=copy.deepcopy(valid_dataloader),
                            test_dataloader=copy.deepcopy(test_dataloader),
                            augmentation=copy.deepcopy(augmentation),
                            network=copy.deepcopy(network),
                            optimizer=copy.deepcopy(optimizer),
                            scheduler=copy.deepcopy(scheduler),verbose=False
                        ),
    }
    for name,algorithm in algorithms.items():
        performance_list = []
        for seed in range(3):
            set_seed(seed)
            S_X=source_x
            S_y=source_y
            T_X=target_x
            T_y=target_y
            S_y = np.array(S_y)
            T_y = np.array(T_y)
            labeled_X, labeled_y, _S_X, _S_y = DataSplit(stratified=True, shuffle=True, X=S_X, y=S_y,
                                                             size_split=labels,
                                                             random_state=seed)
            test_X, test_y, __S_X, __S_y = DataSplit(stratified=True, shuffle=True, X=_S_X, y=_S_y,
                                                             size_split=0.5,
                                                             random_state=seed)
            unlabels=min(len(T_X),len(__S_X))
            unlabelsood=ceil(unlabels*rate)
            unlabelsiid=unlabels-unlabelsood
            if unlabelsiid == 0:
                unlabeled_X, unlabeled_y, _, _ = DataSplit(stratified=True, shuffle=True, X=T_X, y=T_y,
                                                           size_split=unlabelsood,
                                                           random_state=seed)

            elif unlabelsood == 0:
                unlabeled_X, unlabeled_y, _, _ = DataSplit(stratified=True, shuffle=True, X=__S_X, y=__S_y,
                                                           size_split=unlabelsiid,
                                                           random_state=seed)
            else:
                unlabeled_X_iid, unlabeled_y_iid, _, _ = DataSplit(stratified=True, shuffle=True, X=__S_X, y=__S_y,
                                                                   size_split=unlabelsiid,
                                                                   random_state=seed)
                unlabeled_X_ood, unlabeled_y_ood, _, _ = DataSplit(stratified=True, shuffle=True, X=T_X, y=T_y,
                                                                   size_split=unlabelsood,
                                                                   random_state=seed)
                unlabeled_X = unlabeled_X_iid + unlabeled_X_ood
                unlabeled_y = np.concatenate((unlabeled_y_iid, unlabeled_y_ood), axis=0)
            algorithm_1 = copy.deepcopy(algorithm)
            labeled_dataset = LabeledDataset(pre_transform=train_pre_transform, transform=transform)
            algorithm_1.labeled_dataset = copy.deepcopy(labeled_dataset)
            unlabeled_dataset = UnlabeledDataset(pre_transform=train_pre_transform, transform=transform)
            algorithm_1.unlabeled_dataset = copy.deepcopy(unlabeled_dataset)
            valid_dataset = UnlabeledDataset(pre_transform=valid_pre_transform, transform=transform)
            algorithm_1.valid_dataset = copy.deepcopy(valid_dataset)
            test_dataset = UnlabeledDataset(pre_transform=test_pre_transform, transform=transform)
            algorithm_1.test_dataset = copy.deepcopy(test_dataset)

            pred_y = algorithm_1.fit(labeled_X, labeled_y, unlabeled_X).predict(test_X)
            performance = Accuracy().scoring(test_y, pred_y)
            performance_list.append(performance)
        performance_list = np.array(performance_list)
        mean = performance_list.mean()
        std = performance_list.std()
        d = {}
        d['algorithm'] = name
        #d['source'] = source
        #d['target'] = target
        d['mean'] = mean
        d['std'] = std
        d['rate'] = rate
        print(d)
        r.writerow(d)
        f.flush()
f.close()





# test_X,test_y= dataset['test']['text'],dataset['test']['label']

# sum_len=0
# num=0
# for text in target_X:
#     sum_len+=len(text)
#     num+=1
# print(sum_len/num)

# dataset = load_dataset('yelp_polarity')
# target_X, target_y = dataset['train']['text'],dataset['train']['label']
# # test_X,test_y= dataset['test']['text'],dataset['test']['label']

# sum_len=0
# num=0
# for text in target_X:
#     sum_len+=len(text)
#     num+=1
# print(sum_len/num)
# dataset = load_dataset('glue','sst2')
# target_X, target_y = dataset['train']['sentence'],dataset['train']['label']
# # test_X,test_y= dataset['test']['text'],dataset['test']['label']

# sum_len=0
# num=0
# for text in target_X:
#     sum_len+=len(text)
#     num+=1
# print(sum_len/num)

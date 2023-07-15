import re
import os
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Algorithm.Classification.Supervised import Supervised
from LAMDA_SSL.Algorithm.Classification.PiModel import PiModel
from LAMDA_SSL.Algorithm.Classification.UDA import UDA
from LAMDA_SSL.Algorithm.Classification.PseudoLabel import PseudoLabel
from datasets import load_dataset
from LAMDA_SSL.Algorithm.Classification.ICT import ICT
from LAMDA_SSL.Algorithm.Classification.MixMatch import MixMatch
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
from LAMDA_SSL.Algorithm.Classification.SoftMatch import SoftMatch
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
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--iteration', type=int, default=5000)
parser.add_argument('--labels', type=int, default=100)
parser.add_argument('--device', type=str, default='cuda:3')
args = parser.parse_args()
batch_size = args.batch_size
iteration= args.iteration
labels=args.labels
device=args.device
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


evaluation= Accuracy()
rate_list=[0,0.2,0.4,0.6,0.8,1.0]
f=open('ag_news'+'_class'+'_labels_'+str(labels)+'.csv', "w", encoding="utf-8")
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

def load_agnews_dataset(path):
    texts = []
    labels = []
    sum_len=0
    num=0
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            label = int(row[0]) - 1
            text = ' '.join(row[1:]) 
            sum_len+=len(text)
            num+=1
            texts.append(text)
            labels.append(label)
    print(sum_len/num)
    return texts, labels

train_path = 'train.csv'
test_path='test.csv'

#dataset = load_dataset('ag_news')

train_texts, train_labels = load_agnews_dataset(train_path)
#dataset['train']['text'],dataset['train']['label']#load_agnews_dataset(train_path)
test_texts, test_labels = load_agnews_dataset(test_path) 
#dataset['test']['text'],dataset['test']['label']#load_agnews_dataset(test_path)

num_classes=4

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
        # 'Supervised': Supervised(lambda_u=1.0,
        #            mu=1, weight_decay=5e-4,
        #            eval_it=None,
        #            epoch=1, num_it_epoch=iteration, num_it_total=iteration,
        #            device=device,
        #            labeled_sampler=copy.deepcopy(labeled_sampler),
        #            unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
        #            valid_sampler=copy.deepcopy(valid_sampler),
        #            test_sampler=copy.deepcopy(test_sampler),
        #            labeled_dataloader=copy.deepcopy(labeled_dataloader),
        #            unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
        #            valid_dataloader=copy.deepcopy(valid_dataloader),
        #            test_dataloader=copy.deepcopy(test_dataloader),
        #            augmentation=copy.deepcopy(augmentation),
        #            network=copy.deepcopy(network),
        #            optimizer=copy.deepcopy(optimizer),
        #            scheduler=copy.deepcopy(scheduler),
        #            verbose=False
        #            ),
        # 'PiModel': PiModel(lambda_u=1.0,
        #             mu=1, weight_decay=5e-4,
        #             eval_it=None,
        #             epoch=1, num_it_epoch=iteration, num_it_total=iteration,
        #             device=device,
        #             labeled_sampler=copy.deepcopy(labeled_sampler),
        #             unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
        #             valid_sampler=copy.deepcopy(valid_sampler),
        #             test_sampler=copy.deepcopy(test_sampler),
        #             labeled_dataloader=copy.deepcopy(labeled_dataloader),
        #             unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
        #             valid_dataloader=copy.deepcopy(valid_dataloader),
        #             test_dataloader=copy.deepcopy(test_dataloader),
        #             augmentation=copy.deepcopy(augmentation),
        #             network=copy.deepcopy(network),
        #             optimizer=copy.deepcopy(optimizer),
        #             scheduler=copy.deepcopy(scheduler),
        #             verbose=False
        #             ),
        # 'UDA': UDA(lambda_u=1.0,
        #             mu=1, weight_decay=5e-4,
        #             eval_it=None,
        #             epoch=1, num_it_epoch=iteration, num_it_total=iteration,
        #             device=device,
        #             labeled_sampler=copy.deepcopy(labeled_sampler),
        #             unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
        #             valid_sampler=copy.deepcopy(valid_sampler),
        #             test_sampler=copy.deepcopy(test_sampler),
        #             labeled_dataloader=copy.deepcopy(labeled_dataloader),
        #             unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
        #             valid_dataloader=copy.deepcopy(valid_dataloader),
        #             test_dataloader=copy.deepcopy(test_dataloader),
        #             augmentation=copy.deepcopy(augmentation),
        #             network=copy.deepcopy(network),
        #             optimizer=copy.deepcopy(optimizer),
        #             scheduler=copy.deepcopy(scheduler),
        #             verbose=False
        #             ),
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
        # 'FixMatch': FixMatch(
        #             lambda_u=1.0,
        #             mu=1, weight_decay=5e-4,
        #             eval_it=None,
        #             epoch=1, num_it_epoch=iteration, num_it_total=iteration,
        #             device=device,
        #             labeled_sampler=copy.deepcopy(labeled_sampler),
        #             unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
        #             valid_sampler=copy.deepcopy(valid_sampler),
        #             test_sampler=copy.deepcopy(test_sampler),
        #             labeled_dataloader=copy.deepcopy(labeled_dataloader),
        #             unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
        #             valid_dataloader=copy.deepcopy(valid_dataloader),
        #             test_dataloader=copy.deepcopy(test_dataloader),
        #             augmentation=copy.deepcopy(augmentation),
        #             network=copy.deepcopy(network),
        #             optimizer=copy.deepcopy(optimizer),
        #             scheduler=copy.deepcopy(scheduler),
        #             verbose=False
        #         ),
        # 'FlexMatch': FlexMatch(
        #             lambda_u=1.0,
        #             mu=1, weight_decay=5e-4,
        #             eval_it=None,
        #             epoch=1, num_it_epoch=iteration, num_it_total=iteration,
        #             device=device,
        #             labeled_sampler=copy.deepcopy(labeled_sampler),
        #             unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
        #             valid_sampler=copy.deepcopy(valid_sampler),
        #             test_sampler=copy.deepcopy(test_sampler),
        #             labeled_dataloader=copy.deepcopy(labeled_dataloader),
        #             unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
        #             valid_dataloader=copy.deepcopy(valid_dataloader),
        #             test_dataloader=copy.deepcopy(test_dataloader),
        #             augmentation=copy.deepcopy(augmentation),
        #             network=copy.deepcopy(network),
        #             optimizer=copy.deepcopy(optimizer),
        #             scheduler=copy.deepcopy(scheduler),verbose=False
        #         ),
                # 'SoftMatch': SoftMatch(
                #     lambda_u=1.0,
                #     mu=1, weight_decay=5e-4,
                #     eval_it=None,
                #     epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                #     device=device,
                #     labeled_sampler=copy.deepcopy(labeled_sampler),
                #     unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                #     valid_sampler=copy.deepcopy(valid_sampler),
                #     test_sampler=copy.deepcopy(test_sampler),
                #     labeled_dataloader=copy.deepcopy(labeled_dataloader),
                #     unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                #     valid_dataloader=copy.deepcopy(valid_dataloader),
                #     test_dataloader=copy.deepcopy(test_dataloader),
                #     augmentation=copy.deepcopy(augmentation),
                #     network=copy.deepcopy(network),
                #     optimizer=copy.deepcopy(optimizer),
                #     scheduler=copy.deepcopy(scheduler),verbose=False
                # ),
    }
    for name,algorithm in algorithms.items():
        performance_list = []
        for seed in range(3):
            set_seed(seed)
            all_labeled_X=train_texts
            all_labeled_y=np.array(train_labels)
            num_classes=np.unique(all_labeled_y).shape[0]
            test_X=test_texts
            test_y=np.array(test_labels)
            S_X, S_y = [all_labeled_X[_] for _ in range(len(all_labeled_X)) if all_labeled_y[_] < int(num_classes / 2)], all_labeled_y[all_labeled_y <int( num_classes / 2)]
            T_X, T_y = [all_labeled_X[_] for _ in range(len(all_labeled_X)) if all_labeled_y[_] >= int(num_classes / 2)], all_labeled_y[all_labeled_y >= int(num_classes / 2)]
            test_X, test_y = [test_X[_] for _ in range(len(test_X)) if test_y[_] < int(num_classes / 2)], test_y[test_y <int( num_classes / 2)]
            labeled_X, labeled_y, _S_X, _S_y = DataSplit(stratified=True, shuffle=True, random_state=seed,
                                                             X=S_X, y=S_y, size_split=labels)
            unlabels = len(T_X)
            target_X_in, target_y_in, target_X_in_r, target_y_in_r = DataSplit(stratified=True,
                                                                               shuffle=True, random_state=seed,
                                                                               X=S_X, y=S_y,
                                                                               size_split=int(
                                                                               unlabels * (1 - rate)))
            target_X_out, target_y_out, target_X_out_r, target_y_out_r = DataSplit(stratified=True,
                                                                                   shuffle=True,
                                                                                   random_state=seed,
                                                                                   X=T_X,
                                                                                   y=T_y,
                                                                                   size_split=int(
                                                                                   unlabels * rate))
            if rate==0:
                unlabeled_X = target_X_in
                unlabeled_y = target_y_in
            elif rate==1:
                unlabeled_X = target_X_out                
                unlabeled_y = target_y_out
            else:
                unlabeled_X = target_X_in+target_X_out
                unlabeled_y = np.concatenate((target_y_in, target_y_out), axis=0)
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
        d['mean'] = mean
        d['std'] = std
        d['rate'] = rate
        r.writerow(d)
        f.flush()
f.close()



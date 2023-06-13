from LAMDA_SSL.Augmentation.Tabular.Noise import Noise
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Algorithm.Classification.Supervised import Supervised
from LAMDA_SSL.Algorithm.Classification.PiModel import PiModel
from LAMDA_SSL.Algorithm.Classification.UDA import UDA
from LAMDA_SSL.Algorithm.Classification.PseudoLabel import PseudoLabel
from LAMDA_SSL.Algorithm.Classification.ICT import ICT
from LAMDA_SSL.Algorithm.Classification.MixMatch import MixMatch
from LAMDA_SSL.Algorithm.Classification.FlexMatch import FlexMatch
from LAMDA_SSL.Algorithm.Classification.SoftMatch import SoftMatch
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
from Network.ResNet50Fc import ResNet50Fc
from LAMDA_SSL.Opitimizer.SGD import SGD
from LAMDA_SSL.Scheduler.CosineWarmup import CosineWarmup
from LAMDA_SSL.Algorithm.Classification.FixMatch import FixMatch
from ImageCLEF import ImageCLEF
from Office31 import Office31
from math import ceil
from VisDA import VisDA
# from Benchmark.Dataset.VisDA import VisDA
import torch
import random
import os
import argparse
from LAMDA_SSL.Transform.ToImage import ToImage
import csv
from LAMDA_SSL.Split.DataSplit import DataSplit
import copy
from PIL import Image
# from sklearn.preprocessing import StandardScaler
import numpy as np
# from math import ceil

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='/data/jialh/VisDA-2017')
parser.add_argument('--dataset',type=str,default='VisDA')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--iteration', type=int, default=2000)
parser.add_argument('--device', type=str, default='cuda:3')
parser.add_argument('--labels', type=int, default=100)
args = parser.parse_args()
root = args.root
dataset=args.dataset
batch_size = args.batch_size
iteration= args.iteration
device=args.device
labels=args.labels
#unlabels=args.unlabels


domain_list=['validation','train']


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




# 6 letter
# 1596 covertype
# 41168 jannis
# 41169 helena
# aloi 1592
# X,y=fetch_openml(data_id=6,return_X_y=True)
# X=np.array(X).astype(np.float32)
# y=LabelEncoder().fit_transform(np.array(y))
# num_classes=len(np.unique(y))
# labels=labels*num_classes
f=open("VisDA"+'_distribution'+'_labels_'+str(labels)+'_6.csv', "w", encoding="utf-8")
r = csv.DictWriter(f,['algorithm','source','target','rate','mean','std'])


def get_mena_std(imgs):
    tmp_imgs = []
    for _ in range(len(imgs)):
        tmp_imgs.append(np.array(Image.open(imgs[_]).convert('RGB')).transpose(2,0,1).reshape(3,-1))
    tmp_imgs=np.hstack(tmp_imgs)
    mean = np.mean(tmp_imgs / 255, axis=1)
    std = np.std(tmp_imgs / 255, axis=1)
    tmp_imgs=[]
    return mean,std


def worker_init(worked_id):
    worker_seed = seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

for source in ['train','validation']:
    for target in ['train','validation']:
        if dataset == 'Office-31':
            source_dataset = Office31(root=root, domain=source)
            target_dataset = Office31(root=root, domain=target)
            num_classes = 31
        elif dataset == 'image-CLEF':
            source_dataset = ImageCLEF(root=root, domain=source)
            target_dataset = ImageCLEF(root=root, domain=target)
            num_classes = 12
        else:
            source_dataset = VisDA(root=root, domain=source)
            target_dataset = VisDA(root=root, domain=target)
            num_classes = 12
        if source==target:
            continue
        for rate in rate_list:
                train_pre_transform =[ [ToImage(load_from_path=True,format='RGB'),transforms.Compose([transforms.Resize([256, 256]), transforms.RandomCrop(224)])]]
                valid_pre_transform = [[ToImage(load_from_path=True,format='RGB'),transforms.Compose([transforms.Resize([256, 256]), transforms.CenterCrop(224)])]]
                test_pre_transform = [[ToImage(load_from_path=True,format='RGB'),transforms.Compose([transforms.Resize([256, 256]), transforms.CenterCrop(224)])]]

                weak_augmentation = RandomHorizontalFlip()
                strong_augmentation = Pipeline([('RandomHorizontalFlip', RandomHorizontalFlip()),
                                                ('RandAugment', RandAugment(n=2, m=10, num_bins=10)),
                                                ('Cutout', Cutout(v=0.5, fill=(127, 127, 127)))
                                                ])
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

                network = ResNet50Fc(num_classes=num_classes, output_feature=False)
                meta_network = ResNet50Fc(num_classes=num_classes, output_feature=False)

                optimizer = SGD(lr=5e-4, momentum=0.9)
                optimizer_wnet = SGD(lr=5e-4, momentum=0.9)

                scheduler = CosineWarmup(num_cycles=7. / 16, num_training_steps=iteration)
                scheduler_wnet = CosineWarmup(num_cycles=7. / 16, num_training_steps=iteration)

                algorithms = {
                    #     'Supervised': Supervised(lambda_u=1.0,
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
                    # 'PseudoLabel': PseudoLabel(lambda_u=1.0,
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
                    # 'ICT': ICT(lambda_u=1.0,
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
                    # 'FixMatch': FixMatch(
                    #                     lambda_u=1.0,
                    #                     mu=1, weight_decay=5e-4,
                    #                     eval_it=None,
                    #                     epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                    #                     device=device,
                    #                     labeled_sampler=copy.deepcopy(labeled_sampler),
                    #                     unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                    #                     valid_sampler=copy.deepcopy(valid_sampler),
                    #                     test_sampler=copy.deepcopy(test_sampler),
                    #                     labeled_dataloader=copy.deepcopy(labeled_dataloader),
                    #                     unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                    #                     valid_dataloader=copy.deepcopy(valid_dataloader),
                    #                     test_dataloader=copy.deepcopy(test_dataloader),
                    #                     augmentation=copy.deepcopy(augmentation),
                    #                     network=copy.deepcopy(network),
                    #                     optimizer=copy.deepcopy(optimizer),
                    #                     scheduler=copy.deepcopy(scheduler),
                    #                     verbose=False
                    #                 ),
                    # 'FlexMatch': FlexMatch(
                    #                     lambda_u=1.0,
                    #                     mu=1, weight_decay=5e-4,
                    #                     eval_it=None,
                    #                     epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                    #                     device=device,
                    #                     labeled_sampler=copy.deepcopy(labeled_sampler),
                    #                     unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                    #                     valid_sampler=copy.deepcopy(valid_sampler),
                    #                     test_sampler=copy.deepcopy(test_sampler),
                    #                     labeled_dataloader=copy.deepcopy(labeled_dataloader),
                    #                     unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                    #                     valid_dataloader=copy.deepcopy(valid_dataloader),
                    #                     test_dataloader=copy.deepcopy(test_dataloader),
                    #                     augmentation=copy.deepcopy(augmentation),
                    #                     network=copy.deepcopy(network),
                    #                     optimizer=copy.deepcopy(optimizer),
                    #                     scheduler=copy.deepcopy(scheduler),verbose=False
                    #                 ),
                                        'SoftMatch': SoftMatch(
                                        lambda_u=1.0,
                                        mu=1, weight_decay=5e-4,
                                        eval_it=None,
                                        use_DA=True,
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
                    performance_list_r = []
                    for seed in range(3):
                        S_X, S_y = source_dataset.data_paths, source_dataset.data_labels
                        T_X, T_y = target_dataset.data_paths, target_dataset.data_labels
                        set_seed(seed)
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
                        mean, std = get_mena_std(labeled_X)
                        transform = Pipeline(
                            [('ToTensor', ToTensor(dtype='float', image=True)),
                             ('Normalization', Normalization(mean=mean, std=std))])
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
                    d['source'] = source
                    d['target'] = target
                    d['mean'] = mean
                    d['std'] = std
                    d['rate'] = rate
                    print(d)
                    r.writerow(d)
                    f.flush()
f.close()

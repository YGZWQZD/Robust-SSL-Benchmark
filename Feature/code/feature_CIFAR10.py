from LAMDA_SSL.Augmentation.Tabular.Noise import Noise
from LAMDA_SSL.Evaluation.Classifier.Accuracy import Accuracy
from LAMDA_SSL.Algorithm.Classification.Supervised import Supervised
from LAMDA_SSL.Algorithm.Classification.PiModel import PiModel
from LAMDA_SSL.Algorithm.Classification.UDA import UDA
from LAMDA_SSL.Algorithm.Classification.PseudoLabel import PseudoLabel
from LAMDA_SSL.Algorithm.Classification.ICT import ICT
from LAMDA_SSL.Algorithm.Classification.MixMatch import MixMatch
from LAMDA_SSL.Algorithm.Classification.FlexMatch import FlexMatch
from LAMDA_SSL.Transform.ToTensor import ToTensor
from LAMDA_SSL.Transform.Vision.Normalization import Normalization
from sklearn.pipeline import Pipeline
from LAMDA_SSL.Augmentation.Vision.CenterCrop import CenterCrop
from LAMDA_SSL.Augmentation.Vision.RandomCrop import RandomCrop
from LAMDA_SSL.Transform.ToImage import ToImage
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
from LAMDA_SSL.Algorithm.Classification.SoftMatch import SoftMatch

from LAMDA_SSL.Dataset.Vision.CIFAR10 import CIFAR10
from ImageCLEF import ImageCLEF
from Office31 import Office31
from math import ceil
from VisDA import VisDA
# from Benchmark.Dataset.VisDA import VisDA
import torch
import random
import os
import argparse
import csv
from LAMDA_SSL.Split.DataSplit import DataSplit
import copy
# from sklearn.preprocessing import StandardScaler
import numpy as np
# from math import ceil
from LAMDA_SSL.Dataset.Vision.CIFAR10 import CIFAR10
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='/data/jialh/image_CLEF')
parser.add_argument('--dataset',type=str,default='cifar-10')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--iteration', type=int, default=10000)
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--labels', type=int, default=200)
args = parser.parse_args()
root = args.root
dataset=args.dataset
batch_size = args.batch_size
iteration= args.iteration
device=args.device
labels=args.labels
#unlabels=args.unlabels


domain_list=['c','i','p']


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
f=open("CIFAR10"+'_feature'+'_labels_'+str(labels)+'_'+str(iteration)+'_5`.csv', "w", encoding="utf-8")
r = csv.DictWriter(f,['algorithm','source','target','rate','mean','std'])


def get_mena_std(imgs):
    tmp_imgs=[]
    for _ in range(imgs.shape[0]):
        tmp_imgs.append(imgs[_].transpose(2,0,1).reshape(3,-1))
    tmp_imgs=np.hstack(tmp_imgs)
    mean = np.mean(tmp_imgs / 255, axis=1)
    std = np.std(tmp_imgs / 255, axis=1)
    return mean,std


def worker_init(worked_id):
    worker_seed = seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

for source in ['cifar10']:
    for target in ['cifar10']:
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
        dataset=CIFAR10(root='\lamda\jialh\Work\LAMDA-SSL_NJU\lamda-ssl\LAMDA-SSL\Download\cifar-10-python',labeled_size=None,stratified=False,shuffle=False,download=True,default_transforms=True)
        num_classes=10
        '''
        if source==target:
            continue
        '''
        for rate in rate_list:
                train_pre_transform = Pipeline([('ToImage',ToImage()),
                                             #     ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect'))
                                                                                ])
                #dataset.pre_transform#transforms.Compose([transforms.Resize([256, 256]), transforms.RandomCrop(224)])
                valid_pre_transform = Pipeline([('ToImage',ToImage()),
                                            #                          ('CenterCrop',CenterCrop(padding=0.125,padding_mode='reflect'))
                                                                                                                                                      ])

                #dataset.pre_transform #transforms.Compose([transforms.Resize([256, 256]), transforms.CenterCrop(224)])
                test_pre_transform = Pipeline([('ToImage',ToImage()),
                                            #('CenterCrop',CenterCrop(padding=0.125,padding_mode='reflect'))
                                                                                                                                                      ])
                
                #dataset.pre_transform#transforms.Compose([transforms.Resize([256, 256]), transforms.CenterCrop(224)])

                weak_augmentation = Pipeline([('RandomHorizontalFlip',RandomHorizontalFlip()),
                                                 ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
                                                                                ])
                strong_augmentation = Pipeline([('RandomHorizontalFlip', RandomHorizontalFlip()),
                                                ('RandomCrop',RandomCrop(padding=0.125,padding_mode='reflect')),
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

                scheduler = CosineWarmup(num_cycles=7./16, num_training_steps=iteration)
                scheduler_wnet = CosineWarmup(num_cycles=7./16, num_training_steps=iteration)

                algorithms = {
                  # 'FixMatch': FixMatch(
                  #               lambda_u=1.0,
                  #               mu=1, weight_decay=5e-4,
                  #               eval_it=None,
                  #               epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                  #               device=device,
                  #               labeled_sampler=copy.deepcopy(labeled_sampler),
                  #               unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                  #               valid_sampler=copy.deepcopy(valid_sampler),
                  #               test_sampler=copy.deepcopy(test_sampler),
                  #               labeled_dataloader=copy.deepcopy(labeled_dataloader),
                  #               unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                  #               valid_dataloader=copy.deepcopy(valid_dataloader),
                  #               test_dataloader=copy.deepcopy(test_dataloader),
                  #               augmentation=copy.deepcopy(augmentation),
                  #               network=copy.deepcopy(network),
                  #               optimizer=copy.deepcopy(optimizer),
                  #               scheduler=copy.deepcopy(scheduler),
                  #               verbose=False
                  #           ),
                  #       'Supervised': Supervised(lambda_u=1.0,
                  #               mu=1, weight_decay=5e-4,
                  #               eval_it=None,
                  #               epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                  #               device=device,
                  #               labeled_sampler=copy.deepcopy(labeled_sampler),
                  #               unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                  #               valid_sampler=copy.deepcopy(valid_sampler),
                  #               test_sampler=copy.deepcopy(test_sampler),
                  #               labeled_dataloader=copy.deepcopy(labeled_dataloader),
                  #               unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                  #               valid_dataloader=copy.deepcopy(valid_dataloader),
                  #               test_dataloader=copy.deepcopy(test_dataloader),
                  #               augmentation=copy.deepcopy(augmentation),
                  #               network=copy.deepcopy(network),
                  #               optimizer=copy.deepcopy(optimizer),
                  #               scheduler=copy.deepcopy(scheduler),
                  #               verbose=False
                  #               ),
                  #   'PiModel': PiModel(lambda_u=1.0,
                  #               mu=1, weight_decay=5e-4,
                  #               eval_it=None,
                  #               epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                  #               device=device,
                  #               labeled_sampler=copy.deepcopy(labeled_sampler),
                  #               unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                  #               valid_sampler=copy.deepcopy(valid_sampler),
                  #               test_sampler=copy.deepcopy(test_sampler),
                  #               labeled_dataloader=copy.deepcopy(labeled_dataloader),
                  #               unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                  #               valid_dataloader=copy.deepcopy(valid_dataloader),
                  #               test_dataloader=copy.deepcopy(test_dataloader),
                  #               augmentation=copy.deepcopy(augmentation),
                  #               network=copy.deepcopy(network),
                  #               optimizer=copy.deepcopy(optimizer),
                  #               scheduler=copy.deepcopy(scheduler),
                  #               verbose=False
                  #               ),
                  #   'UDA': UDA(lambda_u=1.0,
                  #               mu=1, weight_decay=5e-4,
                  #               eval_it=None,
                  #               epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                  #               device=device,
                  #               labeled_sampler=copy.deepcopy(labeled_sampler),
                  #               unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                  #               valid_sampler=copy.deepcopy(valid_sampler),
                  #               test_sampler=copy.deepcopy(test_sampler),
                  #               labeled_dataloader=copy.deepcopy(labeled_dataloader),
                  #               unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                  #               valid_dataloader=copy.deepcopy(valid_dataloader),
                  #               test_dataloader=copy.deepcopy(test_dataloader),
                  #               augmentation=copy.deepcopy(augmentation),
                  #               network=copy.deepcopy(network),
                  #               optimizer=copy.deepcopy(optimizer),
                  #               scheduler=copy.deepcopy(scheduler),
                  #               verbose=False
                  #               ),
                  #   'PseudoLabel': PseudoLabel(lambda_u=1.0,
                  #               mu=1, weight_decay=5e-4,
                  #               eval_it=None,
                  #               epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                  #               device=device,
                  #               labeled_sampler=copy.deepcopy(labeled_sampler),
                  #               unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                  #               valid_sampler=copy.deepcopy(valid_sampler),
                  #               test_sampler=copy.deepcopy(test_sampler),
                  #               labeled_dataloader=copy.deepcopy(labeled_dataloader),
                  #               unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                  #               valid_dataloader=copy.deepcopy(valid_dataloader),
                  #               test_dataloader=copy.deepcopy(test_dataloader),
                  #               augmentation=copy.deepcopy(augmentation),
                  #               network=copy.deepcopy(network),
                  #               optimizer=copy.deepcopy(optimizer),
                  #               scheduler=copy.deepcopy(scheduler),
                  #               verbose=False
                  #               ),
                  #   'ICT': ICT(lambda_u=1.0,
                  #               mu=1, weight_decay=5e-4,
                  #               eval_it=None,
                  #               epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                  #               device=device,
                  #               labeled_sampler=copy.deepcopy(labeled_sampler),
                  #               unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                  #               valid_sampler=copy.deepcopy(valid_sampler),
                  #               test_sampler=copy.deepcopy(test_sampler),
                  #               labeled_dataloader=copy.deepcopy(labeled_dataloader),
                  #               unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                  #               valid_dataloader=copy.deepcopy(valid_dataloader),
                  #               test_dataloader=copy.deepcopy(test_dataloader),
                  #               augmentation=copy.deepcopy(augmentation),
                  #               network=copy.deepcopy(network),
                  #               optimizer=copy.deepcopy(optimizer),
                  #               scheduler=copy.deepcopy(scheduler),
                  #               verbose=False
                  #               ),
                  #   'FlexMatch': FlexMatch(
                  #                       lambda_u=1.0,
                  #                       mu=1, weight_decay=5e-4,
                  #                       eval_it=None,
                  #                       epoch=1, num_it_epoch=iteration, num_it_total=iteration,
                  #                       device=device,
                  #                       labeled_sampler=copy.deepcopy(labeled_sampler),
                  #                       unlabeled_sampler=copy.deepcopy(unlabeled_sampler),
                  #                       valid_sampler=copy.deepcopy(valid_sampler),
                  #                       test_sampler=copy.deepcopy(test_sampler),
                  #                       labeled_dataloader=copy.deepcopy(labeled_dataloader),
                  #                       unlabeled_dataloader=copy.deepcopy(unlabeled_dataloader),
                  #                       valid_dataloader=copy.deepcopy(valid_dataloader),
                  #                       test_dataloader=copy.deepcopy(test_dataloader),
                  #                       augmentation=copy.deepcopy(augmentation),
                  #                       network=copy.deepcopy(network),
                  #                       optimizer=copy.deepcopy(optimizer),
                  #                       scheduler=copy.deepcopy(scheduler),verbose=False
                  #                   ),
                                        'SoftMatch': SoftMatch(
                                        lambda_u=1.0,
                                        mu=1, weight_decay=5e-4,
                                        use_DA=True,
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
                    performance_list_r = []
                    for seed in range(3):
                        all_labeled_X=dataset.labeled_X
                        all_labeled_y=np.array(dataset.labeled_y)
                        num_classes=np.unique(all_labeled_y).shape[0]
                        #print(np.unique(labeled_y))
                        test_X=dataset.test_X
                        test_y=np.array(dataset.test_y)
                        #valid_X=dataset.valid_X
                        #valid_y=np.array(dataset.valid_y)
                        #S_X, S_y = source_dataset.imgs, source_dataset.data_labels
                        #T_X, T_y = target_dataset.imgs, target_dataset.data_labels
                        set_seed(seed)
                        #S_X, S_y = labeled_X[labeled_y < int(num_classes / 2)], labeled_y[labeled_y <int( num_classes / 2)]
                        #T_X, T_y = labeled_X[labeled_y >= int(num_classes / 2)], labeled_y[labeled_y >= int(num_classes / 2)]
                        #test_X, test_y = test_X[test_y < int(num_classes / 2)], test_y[test_y <int( num_classes / 2)]
                        # print(target_X.shape)
                        S_X, S_y, T_X, T_y = DataSplit(stratified=True, shuffle=True, random_state=seed,
                                                                    X=all_labeled_X, y=all_labeled_y, size_split=0.5)
                        print('S_X')
                        print(S_X.shape)
                        labeled_X, labeled_y, _S_X, _S_y = DataSplit(stratified=True, shuffle=True, random_state=seed,
                                                                         X=S_X, y=S_y, size_split=labels)
                        print('labeled_X')
                        print(labeled_X.shape)
                        '''
                        target_X_in, target_y_in = T_X[T_y < num_classes / 2], T_y[
                            T_y < num_classes / 2]
                        target_X_out, target_y_out = T_X[T_y >= num_classes / 2], T_y[
                            T_y >= num_classes / 2]
                        '''
                        '''
                        if rate == 0:
                            unlabels = T_X.shape[0]
                        else:
                            unlabels = min(T_X.shape[0] / (1 - rate),
                                           target_X_out.shape[0] / rate) if rate != 1 else \
                                target_X_out.shape[0]
                        '''
                        unlabels = T_X.shape[0]
                        #print(S_X.shape)
                        print('T_X')
                        print(T_X.shape)
                        T_X[:,:,:,1]=T_X[:,:,:,0]
                        T_X[:,:,:,2]=T_X[:,:,:,0]
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
                        print('target_X_in')
                        print(target_X_in.shape)
                        print('target_X_out')                                                   
                        print(target_X_out.shape)
                        if rate==0:
                            print('rate is 0')
                            unlabeled_X = target_X_in
                            unlabeled_y = target_y_in
                        elif rate==1:
                            print('rate is 1')
                            unlabeled_X = target_X_out
                            unlabeled_y = target_y_out
                        else:
                            unlabeled_X = np.concatenate((target_X_in , target_X_out), axis=0)
                            unlabeled_y = np.concatenate((target_y_in, target_y_out), axis=0)
                        print('labeled_X')
                        print(labeled_X.shape)
                        print('unlabeled_X')
                        print(unlabeled_X.shape)
                        #unlabeled_X = np.concatenate((target_X_in , target_X_out), axis=0)
                        #unlabeled_y = np.concatenate((target_y_in, target_y_out), axis=0)

                        mean, std = get_mena_std(labeled_X)
                        transform = Pipeline(
                            [('ToTensor', ToTensor(dtype='float', image=True)),
                             ('Normalization', Normalization(mean=mean, std=std))])
                        print(mean)
                        print(std)
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

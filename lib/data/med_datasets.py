from json import load
import os
import math
import numpy as np
import torchio as tio
import torch
from sklearn.preprocessing import OneHotEncoder
from monai import data
from monai.data import load_decathlon_datalist
from monai.apps import DecathlonDataset, CrossValidation
import h5py
from .custom_image_dataset import CustomImageDataset
import pandas as pd

from packaging import version
_persistent_workers = False if version.parse(torch.__version__) < version.parse('1.8.2') else True

idx2label_all = {
    'btcv': ['spleen', 
             'right kidney', 
             'left kidney', 
             'gallbladder', 
             'esophagus', 
             'liver', 
             'stomach',
             'aorta',
             'inferior vena cava',
             'portal vein and splenic vein',
             'pancreas',
             'right adrenal gland',
             'left adrenal gland'],
    'msd_brats': ['TC', 'WT', 'ET']
}

btcv_8cls_idx = [0, 1, 2, 3, 5, 6, 7, 10]

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank:self.total_size:self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[:(self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0,high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch



def get_json_trainset(args, workers, train_transform=None):
    data_dir = args.data_path
    print(f'=> Get trainset from specified json file {args.json_list}')
    datalist_json = os.path.join(data_dir, args.json_list)

    datalist = load_decathlon_datalist(datalist_json,
                                        True,
                                        "training",
                                        base_dir=data_dir)
    train_ds = data.CacheDataset(
        data=datalist,
        transform=train_transform,
        cache_num=len(datalist),
        cache_rate=1.0,
        num_workers=workers,
    )
    return train_ds

def get_json_valset(args, val_transform=None):
    data_dir = args.data_path
    print(f'=> Get valset from specified json file {args.json_list}')
    datalist_json = os.path.join(data_dir, args.json_list)

    val_files = load_decathlon_datalist(datalist_json,
                                        True,
                                        "validation",
                                        base_dir=data_dir)
    val_ds = data.Dataset(data=val_files, transform=val_transform)
    return val_ds

def get_msd_trainset(args, workers, train_transform=None, task='Task01_BrainTumour', nfolds=5, ts_fold=0, seed=12345):
    fold_list = list(range(nfolds))
    fold_list.remove(ts_fold)
    assert len(fold_list) == nfolds - 1
    data_dir = args.data_path
    cvdataset = CrossValidation(
                    dataset_cls=DecathlonDataset,
                    nfolds=nfolds,
                    seed=seed,
                    root_dir=data_dir,
                    task=task,
                    section="training",
                    download=False)
    train_ds = cvdataset.get_dataset(folds=fold_list, 
                                     transform=train_transform,
                                     cache_rate=args.cache_rate,
                                     num_workers=workers)
    return train_ds

def get_msd_valset(args, workers, val_transform=None, task='Task01_BrainTumour', nfolds=5, ts_fold=0, seed=12345):
    data_dir = args.data_path
    cvdataset = CrossValidation(
                    dataset_cls=DecathlonDataset,
                    nfolds=nfolds,
                    seed=seed,
                    root_dir=data_dir,
                    task=task,
                    section="training",
                    download=False)
    val_ds = cvdataset.get_dataset(folds=[ts_fold], 
                                     transform=val_transform,
                                     cache_rate=0.,
                                     num_workers=workers)
    return val_ds

def get_adni_trainset(args, train_transform=None):
    data_dir = os.path.join(args.data_path, "train.h5")
    diagnosis = []
    image_train = []
    label_train = []
    with h5py.File(data_dir, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            rid = group.attrs['RID']
            mri_data = group['MRI/T1/data'][:]
            # print(mri_data[np.newaxis, :, :, :].shape)

            if train_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = train_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            diagnosis.append(group.attrs['DX'])
            image_train.append(transformed_image)
            label_train.append(group.attrs['DX'])

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    label_train = OH_encoder.transform(np.array(label_train).reshape(-1, 1))
    train_set = CustomImageDataset(image_train, label_train)
    return train_set

def get_adni_valset(args, val_transform=None):
    data_dir = os.path.join(args.data_path, "valid.h5")
    diagnosis = []
    image_val = []
    label_val = []
    with h5py.File(data_dir, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            rid = group.attrs['RID']
            mri_data = group['MRI/T1/data'][:]
            # print(mri_data[np.newaxis, :, :, :].shape)

            if val_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = val_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            diagnosis.append(group.attrs['DX'])
            image_val.append(transformed_image)
            label_val.append(group.attrs['DX'])

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    label_val = OH_encoder.transform(np.array(label_val).reshape(-1, 1))
    val_set = CustomImageDataset(image_val, label_val)
    return val_set

## UKB, no label?
def get_ukb_trainset(args, train_transform=None):
    data_dir = os.path.join(args.data_path, "uk_train.h5")
    image_train = []
    label_train = []
    with h5py.File(data_dir, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            # rid = group.attrs['RID']
            mri_data = group['MRI/T1/data'][:]
            # print(mri_data[np.newaxis, :, :, :].shape)

            if train_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = train_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            image_train.append(transformed_image)
            label_train.append('X')

    train_set = CustomImageDataset(image_train, label_train)
    return train_set

def get_ukb_valset(args, val_transform=None):
    data_dir = os.path.join(args.data_path, "uk_valid.h5")
    image_val = []
    label_val = []
    with h5py.File(data_dir, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            # rid = group.attrs['RID']
            mri_data = group['MRI/T1/data'][:]
            # print(mri_data[np.newaxis, :, :, :].shape)

            if val_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = val_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            image_val.append(transformed_image)
            label_val.append('X')

    val_set = CustomImageDataset(image_val, label_val)
    return val_set

def get_hospital_trainset(args, train_transform=None):
    data_dir = "/dss/dsshome1/0C/ge79qex2/ModelsGenesis/dataset/238+19+72_tum_splits/"
    file_name = "238+19+72_tum.h5"
    i = args.fold
    train_data = np.load(f'{data_dir}{i}-train.npy', allow_pickle=True)
    diagnosis = []
    image_train = []
    label_train = []
    with h5py.File(data_dir+file_name, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            rid = group.attrs['RID']
            mri_data = group['MRI/T1/data'][:]
            if train_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = train_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            diagnosis.append(group.attrs['DX'])
            if rid in train_data:
                image_train.append(transformed_image)
                label_train.append(group.attrs['DX'])

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    label_train = OH_encoder.transform(np.array(label_train).reshape(-1, 1))
    train_set = CustomImageDataset(image_train, label_train)
    return train_set

def get_dzne_trainset(args, train_transform=None):
    data_dir = "/dss/dsshome1/0C/ge79qex2/ModelsGenesis/dataset/DZNE/"
    file_name = "DZNE_CN_FTD_AD.h5"
    i = args.fold

    train_df = pd.read_csv(f'{data_dir}{i}-train.csv')
    train_data = list(train_df["IMAGEID"])

    diagnosis = []
    image_train = []
    label_train = []
    with h5py.File(data_dir+file_name, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            rid = group.attrs["IMAGEID"]
            mri_data = group['MRI/T1/data'][:]
            if train_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = train_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            diagnosis.append(group.attrs['DX'])
            if rid in train_data:
                image_train.append(transformed_image)
                label_train.append(group.attrs['DX'])

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    label_train = OH_encoder.transform(np.array(label_train).reshape(-1, 1))
    train_set = CustomImageDataset(image_train, label_train)
    return train_set

def get_hospital_valset(args, val_transform=None):
    data_dir = "/dss/dsshome1/0C/ge79qex2/ModelsGenesis/dataset/238+19+72_tum_splits/"
    file_name = "238+19+72_tum.h5"
    i = args.fold
    val_data = np.load(f'{data_dir}{i}-valid.npy', allow_pickle=True)
    diagnosis = []
    image_val = []
    label_val = []
    with h5py.File(data_dir+file_name, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            rid = group.attrs['RID']
            mri_data = group['MRI/T1/data'][:]
            if val_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = val_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            diagnosis.append(group.attrs['DX'])
            if rid in val_data:
                image_val.append(transformed_image)
                label_val.append(group.attrs['DX'])

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    label_val = OH_encoder.transform(np.array(label_val).reshape(-1, 1))
    val_set = CustomImageDataset(image_val, label_val)
    return val_set

def get_dzne_valset(args, val_transform=None):
    data_dir = "/dss/dsshome1/0C/ge79qex2/ModelsGenesis/dataset/DZNE/"
    file_name = "DZNE_CN_FTD_AD.h5"
    i = args.fold

    val_df = pd.read_csv(f'{data_dir}{i}-valid.csv')
    val_data = list(val_df["IMAGEID"])

    diagnosis = []
    image_val = []
    label_val = []
    with h5py.File(data_dir+file_name, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            rid = group.attrs["IMAGEID"]
            mri_data = group['MRI/T1/data'][:]
            if val_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = val_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            diagnosis.append(group.attrs['DX'])
            if rid in val_data:
                image_val.append(transformed_image)
                label_val.append(group.attrs['DX'])

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    label_val = OH_encoder.transform(np.array(label_val).reshape(-1, 1))
    val_set = CustomImageDataset(image_val, label_val)
    return val_set

def get_hospital_testset(args, val_transform=None):
    data_dir = "/dss/dsshome1/0C/ge79qex2/ModelsGenesis/dataset/238+19+72_tum_splits/"
    file_name = "238+19+72_tum.h5"
    i = args.fold
    test_data = np.load(f'{data_dir}{i}-test.npy', allow_pickle=True)
    diagnosis = []
    image_test = []
    label_test = []
    with h5py.File(data_dir+file_name, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            rid = group.attrs['RID']
            mri_data = group['MRI/T1/data'][:]
            if val_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = val_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            diagnosis.append(group.attrs['DX'])
            if rid in test_data:
                image_test.append(transformed_image)
                label_test.append(group.attrs['DX'])

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    label_test = OH_encoder.transform(np.array(label_test).reshape(-1, 1))
    test_set = CustomImageDataset(image_test, label_test)
    return test_set

def get_dzne_testset(args, val_transform=None):
    data_dir = "/dss/dsshome1/0C/ge79qex2/ModelsGenesis/dataset/DZNE/"
    file_name = "DZNE_CN_FTD_AD.h5"
    i = args.fold

    # test_df = pd.read_csv(f'{data_dir}test.csv')
    test_df = pd.read_csv(f'{data_dir}{i}-test.csv')
    test_data = list(test_df["IMAGEID"])

    diagnosis = []
    image_test = []
    label_test = []
    with h5py.File(data_dir+file_name, mode='r') as file:
        for name, group in file.items():
            if name == "stats":
                continue
            rid = group.attrs["IMAGEID"]
            mri_data = group['MRI/T1/data'][:]
            if val_transform:
                # Using torchio
                image_tensor = torch.tensor(mri_data[np.newaxis])
                subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor))
                transformed_subject = val_transform(subject)
                transformed_image = transformed_subject['image'].data
            else:
                transformed_image = mri_data[np.newaxis, :, :, :]

            diagnosis.append(group.attrs['DX'])
            if rid in test_data:
                image_test.append(transformed_image)
                label_test.append(group.attrs['DX'])

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_encoder.fit(np.array(diagnosis).reshape(-1, 1))
    label_test = OH_encoder.transform(np.array(label_test).reshape(-1, 1))
    test_set = CustomImageDataset(image_test, label_test)
    return test_set

# get data loaders
def get_train_loader(args, batch_size, workers, train_transform=None):
    if args.dataset in ['btcv']:
        train_ds = get_json_trainset(args, 
                                     workers=workers, 
                                     train_transform=train_transform)
    elif args.dataset == 'msd_brats':
        train_ds = get_msd_trainset(args, workers, 
                                    train_transform=train_transform, 
                                    task='Task01_BrainTumour', 
                                    ts_fold=args.ts_fold, 
                                    seed=args.data_seed)
    elif args.dataset == 'adni':
        train_ds = get_adni_trainset(args, train_transform=train_transform)
    elif args.dataset == 'ukb':
        train_ds = get_ukb_trainset(args, train_transform=train_transform)
    elif args.dataset == 'hospital':
        train_ds = get_hospital_trainset(args, train_transform=train_transform)
    elif args.dataset == 'dzne':
        train_ds = get_dzne_trainset(args, train_transform=train_transform)
    else:
        raise NotImplementedError(f"{args.dataset} is not supported yet.")

    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(train_ds,
                                    batch_size=batch_size,
                                    shuffle=(train_sampler is None),
                                    num_workers=workers,
                                    sampler=train_sampler,
                                    pin_memory=True,
                                    persistent_workers=_persistent_workers)
    return train_loader

def get_val_loader(args, batch_size, workers, val_transform=None):
    if args.dataset in ['btcv']:
        val_ds = get_json_valset(args, val_transform=val_transform)
    elif args.dataset == 'msd_brats':
        val_ds = get_msd_valset(args, 
                                workers, 
                                val_transform=val_transform,
                                task='Task01_BrainTumour',
                                ts_fold=args.ts_fold, 
                                seed=args.data_seed)
    elif args.dataset == 'adni':
        val_ds = get_adni_valset(args, val_transform=val_transform)
    elif args.dataset == 'ukb':
        val_ds = get_ukb_valset(args, val_transform=val_transform)
    elif args.dataset == 'hospital':
        val_ds = get_hospital_valset(args, val_transform=val_transform)
    elif args.dataset == 'dzne':
        val_ds = get_dzne_valset(args, val_transform=val_transform)
    else:
        raise NotImplementedError(f"{args.dataset} is not supported yet.")

    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(val_ds,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=workers,
                                 sampler=val_sampler,
                                 pin_memory=True,
                                 persistent_workers=_persistent_workers)
    return val_loader

def get_test_loader(args, batch_size, workers, val_transform=None):
    if args.dataset == 'hospital':
        test_ds = get_hospital_testset(args, val_transform=val_transform)
    elif args.dataset == 'dzne':
        test_ds = get_dzne_testset(args, val_transform=val_transform)
    else:
        raise NotImplementedError(f"{args.dataset} is not supported yet.")

    test_sampler = Sampler(test_ds) if args.distributed else None
    test_loader = data.DataLoader(test_ds,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=workers,
                                  sampler=test_sampler,
                                  pin_memory=True,
                                  persistent_workers=_persistent_workers)
    return test_loader
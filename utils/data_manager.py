import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFAR10_224, iCIFAR100_224, iImageNetR, iCUB200_224, iResisc45_224, iCARS196_224, iSketch345_224
from copy import deepcopy
import random
from utils.nncsl_functions import init_transform
import torch

class DataManager(object): 
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), 'No enough classes.'
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)   
        offset = len(self._class_order) - sum(self._increments) 
        if offset > 0:
            self._increments.append(offset) 

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task): 
        return self._increments[task] 
    
    def get_dataset(self, indices, source, mode, tasks , task_idx, buffer_lst, appendent=None, ret_data=False, with_raw=False, with_noise=False, keep_file= None, compute_mean = False, unsupervised = False): 
        if source == 'train':
            x, y = self._train_data, self._train_targets
            training = True
        elif source == 'test':
            x, y = self._test_data, self._test_targets
            training = False
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'supervised':
            trsf = transforms.Compose([*self._supervised_trsf, *self._common_trsf])
        elif mode == 'unsupervised': 
            w_trsf = transforms.Compose([*self._w_unsupervised_trsf, *self._common_trsf])
            s_trsf = transforms.Compose([*self._s_unsupervised_trsf, *self._common_trsf])
        elif mode == 'flip':
            trsf = transforms.Compose([*self._test_trsf, transforms.RandomHorizontalFlip(p=1.), *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        if not unsupervised:
            targets, data = init_transform(y.tolist(), x, keep_file=keep_file, training=training, tasks=tasks, task_idx=task_idx, buffer_lst=buffer_lst) 
        else:
            cls_per_task = int(len(tasks) / (task_idx+1))
            cur_cls = tasks[-cls_per_task:]
            new_targets, new_samples = [], []
            imbalance_num_samples = torch.linspace(500,5,10).int()     #CIFAR100 : 10 cls per task. 10 task
            for idx, indx in enumerate(cur_cls):
                indices = np.where(y == indx)
                new_targets.extend(y[indices])
                new_samples.extend(x[indices])

            targets, data = np.array(new_targets), np.array(new_samples)
            return DummyDataset(data, targets, w_trsf, self.use_path, with_raw, with_noise, unsupervised = True, s_trsf= s_trsf)

        if compute_mean:
            data, targets = self._select(data, targets, low_range=indices[0], high_range=indices[0]+1)

        if ret_data: 
            return data, targets, DummyDataset(data, targets, trsf, self.use_path, with_raw, with_noise)
        else:
            return DummyDataset(data, targets, trsf, self.use_path, with_raw, with_noise)

    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'supervised':
            trsf = transforms.Compose([*self._supervised_trsf, *self._common_trsf])
        elif mode == 'unsupervised': # TODO
            w_trsf = transforms.Compose([*self._w_unsupervised_trsf, *self._common_trsf])
            s_trsf = transforms.Compose([*self._s_unsupervised_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data), val_samples_per_class, replace=False)
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:

            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets))+1):
                append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                           low_range=idx, high_range=idx+1)
                val_indx = np.random.choice(len(append_data), val_samples_per_class, replace=False)
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(train_data, train_targets, trsf, self.use_path), \
            DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed): # DataMananger() calls this function 
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path        #False for CIFAR100

        # Transforms
        self._supervised_trsf = idata.supervised_trsf
        self._w_unsupervised_trsf = idata.w_unsupervised_trsf
        self._s_unsupervised_trsf = idata.s_unsupervised_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order) 

        # Map indices
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order) 
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range): 
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


class DummyDataset(Dataset): 
    def __init__(self, images, labels, trsf, use_path=False, with_raw=False, with_noise=False, unsupervised = False, s_trsf= None):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.with_raw = with_raw
        self.unsupervised = unsupervised
        self.s_trsf = s_trsf
        if use_path and with_raw:
            self.raw_trsf = transforms.Compose([transforms.Resize((500, 500)), transforms.ToTensor()])
        else:
            self.raw_trsf = transforms.Compose([transforms.ToTensor()])
        if with_noise:
            class_list = np.unique(self.labels)
            self.ori_labels = deepcopy(labels)
            for cls in class_list:
                random_target = class_list.tolist()
                random_target.remove(cls)
                tindx = [i for i, x in enumerate(self.ori_labels) if x == cls]
                for i in tindx[:round(len(tindx)*0.2)]:
                    self.labels[i] = random.choice(random_target)

        self.target_indices = []
        for t in range(100):  #100 due to CIFAR100 #TODO Remove hardcoded value
            indices = np.squeeze(np.argwhere(self.labels == t)).tolist()
            if isinstance(indices, int):
                indices = [indices]
            self.target_indices.append(indices)
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if not self.unsupervised:
            if self.use_path:
                load_image = pil_loader(self.images[idx])
                image = self.trsf(load_image)
            else:
                load_image = Image.fromarray(self.images[idx])
                image = self.trsf(load_image)
            label = self.labels[idx]
            if self.with_raw:
                return idx, image, label, self.raw_trsf(load_image) 
            return idx, image, label
        
        else:
            if self.use_path:
                load_image = pil_loader(self.images[idx])
                w_image = self.trsf(load_image)
                s_image = self.s_trsf(load_image)
            else:
                load_image = Image.fromarray(self.images[idx])
                w_image = self.trsf(load_image)
                s_image = self.s_trsf(load_image)
            label = self.labels[idx]
            if self.with_raw:
                return idx, w_image, s_image, label, self.raw_trsf(load_image) 
            return idx, w_image, s_image, label


def _map_new_class_index(y, order): # _setup_data() calls this function
    return np.array(list(map(lambda x: order.index(x), y)))

def _get_idata(dataset_name): # _setup_data() in DataManager() calls this function
    name = dataset_name.lower()
    if name == 'cifar10':
        return iCIFAR10()
    elif name == 'cifar10_224':
        return iCIFAR10_224()
    elif name == 'cifar100':
        return iCIFAR100()
    elif name == 'cifar100_224':
        return iCIFAR100_224()
    elif name == 'imagenet1000':
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "imagenet-r":
        return iImageNetR()
    elif name == 'cub200_224':
        return iCUB200_224()
    elif name == 'resisc45':
        return iResisc45_224()
    elif name == 'cars196_224':
        return iCARS196_224()
    elif name == 'sketch345_224':
        return iSketch345_224()
    else:
        raise NotImplementedError('Unknown dataset {}.'.format(dataset_name))

def pil_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    '''
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    '''
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

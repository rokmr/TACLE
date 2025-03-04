import os
import subprocess
import time
from logging import getLogger
import numpy as np
from math import ceil
import random 
import torch
import torchvision.transforms as transforms
import torchvision
import PIL
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps

_GLOBAL_SEED = 0
logger = getLogger()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

buffer_lst = None

def init_transform(targets, samples, training=True, keep_file=None, tasks=None, task_idx=None, buffer_lst=None): #buffer_lst: None
    """ Transforms applied to dataset at the start of training """
    cls_per_task = int(len(tasks) / (task_idx+1))
    cur_cls = tasks[-cls_per_task:]
    new_targets, new_samples = [], []
    buffer_lst = None
    if training and (keep_file is not None):
        assert os.path.exists(keep_file), 'keep file does not exist'
        with open(keep_file, 'r') as rfile:
            for line in rfile:
                indx = int(line.split('\n')[0])
                if targets[indx] in cur_cls:
                    new_targets.append(targets[indx])
                    new_samples.append(samples[indx])
                elif buffer_lst is not None and indx in buffer_lst:
                    new_targets.append(targets[indx])
                    new_samples.append(samples[indx])
    else:
        if tasks is not None:
            for s, t in zip(samples, targets):
                if t in tasks:
                    new_targets.append(t)
                    new_samples.append(s)
        else:
            new_targets, new_samples = targets, samples
    return np.array(new_targets), np.array(new_samples)


def make_buffer_lst(buffer_lst, buffer_size, subset_path, subset_path_cls, tasks, task_idx):
    import random
    # Get the index in dataset for labeled samples of current task
    def get_lst(subset_path, subset_path_cls, target_cls):
        buffer_lst = []
        cls_idx_lst = []
        cls_lst = []
        with open(subset_path_cls, 'r') as rfile:
            for i, line in enumerate(rfile):
                label = int(line.split('\n')[0])
                if label in target_cls:
                    cls_idx_lst.append(i)
                    cls_lst.append(label)
        index_lst = []
        with open(subset_path, 'r') as rfile:
            for i, line in enumerate(rfile):
                index = int(line.split('\n')[0])
                if i in cls_idx_lst:
                    index_lst.append(index)
        return cls_lst, cls_idx_lst, index_lst

    pre_classes = sum(tasks[:task_idx], [])
    num_pre_classes = len(pre_classes)
    seen_classes = sum(tasks[:task_idx+1], [])
    num_seen_classes = len(seen_classes)
    num_cur_classes = len(tasks[task_idx])
    cls_lst, cls_idx_lst, index_lst = get_lst(subset_path, subset_path_cls, tasks[task_idx])
    sorted_idx = np.argsort(cls_lst)
    cls_lst = np.array(cls_lst)[sorted_idx].tolist()
    index_lst = np.array(index_lst)[sorted_idx].tolist()
    if buffer_lst is None:
        if buffer_size >= len(index_lst):
            buffer_lst = index_lst
        else:
            cur_num_per_class = int(len(index_lst)/num_cur_classes)
            new_num_per_class = int(buffer_size / num_seen_classes)
            assert new_num_per_class <= cur_num_per_class
            buffer_lst = []
            for i in range(num_seen_classes):
                buffer_lst += index_lst[i*cur_num_per_class:(i+1)*cur_num_per_class][:new_num_per_class]
            # Fill the empty space of buffer
            if len(buffer_lst) < buffer_size:
                diff = buffer_size - len(buffer_lst)
                for i in range(min(num_cur_classes, diff)):
                    buffer_lst += index_lst[i*cur_num_per_class:(i+1)*cur_num_per_class][-1:]
    else:
        num_in_buffer = len(buffer_lst)
        if buffer_size - num_in_buffer >= len(index_lst):
            buffer_lst += index_lst
        else:
            cur_num_per_class = int(len(index_lst)/num_cur_classes)
            pre_num_per_class = int(len(buffer_lst)/num_pre_classes)
            new_num_per_class = int(buffer_size / num_seen_classes)
            num_modulo = buffer_size - pre_num_per_class * num_pre_classes
            assert new_num_per_class <= cur_num_per_class
            assert new_num_per_class <= pre_num_per_class
            temp_lst = []
            for i in range(num_pre_classes):
                temp_lst += buffer_lst[i*pre_num_per_class:(i+1)*pre_num_per_class][:new_num_per_class]
            for i in range(num_cur_classes):
                temp_lst += index_lst[i*cur_num_per_class:(i+1)*cur_num_per_class][:new_num_per_class]      
            # Fill the empty space of buffer
            if len(temp_lst) < buffer_size:
                diff = buffer_size - len(temp_lst)
                for i in range(min(num_pre_classes, diff)):
                    temp_lst += buffer_lst[i*pre_num_per_class:(i+1)*pre_num_per_class][-1:]
            if len(temp_lst) < buffer_size:
                diff = buffer_size - len(temp_lst)
                for i in range(min(num_cur_classes, diff)):
                    temp_lst += index_lst[i*cur_num_per_class:(i+1)*cur_num_per_class][-1:]    
            buffer_lst = temp_lst

    return buffer_lst


class ClassStratifiedSampler(torch.utils.data.Sampler):

    def __init__(
        self,
        data_source,
        world_size,
        rank,
        batch_size=1,
        classes_per_batch=10,
        epochs=1,
        seed=0,
        unique_classes=False
    ):
        super(ClassStratifiedSampler, self).__init__(data_source)
        self.data_source = data_source

        self.rank = rank
        self.world_size = world_size
        self.cpb = classes_per_batch
        self.unique_cpb = unique_classes
        self.batch_size = batch_size
        self.num_classes = len(set(data_source.labels))
        self.classes = set(data_source.labels)
        self.epochs = epochs
        self.outer_epoch = 0
        if not self.unique_cpb:
            assert self.num_classes % self.cpb == 0

        self.base_seed = seed  # instance seed
        self.seed = seed  # subsample sampler seed

    def set_epoch(self, epoch):
        self.outer_epoch = epoch

    def set_inner_epochs(self, epochs):
        self.epochs = epochs

    def _next_perm(self):
        self.seed += 1
        g = torch.Generator()
        g.manual_seed(self.seed)
        self._perm = torch.randperm(self.num_classes, generator=g)

    def _get_perm_ssi(self):
        start = self._ssi
        end = self._ssi + self.cpb
        subsample = self._perm[start:end]
        return subsample

    def _next_ssi(self):
        if not self.unique_cpb:
            self._ssi = (self._ssi + self.cpb) % self.num_classes
            if self._ssi == 0:
                self._next_perm()
        else:
            self._ssi += self.cpb * self.world_size
            max_end = self._ssi + self.cpb * (self.world_size - self.rank)
            if max_end > self.num_classes:
                self._ssi = self.rank * self.cpb
                self._next_perm()

    def _get_local_samplers(self, epoch):
        """ Generate samplers for local data set in given epoch """
        seed = int(self.base_seed + epoch
                   + self.epochs * self.rank
                   + self.outer_epoch * self.epochs * self.world_size)
        g = torch.Generator()
        g.manual_seed(seed)
        samplers = []
        for t in self.classes:
            t_indices = np.array(self.data_source.target_indices[t])
            if not self.unique_cpb:
                i_size = len(t_indices) // self.world_size
                if i_size > 0:
                    t_indices = t_indices[self.rank*i_size:(self.rank+1)*i_size]
            if len(t_indices) > 1:
                t_indices = t_indices[torch.randperm(len(t_indices), generator=g)]
            samplers.append(iter(t_indices))
        return samplers

    def _subsample_samplers(self, samplers):
        """ Subsample a small set of samplers from all class-samplers """
        subsample = self._get_perm_ssi()
        subsampled_samplers = []
        for i in subsample:
            subsampled_samplers.append(samplers[i])
        self._next_ssi()
        return zip(*subsampled_samplers)

    def __iter__(self):
        self._ssi = self.rank*self.cpb if self.unique_cpb else 0
        self._next_perm()

        # -- iterations per epoch (extract batch-size samples from each class)
        ipe = (self.num_classes // self.cpb if not self.unique_cpb
               else self.num_classes // (self.cpb * self.world_size)) * self.batch_size

        for epoch in range(self.epochs):

            # -- shuffle class order
            samplers = self._get_local_samplers(epoch)
            subsampled_samplers = self._subsample_samplers(samplers)

            counter, batch = 0, []
            for i in range(ipe):
                batch += list(next(subsampled_samplers))
                counter += 1
                if counter == self.batch_size:
                    yield batch
                    counter, batch = 0, []
                    if i + 1 < ipe:
                        subsampled_samplers = self._subsample_samplers(samplers)

    def __len__(self):
        if self.batch_size == 0:
            return 0

        ipe = (self.num_classes // self.cpb if not self.unique_cpb
               else self.num_classes // (self.cpb * self.world_size))
        return self.epochs * ipe

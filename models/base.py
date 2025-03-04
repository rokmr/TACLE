import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist

EPSILON = 1e-8
batch_size = 64

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._device = args['device'][0]
        self._multiple_gpus = args['device']

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(self._targets_memory), 'Exemplar size error.'
        return len(self._targets_memory)

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim


    def save_checkpoint(self, filename, head_only=False): 
        if hasattr(self._network, 'module'):
            to_save = self._network.module
        else:
            to_save = self._network

        if head_only:
            to_save = to_save.fc
            
        save_dict = {
            'tasks': self._cur_task,
            'model_state_dict': to_save.state_dict(),
        }
        torch.save(save_dict, '{}_{}.pth'.format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true): 
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, increment=self.increment)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        ret['top{}'.format(5)] = np.around((y_pred.T == np.tile(y_true, (self.topk, 1))).sum()*100/len(y_true),
                                                   decimals=2)
        return ret

    def eval_task(self): 
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        return cnn_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)


    def _inner_eval(self, model, loader):
        model.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)['logits']
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        y_pred, y_true = np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]       

        cnn_accy = self._evaluate(y_pred, y_true) 
        return cnn_accy

    def _compute_accuracy(self, model, loader, unsupervised =  False): 
        model.eval()
        correct, total = 0, 0
        t_correct, t_total = 0, 0
        targets_u_vector = torch.tensor([]).to(self._device)
        if not unsupervised:
            for i, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)

                with torch.no_grad():
                    outputs = model(inputs)['logits']
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == targets).sum()
                total += len(targets)
        else:
            for i, (_, inputs, _, targets) in enumerate(loader):
                inputs = inputs.to(self._device)

                with torch.no_grad():
                    outputs = model(inputs)['logits']
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == targets).sum()
                total += len(targets)

                outputs_probs = torch.nn.functional.softmax(outputs, dim=-1)
                max_probs, targets_u = torch.max(outputs_probs, dim=-1)
                mask = max_probs.ge(self.threshold)
                
                preds_after_mask = targets_u[mask]
                targets_u_vector = torch.cat((targets_u_vector.float(), preds_after_mask.float()), dim=0)
                targets_after_mask = targets[mask]
                t_correct += torch.sum(preds_after_mask == targets_after_mask.float().to(self._device))
                t_total += len(targets_after_mask)

        u_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
        if not unsupervised:
            return u_acc
        us_acc_with_thresh = np.around(tensor2numpy(t_correct)*100 / t_total, decimals=2)
        TARGETS_U_COUNTS = self.count_unique_elements(targets_u_vector) 
        
        return u_acc, us_acc_with_thresh, t_total, TARGETS_U_COUNTS

    def _eval_cnn(self, loader): 
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)['logits']
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # (topk values , indices of top k)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true) 

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        norm_means = class_means / np.linalg.norm(class_means)
        dists = cdist(norm_means, vectors, 'sqeuclidean')  
        scores = dists.T  
        return np.argsort(scores, axis=1)[:, :self.topk], y_true  

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()

            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(self._network.module.extract_vector(_inputs.to(self._device))) 
            else:
                _vectors = tensor2numpy(self._network.extract_vector(_inputs.to(self._device)))

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
    
    def _extract_vectors_unlabel_conf(self, loader): 
        self._network.eval()
        vectors, targets = [], []
        for i, (u_i, inputs_w,inputs_s, _) in enumerate(self.unsupervised_loader):
            with torch.no_grad():
                outputs = self._network(inputs_w.to(self._device))
                logits = outputs['logits']
                vectors_unsup= outputs['features']
                pseudo_label = torch.softmax(logits[:, self._known_classes:], dim= 1)
                max_prob, hard_label = torch.max(pseudo_label, dim=1)
                indicator = max_prob>self.threshold
                _vectors = tensor2numpy(vectors_unsup[indicator])
                _targets = tensor2numpy(hard_label[indicator] + self._known_classes)
                
 
                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _extract_vectors_aug(self, loader, repeat=2):
        self._network.eval()
        vectors, targets = [], []
        for _ in range(repeat):
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                with torch.no_grad():
                    if isinstance(self._network, nn.DataParallel):
                        _vectors = tensor2numpy(self._network.module.extract_vector(_inputs.to(self._device)))
                    else:
                        _vectors = tensor2numpy(self._network.extract_vector(_inputs.to(self._device)))

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets) 

    def _reduce_exemplar(self, data_manager, m):
        logging.info('Reducing exemplars...({} per classes)'.format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = np.concatenate((self._data_memory, dd)) if len(self._data_memory) != 0 else dd
            self._targets_memory = np.concatenate((self._targets_memory, dt)) if len(self._targets_memory) != 0 else dt

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source='train', mode='test', appendent=(dd, dt))
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean
            
    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False, unsupervised_dset_loader=None, strategy =None): 
    
        if hasattr(self, '_class_means') and self._class_means is not None and not check_diff: 
            ori_classes = self._class_means.shape[0]
            assert ori_classes==self._known_classes
            new_class_means = np.zeros((self._total_classes, self.feature_dim)) 
            new_class_means[:self._known_classes] = self._class_means
            self._class_means = new_class_means 
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs
            self._class_covs = new_class_cov

        elif not check_diff: 
            self._class_means = np.zeros((self._total_classes, self.feature_dim)) 
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim)) 

        if check_diff: 
            for class_idx in range(0, self._known_classes):
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)    
                class_mean = np.mean(vectors, axis=0)
                class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)
                if check_diff:
                    log_info = "cls {} sim: {}".format(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)).item())
                    logging.info(log_info)
                    np.save('task_{}_cls_{}_mean.npy'.format(self._cur_task, class_idx), class_mean)

        if oracle: 
            for class_idx in range(0, self._known_classes):
                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)

                class_mean = np.mean(vectors, axis=0)
                class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)+torch.eye(class_mean.shape[-1])*1e-5
                self._class_means[class_idx, :] = class_mean
                self._class_covs[class_idx, ...] = class_cov            
        if strategy == 'stage2_mean_var_labelled_unlabelled':
            vectors_unlabel, pseudo_label = self._extract_vectors_unlabel_conf(unsupervised_dset_loader)

        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',  mode='test', tasks =self.tasks, task_idx=self._cur_task, buffer_lst = self.buffer_lst, ret_data=True, keep_file = self.subset_path, compute_mean=True)
            
            
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            vectors, _ = self._extract_vectors(idx_loader) 

            if strategy == 'stage2_mean_var_labelled_unlabelled':
                vectors = np.concatenate([vectors, vectors_unlabel[pseudo_label==class_idx]])

            class_mean = np.mean(vectors, axis=0) 
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)+torch.eye(class_mean.shape[-1])*1e-4 

            if check_diff:
                log_info = "cls {} sim: {}".format(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)).item())
                logging.info(log_info)
                np.save('task_{}_cls_{}_mean.npy'.format(self._cur_task, class_idx), class_mean)
                np.save('task_{}_cls_{}_mean_beforetrain.npy'.format(self._cur_task, class_idx), self._class_means[class_idx, :])
            self._class_means[class_idx, :] = class_mean  
            self._class_covs[class_idx, ...] = class_cov  
            

    def _construct_exemplar(self, data_manager, m, mode='icarl'):
        logging.info('Constructing exemplars...({} per classes)'.format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            if mode == 'icarl':
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)
                m = min(m, vectors.shape[0])
                # Select
                selected_exemplars = []
                exemplar_vectors = []  
                for k in range(1, m+1):
                    S = np.sum(exemplar_vectors, axis=0)  
                    mu_p = (vectors + S) / k  
                    i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                    selected_exemplars.append(np.array(data[i]))  
                    exemplar_vectors.append(np.array(vectors[i])) 

                    vectors = np.delete(vectors, i, axis=0) 
                    data = np.delete(data, i, axis=0)  
                selected_exemplars = np.array(selected_exemplars)
                exemplar_targets = np.full(m, class_idx)
            else:
                selected_index = np.random.choice(len(data), (min(m, len(data)),), replace=False)
                selected_exemplars = data[selected_index]
                exemplar_targets = np.full(min(m, len(data)), class_idx)
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                len(self._targets_memory) != 0 else exemplar_targets

            # Exemplar mean
            idx_dataset = data_manager.get_dataset([], source='train', mode='test',
                                                   appendent=(selected_exemplars, exemplar_targets))
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info('Constructing exemplars for new classes...({} per classes)'.format(m))
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = self._data_memory[mask], self._targets_memory[mask]

            class_dset = data_manager.get_dataset([], source='train', mode='test',
                                                  appendent=(class_data, class_targets))
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                 mode='test', ret_data=True)
            class_loader = DataLoader(class_dset, batch_size=batch_size, shuffle=False, num_workers=4)

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m+1):
                S = np.sum(exemplar_vectors, axis=0)  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(np.array(data[i]))  # New object to avoid passing by inference
                exemplar_vectors.append(np.array(vectors[i]))  # New object to avoid passing by inference

                vectors = np.delete(vectors, i, axis=0)  # Remove it to avoid duplicative selection
                data = np.delete(data, i, axis=0)  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = np.concatenate((self._data_memory, selected_exemplars)) if len(self._data_memory) != 0 \
                else selected_exemplars
            self._targets_memory = np.concatenate((self._targets_memory, exemplar_targets)) if \
                len(self._targets_memory) != 0 else exemplar_targets

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset([], source='train', mode='test',
                                                     appendent=(selected_exemplars, exemplar_targets))
            exemplar_loader = DataLoader(exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means

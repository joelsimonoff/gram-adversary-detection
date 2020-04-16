import utils.calculate_log as callog
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import math

def G_p(temp):
    temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
    temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1)))).sum(dim=2)
    return temp.reshape(temp.shape[0],-1)

def detect(all_test_deviations,all_ood_deviations, test_confs = None, ood_confs=None, verbose=True, normalize=False):
    if test_confs is not None:
        test_confs = np.array(test_confs)
        ood_confs = np.array(ood_confs)

    average_results = {}
    for i in range(1,11):
        test_deviations = all_test_deviations.sum(axis=1)
        ood_deviations = all_ood_deviations.sum(axis=1)

        results = callog.compute_metric(-test_deviations,-ood_deviations)
        for m in results:
            average_results[m] = average_results.get(m,0)+results[m]

    for m in average_results:
        average_results[m] /= i

    return average_results

def cpu(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cpu()
    return ob

def cuda(ob):
    for i in range(len(ob)):
        for j in range(len(ob[i])):
            ob[i][j] = ob[i][j].cuda()
    return ob

class Detector:
    def __init__(self, model, data_train, data_test=None, batch_size=128, pbar = None, batch_detector = False, train_golds=None):
        if pbar is None:
            self.pbar = lambda x, total = None: x
            self.verbose = False
        else:
            self.pbar = pbar
            self.verbose = True
        self.all_test_deviations = None
        self.target_deviation = None
        self.mins = {}
        self.maxs = {}
        self.batch_size = batch_size
        self.classes = range(10)
        self.torch_model = model
        self.batch_detector = batch_detector
        
        if self.batch_detector:
            self.train_preds = train_golds
            
            self.compute_batch_minmaxs(data_train)
        else:
            self.train_preds = []
            self.train_confs = []
            self.train_logits = []
            for idx in range(0,len(data_train),batch_size):
                batch = torch.squeeze(torch.stack([x[0] for x in data_train[idx:idx+batch_size]]),dim=1).cuda()

                logits = self.torch_model(batch)
                confs = F.softmax(logits,dim=1).cpu().detach().numpy()
                preds = np.argmax(confs,axis=1)
                logits = (logits.cpu().detach().numpy())

                self.train_confs.extend(np.max(confs,axis=1))
                self.train_preds.extend(preds)
                self.train_logits.extend(logits)

            self.test_preds = []
            self.test_confs = []
            self.test_logits = []
            for idx in range(0,len(data_test),batch_size):
                batch = torch.squeeze(torch.stack([x[0] for x in data_test[idx:idx+batch_size]]),dim=1).cuda()

                logits = self.torch_model(batch)
                confs = F.softmax(logits,dim=1).cpu().detach().numpy()
                preds = np.argmax(confs,axis=1)
                logits = (logits.cpu().detach().numpy())#**2)#.sum(axis=1)

                self.test_confs.extend(np.max(confs,axis=1))
                self.test_preds.extend(preds)
                self.test_logits.extend(logits)

            if self.verbose:
                print("Computing Train Set Min/Maxs:")
            self.compute_minmaxs(data_train)
            if self.verbose:
                print("Computing Test Set Deviations:")
            self.compute_test_deviations(data_test)

    def compute_batch_minmaxs(self, batch, POWERS=[1]):
        for PRED in self.pbar(self.classes):
            train_indices = np.where(np.array(self.train_preds.cpu())==PRED)[0]
            if len(train_indices) > 0:
                train_PRED = torch.squeeze(torch.stack([batch[i] for i in train_indices]),dim=1)
            else:
                train_PRED = []

            mins,maxs = self.torch_model.get_min_max(train_PRED,power=POWERS)
            self.mins[PRED] = mins
            self.maxs[PRED] = maxs
            torch.cuda.empty_cache()
                         
    def compute_minmaxs(self,data_train,POWERS=[1]):
        for PRED in self.pbar(self.classes):
            train_indices = np.where(np.array(self.train_preds)==PRED)[0]
            if len(train_indices) > 0:
                train_PRED = torch.squeeze(torch.stack([data_train[i][0] for i in train_indices]),dim=1)
            else:
                train_PRED = []
                
            mins,maxs = self.torch_model.get_min_max(train_PRED,power=POWERS)
            self.mins[PRED] = mins
            self.maxs[PRED] = maxs
            torch.cuda.empty_cache()

    def compute_test_deviations(self,data_test,POWERS=[1]):
        all_test_deviations = None
        all_test_deviations_msp = None
        for PRED in self.pbar(self.classes):
            test_indices = np.where(np.array(self.test_preds)==PRED)[0]
            if len(test_indices) > 0:
                test_PRED = torch.squeeze(torch.stack([data_test[i][0] for i in test_indices]),dim=1)
            else:
                test_PRED = []

            mins = self.mins[PRED]
            maxs = self.maxs[PRED]

            if len(test_PRED) != 0 and len(mins) != 0 and len(maxs) != 0:
                test_deviations = self.torch_model.get_deviations(test_PRED,power=POWERS,mins=mins,maxs=maxs)

                if all_test_deviations is None:
                    all_test_deviations = test_deviations
                else:
                    all_test_deviations = np.concatenate([all_test_deviations,test_deviations],axis=0)
                    
            torch.cuda.empty_cache()

        self.all_test_deviations = all_test_deviations
        self.target_deviation = torch.tensor(all_test_deviations.mean(axis=0).sum() * 0.85)

    def compute_ood_deviations_batch(self,ood,POWERS=[1],msp=False):
        ood_preds = []
        ood_confs = []

        for idx in range(0,len(ood),128):
            batch = torch.squeeze(torch.stack([x for x in ood[idx:idx+128]]),dim=1).cuda()
            logits = self.torch_model(batch)
            confs = F.softmax(logits,dim=1).cpu().detach().numpy()
            preds = np.argmax(confs,axis=1)

            ood_confs.extend(np.max(confs,axis=1))
            ood_preds.extend(preds)
            torch.cuda.empty_cache()

        all_ood_deviations = None
        all_ood_deviations_MSP = None
        all_ood_confs = []

        for PRED in self.classes:
            ood_indices = np.where(np.array(ood_preds)==PRED)[0]
            if len(ood_indices)==0:
                continue

            ood_PRED = torch.squeeze(torch.stack([ood[i] for i in ood_indices]),dim=1)

            ood_confs_PRED =  np.array([ood_confs[i] for i in ood_indices])

            all_ood_confs.extend(ood_confs_PRED)

            mins = self.mins[PRED]
            maxs = self.maxs[PRED]
            ood_deviations = self.torch_model.get_deviations(ood_PRED,power=POWERS,mins=mins,maxs=maxs)
            ood_deviations_MSP = ood_deviations/ood_confs_PRED[:,np.newaxis]

            if all_ood_deviations is None:
                all_ood_deviations = ood_deviations
                all_ood_deviations_MSP = ood_deviations_MSP
            else:
                all_ood_deviations = np.concatenate([all_ood_deviations,ood_deviations],axis=0)
                all_ood_deviations_MSP = np.concatenate([all_ood_deviations_MSP,ood_deviations_MSP],axis=0)
            torch.cuda.empty_cache()

        self.all_ood_confs = all_ood_confs

        average_results = detect(self.all_test_deviations,all_ood_deviations)

        return average_results

    def get_deviation(self, feat_list, idx, mins, maxs):
        batch_deviations = []
        for L,feat_L in enumerate(feat_list):
            dev = 0

            g_p = G_p(feat_L)[idx]

            dev +=  (F.relu(mins[L][0]-g_p)/torch.abs(mins[L][0]+10**-6)).sum(dim=1,keepdim=True)
            dev +=  (F.relu(g_p-maxs[L][0])/torch.abs(maxs[L][0]+10**-6)).sum(dim=1,keepdim=True)

            batch_deviations.append(dev)

        return batch_deviations

    def gram_loss(self, logits, feats):
        if not self.batch_detector:
            assert self.target_deviation is not None
        
        confs = F.softmax(logits, dim=1)

        _, indices = torch.max(confs, 1)

        loss = 0
        for i in range(10):
            idxs = indices == i

            if idxs.sum() == 0 or len(self.mins[i]) == 0 or len(self.maxs[i]) == 0:
                continue

            batch_dev = self.get_deviation(feats, idxs, mins=self.mins[i], maxs=self.maxs[i])
            batch_dev = torch.squeeze(torch.stack(batch_dev, dim=1))

            loss += batch_dev.sum()/logits.shape[0]
            
        if self.batch_detector:
            return F.relu(loss/(logits.shape[0] + 10**-6))
        else:
            return F.relu(loss - self.target_deviation)
    
    def compute_auroc_advs(self, adv_logits, adv_feats, adv_ys):
        confs = F.softmax(adv_logits,dim=1).cpu().numpy()
        preds = np.argmax(confs,axis=1)

        adv_deviations = None
        failed_adv_deviations = None

        for PRED in self.classes:
            idxs = np.where((np.array(preds) == PRED) & (np.array(adv_ys) != PRED))[0]
            idxs_failed = np.where((np.array(preds) == PRED) & (np.array(adv_ys) == PRED))[0]

            mins = self.mins[PRED]
            maxs = self.maxs[PRED]

            adv_dev_class = get_deviations(select_features(adv_feats,idxs), mins=mins, maxs=maxs)
            failed_adv_dev_class = get_deviations(select_features(adv_feats,idxs_failed), mins=mins, maxs=maxs)

            if len(adv_dev_class) > 0:
                if adv_deviations is None:
                    adv_deviations = adv_dev_class
                else:
                    adv_deviations = np.concatenate([adv_deviations, adv_dev_class],axis=0)
            
            if len(failed_adv_dev_class) > 0:
                if failed_adv_deviations is None:
                    failed_adv_deviations = failed_adv_dev_class
                else:
                    failed_adv_deviations = np.concatenate([failed_adv_deviations, failed_adv_dev_class], axis=0)

        failed_results, results = 0,0
        if failed_adv_deviations is not None and len(failed_adv_deviations) != 0:
            failed_results = detect(self.all_test_deviations, failed_adv_deviations)["AUROC"]

        if adv_deviations is not None and len(adv_deviations) != 0:
            results = detect(self.all_test_deviations, adv_deviations)["AUROC"]

        return results, failed_results

def gram_margin_loss(feats_reg, feats_adv, margin):
    assert len(feats_reg) == len(feats_adv)

    layer_deviations = torch.zeros((len(feats_reg), len(feats_reg[0])))
    for i in range(len(feats_reg)):
        g_p_reg = G_p(feats_reg[i])
        g_p_adv = G_p(feats_adv[i])
                
        clamp = torch.tensor(1.0).cuda()
        
        orig_max = torch.max(g_p_reg, dim=1)[0]
        clamped_max = torch.max(torch.abs(orig_max), clamp)
               
        max_dist = F.relu(torch.max(g_p_adv, dim=1)[0] - orig_max)/clamped_max
        
        orig_min = torch.min(g_p_reg, dim=1)[0]
        clamped_min = torch.max(torch.abs(orig_min), clamp)
        
        min_dist = F.relu(orig_min - torch.min(g_p_adv, dim=1)[0])/clamped_min
        
        layer_deviations[i] = max_dist + min_dist
        
    return F.relu(margin - layer_deviations.sum(dim=0)).pow(2).mean()

def feats_idxs(feats, idxs):
    return [f[idxs] for f in feats]

def new_gram_margin_loss(feats_reg, output_reg, feats_adv, output_adv, golds, margin=20):
    indices_reg = torch.max(output_reg.cpu(), 1)[1]
    indices_adv = torch.max(output_adv.cpu(), 1)[1]
    
    total_size = 0
    loss = []
    for i in range(10):
        idxs_adv = torch.where(i == indices_adv)[0]
        idxs_reg = torch.where(i == golds)[0]
        
        batch_size = min(idxs_adv.shape[0], idxs_reg.shape[0])
        if batch_size == 0:
            continue
            
        if batch_size < idxs_adv.shape[0]:
            idxs_reg = torch.cat([idxs_reg] * 10)
            batch_size = idxs_adv.shape[0]
            if idxs_reg.shape[0] < batch_size:
                continue
                
            idxs_reg = idxs_reg[:batch_size]
        total_size += batch_size
        
        idxs_adv, idxs_reg = idxs_adv[:batch_size], idxs_reg[:batch_size]
        
        class_loss = gram_margin_loss(feats_idxs(feats_reg, idxs_reg), feats_idxs(feats_adv, idxs_adv), margin)
        loss.append(class_loss)
    
    return torch.stack(loss).sum()

def calc_auroc(all_test_deviations,all_ood_deviations):
    average_results = {}

    test_deviations = all_test_deviations.sum(axis=1)
    ood_deviations = all_ood_deviations.sum(axis=1)

    results = callog.compute_metric(-test_deviations,-ood_deviations)

    return results["AUROC"]

def select_features(feat_list, idxs):
    return [f[idxs] for f in feat_list]
    
def get_deviations(feat_list, mins,maxs):
    if len(feat_list[0]) == 0:
        return np.array([])
    deviations = []
    for L,feat_L in enumerate(feat_list):
        
        g_p = G_p(feat_L)

        dev =  (F.relu(mins[L][0]-g_p)/torch.abs(mins[L][0]+10**-6)).sum(dim=1,keepdim=True)
        dev +=  (F.relu(g_p-maxs[L][0])/torch.abs(maxs[L][0]+10**-6)).sum(dim=1,keepdim=True)

        deviations.append(dev.cpu().numpy())
            
    deviations = np.concatenate(deviations, axis=1)
    
    return deviations

def gram_matrix(layer):
    b, ch, h, w = layer.size()
    features = layer.view(b, ch, w * h)
    gram = torch.matmul(features, features.transpose(1, 2))
    
    return gram /(ch * h * w)

def style_loss(lhs, rhs):
    loss = 0.0
    for i in range(len(lhs)):
        loss += (gram_matrix(lhs[i]) - gram_matrix(rhs[i])).pow(2).sum()
    
    return loss
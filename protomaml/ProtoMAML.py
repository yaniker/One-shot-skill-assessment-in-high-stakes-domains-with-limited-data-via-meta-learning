"""
Author: Dr. Erim Yanik
Date: 04/05/2024

Parts of this script was taken and/or adjusted from "https://pytorch-lightning.readthedocs.io/en/1.5.10/notebooks/course_UvA-DL/12-meta-learning.html"
    Author: Phillip Lippe
    License: CC BY-SA
    Generated: 2021-10-10T18:35:50.818431
"""

import os, random
import numpy as np
from collections import defaultdict
from copy import deepcopy
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

from utils import to_tensor, split_batch, zero_padder, X_normalize
# from utils_model import *
from model import model

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class ProtoMAML(pl.LightningModule):
    def __init__(self, loss_type, weight_decay, inp_size,
                 lr, lr_inner, lr_output, num_inner_steps):
        """
        Inputs
            loss_type (str) - Cross Entropy or Cosine Loss
            weight_decay (float): Decides weight decay during backpropagation.
            inp_size (int): Communicates the input feature size to the DL model.
            lr (float): Learning rate of the outer loop Adam optimizer.
            lr_inner (float): Learning rate of the inner loop SGD optimizer.
            lr_output (float): Learning rate for the output layer in the inner loop.
            num_inner_steps (int): Number of inner loop updates to perform.
        """
        
        super().__init__()
        self.save_hyperparameters()
        self.model = model(inp_size)
        
    def calculate_prototypes(self, features, targets):
        classes, _ = torch.unique(targets).sort()  # Determine which classes we have.
        prototypes = []
        for c in classes:
            p = features[torch.where(targets == c)[0]].mean(dim=0)  # Average class feature vectors.
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
        # Return the 'classes' tensor to know which prototype belongs to which class.
        return prototypes, classes
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay = self.hparams.weight_decay)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max',
                                                         factor = 0.6, patience = 10,
                                                         threshold = 1e-5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_acc'}
    
    def run_model(self, local_model, output_weight, output_bias, imgs, labels):
        # Execute a model with given output layer weights and inputs.
        feats = local_model(imgs)
        preds = F.linear(feats, output_weight, output_bias)
        #Cross entropy loss.
        if self.hparams.loss_type == 'CE': loss = F.cross_entropy(preds, labels)
        #Cosine similarity loss.
        elif self.hparams.loss_type == 'CS': 
            labels_hot = F.one_hot(labels, num_classes = preds.shape[1])
            loss = 1 - torch.sum(F.cosine_similarity(torch.softmax(preds, dim = 1),
                                                                   labels_hot,
                                                                   dim = 0))*(1/preds.shape[0])
        #Compute accuracy and return it with loss and predictions.
        acc = (preds.argmax(dim=1) == labels).float()
        return loss, preds, acc

    def adapt_few_shot(self, support_imgs, support_targets):
        # Determine prototype initialization.
        support_feats = self.model(support_imgs)
        prototypes, classes = self.calculate_prototypes(support_feats, support_targets)
        support_labels = (classes[None,:] == support_targets[:,None]).long().argmax(dim=-1)
        # Create inner-loop model and optimizer
        local_model = deepcopy(self.model)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.hparams.lr_inner)
        local_optim.zero_grad()
        # Create output layer weights with prototype-based initialization.
        init_weight = 2 * prototypes
        init_bias = -torch.norm(prototypes, dim=1)**2
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()

        # Optimize inner loop model on support set.
        for _ in range(self.hparams.num_inner_steps):
            # Determine loss on the support set.
            loss, _, _ = self.run_model(local_model, output_weight, output_bias, support_imgs, 
                                        support_labels)            
            # Calculate gradients and perform inner loop update.
            loss.backward()
            local_optim.step()
            # Update output layer via SGD.
            output_weight.data -= self.hparams.lr_output * output_weight.grad
            output_bias.data -= self.hparams.lr_output * output_bias.grad
            # Reset gradients.
            local_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)

        # Re-attach computation graph of prototypes.
        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias

        return local_model, output_weight, output_bias, classes

    def outer_loop(self, batch, mode="train"):
        accuracies = []
        losses = []
        self.model.zero_grad()

        # Determine gradients for batch of tasks.
        for task_batch in batch:
            imgs, targets = task_batch
            support_imgs, query_imgs, support_targets, query_targets = split_batch(Xs = imgs,
                                                                                   ys = targets)
            # Perform inner loop adaptation.
            local_model, output_weight, output_bias, classes = self.adapt_few_shot(support_imgs, 
                                                                                   support_targets)
            # Determine loss of query set.
            query_labels = (classes[None,:] == query_targets[:,None]).long().argmax(dim=-1)
            loss, preds, acc = self.run_model(local_model, output_weight, output_bias, query_imgs, 
                                              query_labels)
            # Calculate gradients for query set loss.
            if mode == "train":
                loss.backward()

                for p_global, p_local in zip(self.model.parameters(), local_model.parameters()):
                    p_global.grad += p_local.grad  # First-order approx. -> add gradients of finetuned and base model.

            accuracies.append(acc.mean().detach())
            losses.append(loss.detach())

        if mode == "train":
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()

        self.log(f"{mode}_loss", sum(losses) / len(losses), prog_bar = True)
        self.log(f"{mode}_acc", sum(accuracies) / len(accuracies), prog_bar = True)

    def training_step(self, batch, batch_idx):
        self.outer_loop(batch, mode="train")
        return None

    def validation_step(self, batch, batch_idx):
        # Validation requires to finetune a model, hence we need to enable gradients.
        torch.set_grad_enabled(True)
        self.outer_loop(batch, mode="val")
        torch.set_grad_enabled(False)
              
class TaskBatchSampler(object):
    def __init__(self, dataset_targets, batch_size, N_WAY, K_SHOT, include_query=False, 
                 shuffle=True):
        """
        Inputs:
            dataset_targets - PyTorch tensor of the labels of the data elements.
            batch_size - Number of tasks to aggregate in a batch
            N_way - Number of classes to sample per batch.
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training)
        """
        super().__init__()
        self.batch_sampler = FewShotBatchSampler(dataset_targets, N_WAY, K_SHOT, 
                                                 include_query, shuffle)
        self.task_batch_size = batch_size
        self.local_batch_size = self.batch_sampler.batch_size
        self.dataset_targets = dataset_targets
                
    def __iter__(self):
        # Aggregate multiple batches before returning the indices.
        batch_list = []
        for batch_idx, batch in enumerate(self.batch_sampler):
            batch_list.extend(batch)            
            if (batch_idx+1) % self.task_batch_size == 0:
                yield batch_list
                batch_list = []

    def __len__(self):
        return len(self.batch_sampler)//self.task_batch_size

    def get_collate_fn(self):
        # Returns a collate function that converts one big tensor into a list of task-specific tensors.
        def collate_fn(item_list):
            imgs = np.stack([img for img, target in item_list], axis=0)
            imgs = to_tensor(zero_padder(imgs))            
            for i in range(len(imgs)): imgs[i] = X_normalize(imgs[i]) #Mini-batch zero-padding.
            targets = torch.stack([target for img, target in item_list], dim=0)
            imgs = imgs.chunk(self.task_batch_size, dim=0)
            targets = targets.chunk(self.task_batch_size, dim=0)
            return list(zip(imgs, targets))
        return collate_fn
    
class FewShotBatchSampler(object):
    def __init__(self, dataset_targets, N_WAY, K_SHOT, include_query=False, shuffle=True, 
                 shuffle_once=False):
        """
        Inputs:
            dataset_targets - PyTorch tensor of the labels of the data elements.
            N_way - Number of classes to sample per batch.
            K_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size N_way*K_shot*2, which
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training)
            shuffle_once - If True, examples and classes are shuffled once in
                           the beginning, but kept constant across iterations
                           (for validation)
        """
        
        super().__init__()
        self.dataset_targets = dataset_targets
        self.N_WAY = N_WAY
        self.K_SHOT = K_SHOT
        self.shuffle = shuffle
        self.include_query = include_query
        if self.include_query:
            self.K_SHOT *= 2
        self.batch_size = self.N_WAY * self.K_SHOT  # Number of overall images per batch.

        # Organize examples by class.
        self.classes = torch.unique(self.dataset_targets).tolist()
        self.num_classes = len(self.classes)
        self.indices_per_class = {}
        self.batches_per_class = {}  # Number of K-shot batches that each class can provide.
        for c in self.classes:
            #Indices of where the sampels of each class are.
            self.indices_per_class[c] = torch.where(self.dataset_targets == c)[0]
            self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.K_SHOT

        # Create a list of classes from which we select the N classes per batch.        
        self.iterations = sum(self.batches_per_class.values()) // self.N_WAY
        self.class_list = [c for c in self.classes for _ in range(self.batches_per_class[c])]
            
        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            # For testing, we iterate over classes instead of shuffling them.
            sort_idxs = [i+p*self.num_classes for i,
                         c in enumerate(self.classes) for p in range(self.batches_per_class[c])]
            self.class_list = np.array(self.class_list)[np.argsort(sort_idxs)].tolist()
            
    def shuffle_data(self):
        # Shuffle the examples per class.
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]
        # Shuffle the class list from which we sample. Note that this way of shuffling does not prevent to choose the same class twice in a batch. However, for training and validation, this is not a problem.
        random.shuffle(self.class_list)
        
    def __iter__(self):
        # Shuffle data to prevent bias.
        if self.shuffle:
            self.shuffle_data()

        # Sample few-shot batches.
        start_index = defaultdict(int)
        for it in range(self.iterations):
            class_batch = self.class_list[it*self.N_WAY:(it+1)*self.N_WAY]  # Select N classes for the batch.              
            index_batch = []
            for c in class_batch:  # For each class, select the next K examples and add them to the batch.
                index_batch.extend(self.indices_per_class[c][start_index[c]:start_index[c]+self.K_SHOT])
                start_index[c] += self.K_SHOT
            if self.include_query:  # If we return support+query set, sort them: they are easy to split.
                index_batch = index_batch[::2] + index_batch[1::2]
            yield index_batch

    def __len__(self):
        return self.iterations

def train_model(train_set, val_set, batch_size, N_WAY, K_SHOT, loss_type, inp_size,
                weight_decay, CHECKPOINT_PATH, seed, patience, max_epochs, min_epochs, **kwargs):
        
    model_class = ProtoMAML
    # Training set.
    train_protomaml_sampler = TaskBatchSampler(train_set.targets,
                                               include_query = True,
                                               N_WAY = N_WAY,
                                               K_SHOT = K_SHOT,
                                               batch_size = batch_size
                                              )
    train_loader = data.DataLoader(train_set,
                                   batch_sampler = train_protomaml_sampler,
                                   collate_fn = train_protomaml_sampler.get_collate_fn(),
                                   num_workers = 1
                                  )
    
    # Validation set.
    val_protomaml_sampler = TaskBatchSampler(val_set.targets,
                                             include_query = True,
                                             N_WAY = N_WAY,
                                             K_SHOT = K_SHOT,
                                             batch_size = 1,
                                             shuffle = False
                                            )
    val_loader = data.DataLoader(val_set,
                                 batch_sampler=val_protomaml_sampler,
                                 collate_fn=val_protomaml_sampler.get_collate_fn(),
                                 num_workers=1
                                )
    
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_class.__name__),
                         gpus=1 if str(device) == "cuda:0" else 0,
                         max_epochs = max_epochs,
                         min_epochs = min_epochs,
                         num_sanity_val_steps = 2,
                         callbacks=[ModelCheckpoint(save_weights_only = True, mode = "max", 
                                                    monitor = "val_acc"),
                                    LearningRateMonitor("epoch"),
                                    EarlyStopping(monitor = "val_acc", mode = "max", 
                                                  patience = patience)
                                   ]
                        )
    trainer.logger._default_hp_metric = None
    
    pl.seed_everything(seed)  # To be reproducable.
    model = model_class(loss_type, weight_decay, inp_size, **kwargs)
    trainer.fit(model, train_loader, val_loader)
    model = model_class.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training.
    return model

def test_model(model, dataset_support, dataset_query, seed, K_SHOT):
    """
    Evaluates a model's performance in a few-shot learning setting using both support and query datasets. 

    Inputs:
        model: The neural network model to be tested.
        dataset_support: Dataset providing support examples for few-shot adaptation.
        dataset_query: Dataset serving as the query set for model evaluation.
        seed (int): Seed for reproducibility of results.
        K_SHOT (int): Number of examples per class used for adaptation.

    Returns:
        YACT (list): Actual labels of the query dataset, with 'na' for support set instances.
        YPRED (list): Predicted labels for the query dataset, with 'na' for support set instances.
        YPRED_SM (list): Softmax probabilities for the query dataset predictions, with 'na' for support set instances.
    """
    
    
    pl.seed_everything(seed)
    model = model.to(device)
    num_classes = dataset_support.targets.unique().shape[0]
    
    # Data loader for full test set as query set.
    full_dataloader = data.DataLoader(dataset_query,
                                      batch_size=len(dataset_query.targets),
                                      num_workers=1,
                                      shuffle=False,
                                      drop_last=False)

    sampler = FewShotBatchSampler(dataset_support.targets,
                                  include_query=False,
                                  N_WAY = num_classes,
                                  K_SHOT = K_SHOT,
                                  shuffle=False,
                                  shuffle_once=False)
    
    sample_dataloader = data.DataLoader(dataset_support,
                                        batch_sampler=sampler,
                                        num_workers=1)
    
    # We iterate through the full dataset in two manners. First, to select the k-shot batch. Second, we evaluate the model on all other examples.
    accuracies, YACT, YPRED, YPRED_SM = [], [], [], []
    for (support_imgs, support_targets), support_indices in tqdm(zip(sample_dataloader, sampler), 
                                                                 "Performing few-shot finetuning"):
        support_imgs = support_imgs.to(device)
        support_targets = support_targets.to(device)
                
        # Finetune new model on support set.
        local_model, output_weight, output_bias, classes = model.adapt_few_shot(support_imgs, 
                                                                                support_targets)
        
        with torch.no_grad():  # No gradients for query set needed.
            local_model.eval()
            
            # Evaluate all examples in test dataset.
            PREDS, PREDS_SM, LABELS = [], [], []
            for query_imgs, query_targets in full_dataloader:                
                query_imgs = query_imgs.to(device)
                query_targets = query_targets.to(device)
                query_labels = (classes[None,:] == query_targets[:,None]).long().argmax(dim=-1)
                _, preds, acc = model.run_model(local_model, output_weight, output_bias, 
                                                query_imgs, 
                                                query_labels)
                LABELS.extend(query_labels.cpu().numpy())
                PREDS.extend(preds.argmax(dim=1).cpu().numpy())     
                PREDS_SM.extend(torch.softmax(preds, dim=1).cpu().numpy())
                
            # Exclude support set elements.
            for s_idx in support_indices: PREDS[s_idx], PREDS_SM[s_idx], LABELS[s_idx] =\
                                          'na', 'na', 'na'
            YACT.extend(LABELS), YPRED.extend(PREDS), YPRED_SM.extend(PREDS_SM)
            
    return YACT, YPRED, YPRED_SM







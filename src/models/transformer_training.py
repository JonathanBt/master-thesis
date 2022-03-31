import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")
import random
import time
import datetime

# Evaluation
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)

# Transformers
import torch
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import RAdam
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup 
import wandb


def train_epoch(model, optimizer, scheduler, train_dataloader, val_dataloader, loss_fct_train, loss_fct_val, device):
    """This function trains the given model for one epoch.
    Args:
        model (torch.model): The model to train.
	optimizer (torch.optimizer): The optimizer to use.
	scheduler (transformers.scheduler): The scheduler to use.
	train_dataloader (torch.DataLoader): The training dataloader.
	val_dataloader (torch.DataLoader): The validation dataloader.
	loss_fct_train (torch.loss): The loss function to use during training (include weight balancing).
	loss_fct_val (torch.loss): The loss funcion to use during validation (no weight balancing).
	device (torch.device): The device to use (CPU or GPU). 

    Returns:
        avg_train_loss (float): The average training loss of the epoch.
	avg_val_loss (float): The average validation loss of the epoch.
	avg_val_f1 (float): The average macro F1-score of the validation epoch.
    """

    # ========================================
    #               Training
    # ========================================

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        # Unpack the batch from the dataloader and copy each tensor to the GPU
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a backward pass
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch) and return the loss (because we provided labels) and the logits
        result = model(b_input_ids, 
                       token_type_ids=None, 
                       attention_mask=b_input_mask, 
                       labels=b_labels,
                       return_dict=True)

        logits = result.logits
        loss = loss_fct_train(logits, b_labels)

        # Accumulate the training loss over all of the batches so that we can calculate the average loss at the end
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients
        loss.backward()

        # Clip the norm of the gradients to 1.0. to help prevent the "exploding gradients" problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update the learning rate
        scheduler.step()

    # Calculate the average loss over all of the batches
    avg_train_loss = total_train_loss / len(train_dataloader)            
        
    # ========================================
    #               Validation
    # ========================================

    model.eval()
    
    predictions , true_labels = [], []

    total_eval_f1 = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in val_dataloader:
        
        # Unpack the batch from the dataloader and copy each tensor to the GPU
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during the forward pass, since this is only needed for backprop (training)
        with torch.no_grad():        

            # Forward pass, calculate logit predictions
            result = model(b_input_ids, 
                           token_type_ids=None, 
                           attention_mask=b_input_mask,
                           labels=b_labels,
                           return_dict=True)

        # Get the loss and "logits" output by the model
        logits = result.logits
        loss = loss_fct_val(logits, b_labels)
            
        # Accumulate the validation loss
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
        
    # Report the final macro F1 score for this validation run
    predicted_labels = torch.round(torch.sigmoid(torch.Tensor(np.concatenate(predictions, axis=0)))).numpy().astype(int)
    true_labels = np.concatenate(true_labels, axis=0).astype(int)
    avg_val_f1 = f1_score(predicted_labels, true_labels, average='macro') 
    
    # Calculate the average loss over all of the batches
    avg_val_loss = total_eval_loss / len(val_dataloader)
    
    # Return statistics
    return avg_train_loss, avg_val_loss, avg_val_f1


def calculate_pos_weights(labels):
    """This function calculates the weights for the positive class for the BCEWIthLogitsLoss.
    Args:
        labels (np.array): Training set labels.

    Returns:
        torch.Tensor containing the class weights.
    """

    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    pos_weights = neg_counts / (pos_counts + 1e-5)

    return torch.as_tensor(pos_weights, dtype=torch.float)


def format_time(elapsed):
    """This function formats a time in seconds into a string hh:mm:ss.   
    Args:
        elapsed (datetime.time): Time in seconds.

    Returns:
        String containing the time in format hh:mm:ss.
    """

    elapsed_rounded = int(round((elapsed)))

    return str(datetime.timedelta(seconds=elapsed_rounded))
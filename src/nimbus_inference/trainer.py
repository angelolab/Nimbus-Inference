from nimbus_inference.utils import LmdbDataset, segment_mean
from nimbus_inference.nimbus import Nimbus
from alpineer import misc_utils
from pathlib import Path
from tqdm.notebook import tqdm
import nimbus_inference
from copy import deepcopy
import pandas as pd
import numpy as np
import torch
import lmdb
import os
import kornia.augmentation as K


class AugmentationPipeline:
    """Augmentation pipeline for input, mask, and label tensors.

    Args:
        p (float): Probability of applying augmentations.
    """
    def __init__(self, p=0.5):
        # Geometric transforms - applied to all with appropriate interpolation
        self.geometric_transforms = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=p),
            K.RandomVerticalFlip(p=p),
            K.RandomRotation(
                degrees=45.0, p=p,
                same_on_batch=False
            ),
            data_keys=["input", "mask", "mask"]
        )

        # Intensity transforms - applied only to input
        self.intensity_transforms = K.AugmentationSequential(
            K.ColorJiggle(
                brightness=(0.8, 1.2),
                contrast=(0.8, 1.2),
                p=p
            ),
            K.RandomGaussianNoise(
                mean=0.0, 
                std=0.05, 
                p=p
            ),
            data_keys=["input"]
        )

    def __call__(self, input_tensor, mask_tensor, label_tensor):
        """Apply augmentations to input, mask, and label tensors.
    
        Args:
            input_tensor (torch.Tensor): Input tensor.
            mask_tensor (torch.Tensor): Mask tensor.
            label_tensor (torch.Tensor): Label tensor.
        
        Returns:
            tuple: Tuple containing augmented input, mask, and label tensors.
        """
        # Apply geometric transforms to all tensors
        input_t, mask_t, label_t = self.geometric_transforms(
            input_tensor, mask_tensor, label_tensor
        )
        
        # Apply intensity transforms only to input
        input_t[:,0] = self.intensity_transforms(input_t[:,0])        
        return input_t, mask_t, label_t


class SmoothBinaryCELoss(torch.nn.Module):
    def __init__(self, label_smoothing=0.05):
        super(SmoothBinaryCELoss, self).__init__()
        if not 0 <= label_smoothing < 1:
            raise ValueError("Label smoothing must be in [0,1)")
        self.label_smoothing = label_smoothing
        self.eps = 1e-12  # For numerical stability

    def forward(self, inputs, targets):        
        # Clamp for numerical stability
        inputs = torch.clamp(inputs, self.eps, 1.0 - self.eps)
        # get mask which is not (targets == 0 and targets == 1)
        mask = torch.clip(targets, 0, 2) == 2
        # Apply label smoothing
        targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        # Binary cross entropy
        loss = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        # Mask out loss for ignored pixels
        loss[mask] = 0
        return loss


class Trainer:
    """Trainer class for fine-tuning a nimbus model on a new dataset.

    Args:
        nimbus (Nimbus): Nimbus object containing the model to be fine-tuned.
        train_dataset (torch.utils.data.Dataset): Training dataset.
        validation_dataset (torch.utils.data.Dataset): Validation dataset.
        checkpoint_name (str): Name of the checkpoint file to be saved.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        initial_regularization (float): Regularization factor for not deviating too much from the
            initial checkpoint. Set to None to disable.
        num_workers (int): Number of workers for the data loader.
        label_smoothing (float): Label smoothing factor for the loss function.
        gradient_clip (float): Gradient clipping value.
        patience (int): Patience for early stopping.
        augmentation_p (float): Probability of applying augmentations.
    """
    def __init__(
            self, nimbus: Nimbus, train_dataset: LmdbDataset, validation_dataset: LmdbDataset,
            checkpoint_name: str, batch_size: int=4, learning_rate: float=1e-5,
            weight_decay: float=1e-4, initial_regularization: float=1e-4, num_workers: int=4,
            label_smoothing: float=0.05, gradient_clip: float=1.0, patience: int=5,
            augmentation_p: float=0.5,
        ):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.batch_size = batch_size
        self.model = nimbus.model
        self.device = self.model.device
        self.initial_regularization = initial_regularization
        if self.initial_regularization:
            self.initial_checkpoint = deepcopy(self.model)
        self.checkpoint_name = checkpoint_name
        self.augmenter = AugmentationPipeline(p=augmentation_p)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,  # First cycle length (epochs)
            T_mult=2,  # Multiply cycle length by 2 after each restart
            eta_min=1e-7  # Minimum learning rate
        )
        use_pin_memory = self.device.type == "cuda" or self.device.type == "mps"
        self.loss_function = SmoothBinaryCELoss(label_smoothing=label_smoothing)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            drop_last=True, num_workers=num_workers, pin_memory=use_pin_memory
        )
        self.validation_loader = torch.utils.data.DataLoader(
            self.validation_dataset, batch_size=self.batch_size, shuffle=False,
            drop_last=False, pin_memory=use_pin_memory
        )
        self.gradient_clip = gradient_clip
        self.patience = patience
        self.best_f1 = 0.0
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }

    def run_validation(self):
        """Run validation on the validation dataset."""
        self.model.eval()
        loss_ = []
        df_list = []
        for inputs, labels, inst_mask, key in self.validation_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss_.append(loss.mean().item())
            # calculate mean per instance prediction
            outputs = outputs.cpu().numpy()
            inst_mask = inst_mask.cpu().numpy()
            labels = labels.cpu().numpy()
            # split batch to get individual samples
            for i in range(outputs.shape[0]):
                # split key to get fov, channel_name, i, j f"{fov}_,_{channel_name}_,_{i}_,_{j}"
                pred_df = pd.DataFrame(segment_mean(inst_mask[i], outputs[i]))
                gt_df = pd.DataFrame(segment_mean(inst_mask[i], labels[i]))
                # split key to get fov, channel_name, i, j f"{fov}_,_{channel_name}_,_{i}_,_{j}"
                fov, channel, i, j = key[i].split("_,_")
                for df in [pred_df, gt_df]:
                    df["fov"] = fov
                    df["channel"] = channel
                gt_df.rename(columns={"intensity_mean": "gt"}, inplace=True)
                pred_df.rename(columns={"intensity_mean": "pred"}, inplace=True)
                # merge dataframes to one based on fov, channel, label
                merged_df = pd.merge(gt_df, pred_df, on=["fov", "channel", "label"], how="inner")
                df_list.append(merged_df)
        df = pd.concat(df_list)
        metrics = {}    
        # Calculate metrics per channel
        for channel in df['channel'].unique():
            channel_df = df[df['channel'] == channel]
            
            # Convert to binary predictions using 0.5 threshold
            y_true = (channel_df['gt'] > 0.5).astype(int)
            y_pred = (channel_df['pred'] > 0.5).astype(int)
            
            # Calculate basic counts
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[channel] = {
                'precision': precision,
                'recall': recall, 
                'specificity': specificity,
                'f1': f1
            }
        
        # Add mean loss
        metrics['loss'] = np.mean(loss_)
        metrics['mean_f1'] = np.mean([m['f1'] for m in metrics.values() if isinstance(m, dict)])
        return metrics
    
    def initial_checkpoint_regularizer(self):
        """Regularize model during training so that it does not deviate too much from the initial
        checkpoint.

        Returns:
            torch.Tensor: Regularization loss.
        """
        regularization_loss = 0
        for p_init, p in zip(self.initial_checkpoint.parameters(), self.model.parameters()):
            regularization_loss +=  torch.abs(p_init - p).mean()
        return regularization_loss * self.initial_regularization

    def save_checkpoint(self):
        """Save the model checkpoint to a file."""
                # Set up paths
        path = os.path.dirname(nimbus_inference.__file__)
        path = Path(path).resolve()
        local_dir = os.path.join(path, "assets")
        os.makedirs(local_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(local_dir, self.checkpoint_name + ".pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': self.best_f1,
            'history': self.history
        }, self.checkpoint_path)

    def train(self, epochs: int):
        """Train the model for specified number of epochs.
        
        Args:
            epochs (int): Number of epochs to train the model.
        """
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []
            
            for inputs, labels, inst_mask, key in tqdm(self.train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs, inst_mask, labels = self.augmenter(inputs, inst_mask, labels)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                if self.initial_regularization:
                    loss += self.initial_checkpoint_regularizer()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip
                )
                
                self.optimizer.step()
                train_losses.append(loss.item())
            
            # Validation phase
            val_metrics = self.run_validation()
            
            # Update learning rate
            self.scheduler.step()
            
            # Save best model
            if val_metrics['mean_f1'] > self.best_f1:
                self.best_f1 = val_metrics['mean_f1']
                self.epochs_without_improvement = 0
                self.save_checkpoint()
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Update history
            self.history['train_loss'].append(np.mean(train_losses))
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_metrics'].append(val_metrics)
            
            self._print_epoch_summary(epoch, train_losses, val_metrics)

    def _print_epoch_summary(self, epoch: int, train_losses: list, val_metrics: dict):
        """Print epoch training summary.
        
        Args:
            epoch (int): Current epoch number.
            train_losses (list): List of training losses.
            val_metrics (dict): Dictionary containing validation metrics.
        """
        print(f"Epoch {epoch+1}:")
        print(f"Train Loss: {np.mean(train_losses):.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Mean F1: {val_metrics['mean_f1']:.4f}")
        print("Learning rate:", self.optimizer.param_groups[0]['lr'])
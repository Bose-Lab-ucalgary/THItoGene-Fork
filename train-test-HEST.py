import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# coding:utf-8 
import random
import argparse
import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from torch.utils.data import DataLoader

from dataset import ViT_HEST1K, ViT_HER2ST, ViT_SKIN
from predict import model_predict
from utils import *
from vis_model import THItoGene
from config import GENE_LISTS

torch.set_float32_matmul_precision('medium')

# Set all seeds for reproducibility
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)

# Set the default seed at import
set_all_seeds(42)


def train(test_sample_ID=0, vit_dataset=ViT_HEST1K, epochs=300, modelsave_address="model", 
          dataset_name="hest1k", gene_list="3CA", num_workers=16, gpus=1, strategy=None, 
          learning_rate=1e-5, batch_size=1, patience=20):
    
    # Get number of genes from config
    n_genes = GENE_LISTS[gene_list]["n_genes"]
    
    tagname = gene_list + '_' + dataset_name + '_' + str(n_genes)
    model = THItoGene(n_genes=n_genes, learning_rate=learning_rate, route_dim=64, caps=20, heads=[16, 8], n_layers=4)
    
    # Create datasets
    dataset_train = vit_dataset(mode='train', gene_list=gene_list)
    dataset_val = vit_dataset(mode='val', gene_list=gene_list)
    dataset_test = vit_dataset(mode='test', gene_list=gene_list)
    
    # Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=modelsave_address,
        filename=f"best_model_{tagname}_{test_sample_ID}" + "_{epoch:02d}_{val_loss:.2f}",
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min',
        verbose=True
    )
    
    # Setup logger
    mylogger = CSVLogger(save_dir=modelsave_address + "/../logs/",
                         name="my_test_log_" + tagname + '_' + str(test_sample_ID))
    
    memory_monitor = MemoryMonitorCallback()
    
    # Setup trainer with configurable GPU settings
    trainer_kwargs = {
        'max_epochs': epochs,
        'logger': mylogger,
        'callbacks': [checkpoint_callback, early_stopping, memory_monitor],
        'check_val_every_n_epoch': 1,
        'log_every_n_steps': 10
    }
    
    # Configure accelerator and devices based on GPU count
    if gpus > 0:
        trainer_kwargs['accelerator'] = "gpu"
        if gpus == 1:
            trainer_kwargs['devices'] = [0]
        else:
            trainer_kwargs['devices'] = gpus
            if strategy:
                trainer_kwargs['strategy'] = strategy
    else:
        trainer_kwargs['accelerator'] = "cpu"
    
    # trainer_kwargs['enable_oom_protection'] = True  # Removed: not supported in this Lightning version
    
    trainer = pl.Trainer(**trainer_kwargs)

    # Train with validation
    trainer.fit(model, train_loader, val_loader)

    # Test on the best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")
    
    # Load best model for testing
    best_model = THItoGene.load_from_checkpoint(best_model_path, n_genes=n_genes)
    
    # Run test predictions
    pred, gt = trainer.predict(model=best_model, dataloaders=test_loader)[0]
    R, p_val = get_R(pred, gt)
    pred.var["p_val"] = p_val
    pred.var["-log10p_val"] = -np.log10(p_val)

    print('Mean Pearson Correlation:', np.nanmean(R))
    
    # Save final checkpoint
    trainer.save_checkpoint(modelsave_address+"/"+"last_train_"+tagname+'_'+str(test_sample_ID)+".ckpt")
    
    return pred, gt, R, p_val


def test(test_sample_ID=0, vit_dataset=ViT_HEST1K, model_address="model", dataset_name="hest1k", 
         gene_list="3CA", checkpoint_path=None, num_workers=4, gpus=1):
    
    # Get number of genes from config
    n_genes = GENE_LISTS[gene_list]["n_genes"]
    
    tagname = gene_list + '_' + dataset_name + '_' + str(n_genes)
    
    # Load model from checkpoint
    if checkpoint_path is None:
        checkpoint_path = model_address + "/" + "last_train_" + tagname + '_' + str(test_sample_ID) + ".ckpt"
    
    print(f"Loading model from: {checkpoint_path}")
    model = THItoGene.load_from_checkpoint(checkpoint_path, n_genes=n_genes)
    
    # Create test dataset
    dataset_test = vit_dataset(mode='test', gene_list=gene_list)
    test_loader = DataLoader(dataset_test, batch_size=1, num_workers=num_workers, shuffle=False)
    
    # Setup trainer for testing
    trainer_kwargs = {}
    if gpus > 0:
        trainer_kwargs['accelerator'] = "gpu"
        trainer_kwargs['devices'] = [0] if gpus == 1 else gpus
    else:
        trainer_kwargs['accelerator'] = "cpu"
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Run predictions
    predictions = trainer.predict(model=model, dataloaders=test_loader)
    pred, gt = predictions[0]
    
    # Calculate correlations
    R, p_val = get_R(pred, gt)
    pred.var["p_val"] = p_val
    pred.var["-log10p_val"] = -np.log10(p_val)

    print('Test Mean Pearson Correlation:', np.nanmean(R))
    
    return pred, gt, R, p_val


def validate(test_sample_ID=0, vit_dataset=ViT_HEST1K, model_address="model", dataset_name="hest1k", 
            gene_list="3CA", checkpoint_path=None, num_workers=4, gpus=1):
    """Run validation on the validation set"""
    
    # Get number of genes from config
    n_genes = GENE_LISTS[gene_list]["n_genes"]
    
    tagname = gene_list + '_' + dataset_name + '_' + str(n_genes)
    
    # Load model from checkpoint
    if checkpoint_path is None:
        checkpoint_path = model_address + "/" + "last_train_" + tagname + '_' + str(test_sample_ID) + ".ckpt"
    
    print(f"Loading model from: {checkpoint_path}")
    model = THItoGene.load_from_checkpoint(checkpoint_path, n_genes=n_genes)
    
    # Create validation dataset
    dataset_val = vit_dataset(mode='val', gene_list=gene_list)
    val_loader = DataLoader(dataset_val, batch_size=1, num_workers=num_workers, shuffle=False)
    
    # Setup trainer for validation
    trainer_kwargs = {}
    if gpus > 0:
        trainer_kwargs['accelerator'] = "gpu"
        trainer_kwargs['devices'] = [0] if gpus == 1 else gpus
    else:
        trainer_kwargs['accelerator'] = "cpu"
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Run predictions
    predictions = trainer.predict(model=model, dataloaders=val_loader)
    pred, gt = predictions[0]
    
    # Calculate correlations
    R, p_val = get_R(pred, gt)
    pred.var["p_val"] = p_val
    pred.var["-log10p_val"] = -np.log10(p_val)

    print('Validation Mean Pearson Correlation:', np.nanmean(R))
    
    return pred, gt, R, p_val


class ClearCacheCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()

# Callback to print memory summary and clear cache after each epoch
class MemoryMonitorCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            print("\n[MemoryMonitor] CUDA memory summary after epoch:")
            print(torch.cuda.memory_summary())
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description='Train and test THItoGene model with configurable parameters')
    
    # General parameters
    parser.add_argument('--mode', type=str, default='train_test', 
                        choices=['train', 'test', 'validate', 'train_test', 'all'],
                        help='Mode to run: train, test, validate, train_test (train then test), or all (train, validate, test)')
    parser.add_argument('--test_sample_id', type=int, default=0,
                        help='Test sample ID for naming')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='HEST1K',
                        choices=['HEST1K', 'HER2ST', 'SKIN'],
                        help='Dataset to use')
    parser.add_argument('--gene_list', type=str, default='HER2ST',
                        choices=list(GENE_LISTS.keys()),
                        help='Gene list to use')
    
    # Model and training parameters
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training and testing')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Hardware parameters
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use (0 for CPU only)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of data loading workers')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=[None, 'ddp', 'ddp_spawn', 'deepspeed'],
                        help='Training strategy for multi-GPU (None, ddp, ddp_spawn, deepspeed)')
    
    # Path parameters
    parser.add_argument('--model_dir', type=str, default='model',
                        help='Directory to save/load models')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Specific checkpoint path to load (overrides automatic path)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Select dataset class
    dataset_classes = {
        'HEST1K': ViT_HEST1K,
        'HER2ST': ViT_HER2ST,
        'SKIN': ViT_SKIN
    }
    vit_dataset = dataset_classes[args.dataset]
    
    print(f"Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Gene List: {args.gene_list}")
    print(f"  GPUs: {args.gpus}")
    print(f"  Num Workers: {args.num_workers}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Model Directory: {args.model_dir}")
    
    results = {}
    
    if args.mode in ['train', 'train_test', 'all']:
        print("\n" + "="*50)
        print("TRAINING PHASE")
        print("="*50)
        
        pred_train, gt_train, R_train, p_val_train = train(
            test_sample_ID=args.test_sample_id,
            vit_dataset=vit_dataset,
            epochs=args.epochs,
            modelsave_address=args.model_dir,
            dataset_name=args.dataset.lower(),
            gene_list=args.gene_list,
            num_workers=args.num_workers,
            gpus=args.gpus,
            strategy=args.strategy,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            patience=args.patience
        )
        
        results['training'] = (pred_train, gt_train, R_train, p_val_train)
        print(f"Training completed. Mean correlation: {np.nanmean(R_train):.4f}")
    
    if args.mode in ['validate', 'all']:
        print("\n" + "="*50)
        print("VALIDATION PHASE")
        print("="*50)
        
        pred_val, gt_val, R_val, p_val_val = validate(
            test_sample_ID=args.test_sample_id,
            vit_dataset=vit_dataset,
            model_address=args.model_dir,
            dataset_name=args.dataset.lower(),
            gene_list=args.gene_list,
            checkpoint_path=args.checkpoint_path,
            num_workers=args.num_workers,
            gpus=args.gpus
        )
        
        results['validation'] = (pred_val, gt_val, R_val, p_val_val)
        print(f"Validation completed. Mean correlation: {np.nanmean(R_val):.4f}")
    
    if args.mode in ['test', 'train_test', 'all']:
        print("\n" + "="*50)
        print("TESTING PHASE")
        print("="*50)
        
        pred_test, gt_test, R_test, p_val_test = test(
            test_sample_ID=args.test_sample_id,
            vit_dataset=vit_dataset,
            model_address=args.model_dir,
            dataset_name=args.dataset.lower(),
            gene_list=args.gene_list,
            checkpoint_path=args.checkpoint_path,
            num_workers=args.num_workers,
            gpus=args.gpus
        )
        
        results['testing'] = (pred_test, gt_test, R_test, p_val_test)
        print(f"Testing completed. Mean correlation: {np.nanmean(R_test):.4f}")
    
    # Print final summary if multiple phases were run
    if len(results) > 1:
        print("\n" + "="*50)
        print("FINAL RESULTS SUMMARY")
        print("="*50)
        
        if 'training' in results:
            print(f"Training Correlation:   {np.nanmean(results['training'][2]):.4f}")
        if 'validation' in results:
            print(f"Validation Correlation: {np.nanmean(results['validation'][2]):.4f}")
        if 'testing' in results:
            print(f"Test Correlation:       {np.nanmean(results['testing'][2]):.4f}")

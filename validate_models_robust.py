#!/usr/bin/env python3
"""
Memory-safe model validation script that processes samples individually
to handle large memory requirements.
"""

import os
import glob
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import gc
import re
from tqdm import tqdm
import anndata as ann

# Set environment variable for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set tensor core precision for better performance
torch.set_float32_matmul_precision('medium')

from dataset import ViT_HEST1K, ViT_HER2ST, ViT_SKIN
from utils import get_R
from vis_model import THItoGene
from config import GENE_LISTS


def clear_gpu_memory():
    """Clear GPU memory cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_gpu_memory_info():
    """Get current GPU memory usage information"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
        allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # GB
        cached_memory = torch.cuda.memory_reserved(device) / (1024**3)  # GB
        free_memory = total_memory - cached_memory
        
        return {
            'total': total_memory,
            'allocated': allocated_memory,
            'cached': cached_memory,
            'free': free_memory
        }
    return None


def create_single_sample_dataset(dataset, sample_idx):
    """Create a dataset with only one sample"""
    class SingleSampleDataset:
        def __init__(self, dataset, idx):
            self.dataset = dataset
            self.idx = idx
        
        def __len__(self):
            return 1
        
        def __getitem__(self, idx):
            if idx != 0:
                raise IndexError("SingleSampleDataset only has one item")
            return self.dataset[self.idx]
    
    return SingleSampleDataset(dataset, sample_idx)


def validate_single_model_memory_safe(checkpoint_path, gene_list, dataset_class, eval_mode='val', device=0, use_cpu_fallback=True):
    """
    Validate a single model checkpoint using memory-safe sample-by-sample processing
    """
    clear_gpu_memory()
    
    # Log initial memory state
    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"  Initial GPU memory - Free: {mem_info['free']:.2f}GB")
    
    try:
        # Get number of genes
        n_genes = GENE_LISTS[gene_list]["n_genes"]
        
        # Load model from checkpoint
        model = THItoGene.load_from_checkpoint(checkpoint_path, n_genes=n_genes)
        print(f"  ✓ Model loaded with {n_genes} genes")
        
        # Create full dataset to get sample count
        full_dataset = dataset_class(mode=eval_mode, gene_list=gene_list)
        num_samples = len(full_dataset)
        print(f"  ✓ Dataset loaded with {num_samples} samples")
        
        # Collect all predictions and ground truth
        all_predictions = []
        all_ground_truth = []
        successful_samples = 0
        failed_samples = 0
        
        # Process each sample individually
        for sample_idx in tqdm(range(num_samples), desc="Processing samples", leave=False):
            try:
                # Create single sample dataset
                single_dataset = create_single_sample_dataset(full_dataset, sample_idx)
                data_loader = DataLoader(single_dataset, batch_size=1, num_workers=8, shuffle=False)
                
                # Setup trainer
                trainer = pl.Trainer(
                    accelerator="gpu", 
                    devices=[device],
                    logger=False, 
                    enable_progress_bar=False
                )
                
                # Run prediction for this sample
                predictions = trainer.predict(model, data_loader)
                pred, gt = predictions[0]
                
                # Extract numpy data from AnnData objects (predict_step returns AnnData objects)
                if hasattr(pred, 'X'):
                    pred_data = pred.X
                else:
                    pred_data = pred
                
                if hasattr(gt, 'X'):
                    gt_data = gt.X
                else:
                    gt_data = gt
                
                # Convert to tensors if they're numpy arrays
                if isinstance(pred_data, np.ndarray):
                    pred_tensor = torch.from_numpy(pred_data)
                else:
                    pred_tensor = pred_data
                    
                if isinstance(gt_data, np.ndarray):
                    gt_tensor = torch.from_numpy(gt_data)
                else:
                    gt_tensor = gt_data
                
                # Ensure tensors are on CPU for collection
                pred_tensor = pred_tensor.cpu()
                gt_tensor = gt_tensor.cpu()
                
                # Collect results
                all_predictions.append(pred_tensor)
                all_ground_truth.append(gt_tensor)
                successful_samples += 1
                
                # Clean up trainer to free memory
                del trainer, predictions, pred, gt, pred_data, gt_data, pred_tensor, gt_tensor
                clear_gpu_memory()
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and use_cpu_fallback:
                    print(f"    Sample {sample_idx}: GPU OOM, trying CPU...")
                    clear_gpu_memory()
                    
                    try:
                        # Retry with CPU
                        single_dataset = create_single_sample_dataset(full_dataset, sample_idx)
                        data_loader = DataLoader(single_dataset, batch_size=1, num_workers=8, shuffle=False)
                        
                        # Create new model instance for CPU
                        cpu_model = THItoGene.load_from_checkpoint(checkpoint_path, n_genes=n_genes)
                        cpu_model = cpu_model.cpu()
                        
                        trainer = pl.Trainer(
                            accelerator="cpu", 
                            logger=False, 
                            enable_progress_bar=False
                        )
                        
                        predictions = trainer.predict(cpu_model, data_loader)
                        pred, gt = predictions[0]
                        
                        # Extract numpy data from AnnData objects (predict_step returns AnnData objects)
                        if hasattr(pred, 'X'):
                            pred_data = pred.X
                        else:
                            pred_data = pred
                        
                        if hasattr(gt, 'X'):
                            gt_data = gt.X
                        else:
                            gt_data = gt
                        
                        # Convert to tensors if they're numpy arrays
                        if isinstance(pred_data, np.ndarray):
                            pred_tensor = torch.from_numpy(pred_data)
                        else:
                            pred_tensor = pred_data
                            
                        if isinstance(gt_data, np.ndarray):
                            gt_tensor = torch.from_numpy(gt_data)
                        else:
                            gt_tensor = gt_data
                        
                        # Ensure tensors are on CPU for collection
                        pred_tensor = pred_tensor.cpu()
                        gt_tensor = gt_tensor.cpu()
                        
                        all_predictions.append(pred_tensor)
                        all_ground_truth.append(gt_tensor)
                        successful_samples += 1
                        
                        del trainer, cpu_model, predictions, pred, gt, pred_data, gt_data, pred_tensor, gt_tensor
                        clear_gpu_memory()
                        
                    except Exception as cpu_e:
                        print(f"    Sample {sample_idx}: Failed on both GPU and CPU - {cpu_e}")
                        failed_samples += 1
                        clear_gpu_memory()
                        continue
                else:
                    print(f"    Sample {sample_idx}: Failed - {e}")
                    failed_samples += 1
                    clear_gpu_memory()
                    continue
            
            except Exception as e:
                print(f"    Sample {sample_idx}: Unexpected error - {e}")
                failed_samples += 1
                clear_gpu_memory()
                continue
        
        print(f"  ✓ Processed {successful_samples}/{num_samples} samples successfully ({failed_samples} failed)")
        
        if successful_samples == 0:
            return {
                'success': False,
                'device_used': 'no_successful_samples',
                'error': f"No samples processed successfully. {failed_samples} samples failed.",
                'mean_correlation': np.nan,
                'median_correlation': np.nan,
                'std_correlation': np.nan,
                'percentile_25': np.nan,
                'percentile_75': np.nan,
                'num_valid_genes': 0,
                'num_total_genes': 0
            }
        
        # Concatenate all predictions and ground truth
        final_pred = torch.cat(all_predictions, dim=0)
        final_gt = torch.cat(all_ground_truth, dim=0)
        
        # Convert tensors to numpy arrays and create AnnData objects for get_R function
        pred_numpy = final_pred.cpu().numpy()
        gt_numpy = final_gt.cpu().numpy()
        
        # Create AnnData objects (get_R expects objects with .X attribute)
        adata_pred = ann.AnnData(pred_numpy)
        adata_gt = ann.AnnData(gt_numpy)
        
        # Calculate metrics
        R, p_val = get_R(adata_pred, adata_gt)
        
        # Calculate statistics
        mean_correlation = np.nanmean(R)
        median_correlation = np.nanmedian(R)
        std_correlation = np.nanstd(R)
        num_valid_genes = np.sum(~np.isnan(R))
        num_total_genes = len(R)
        
        # Calculate percentiles
        percentile_25 = np.nanpercentile(R, 25)
        percentile_75 = np.nanpercentile(R, 75)
        
        # Clean up
        del model, full_dataset, all_predictions, all_ground_truth, final_pred, final_gt
        clear_gpu_memory()
        
        device_used = 'gpu' if failed_samples == 0 else 'mixed_gpu_cpu'
        
        return {
            'success': True,
            'device_used': device_used,
            'successful_samples': successful_samples,
            'failed_samples': failed_samples,
            'mean_correlation': mean_correlation,
            'median_correlation': median_correlation,
            'std_correlation': std_correlation,
            'percentile_25': percentile_25,
            'percentile_75': percentile_75,
            'num_valid_genes': num_valid_genes,
            'num_total_genes': num_total_genes,
            'correlations': R,
            'p_values': p_val
        }
        
    except Exception as e:
        clear_gpu_memory()
        return {
            'success': False,
            'device_used': 'model_loading_failed',
            'error': str(e),
            'mean_correlation': np.nan,
            'median_correlation': np.nan,
            'std_correlation': np.nan,
            'percentile_25': np.nan,
            'percentile_75': np.nan,
            'num_valid_genes': 0,
            'num_total_genes': 0
        }


def extract_model_info(checkpoint_path):
    """Extract model information from checkpoint filename"""
    filename = os.path.basename(checkpoint_path)
    
    # Extract model number if present
    match = re.search(r'THItoGene_her2st_(\d+)\.ckpt', filename)
    if match:
        model_number = int(match.group(1))
    else:
        model_number = None
    
    return {
        'filename': filename,
        'model_number': model_number,
        'checkpoint_path': checkpoint_path
    }


def validate_all_models_memory_safe(
    model_directory,
    gene_list,
    dataset_class,
    eval_mode='val',
    output_file=None,
    device=0,
    model_pattern="THItoGene_her2st_*.ckpt",
    use_cpu_fallback=True,
    max_models=None
):
    """
    Validate all models using memory-safe approach
    """
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(model_directory, model_pattern)
    checkpoint_files = glob.glob(checkpoint_pattern)
    checkpoint_files.sort()
    
    if max_models:
        checkpoint_files = checkpoint_files[:max_models]
    
    print(f"Found {len(checkpoint_files)} checkpoint files to validate")
    
    if len(checkpoint_files) == 0:
        print(f"No checkpoint files found in {model_directory}")
        return pd.DataFrame()
    
    # Initialize results list
    results = []
    
    # Validate each model
    for checkpoint_path in tqdm(checkpoint_files, desc=f"Validating models on {eval_mode} set"):
        print(f"\nValidating: {os.path.basename(checkpoint_path)}")
        
        # Extract model info
        model_info = extract_model_info(checkpoint_path)
        
        # Show memory status before validation
        mem_info = get_gpu_memory_info()
        if mem_info:
            print(f"  Pre-validation GPU memory - Free: {mem_info['free']:.2f}GB")
        
        # Validate model
        validation_results = validate_single_model_memory_safe(
            checkpoint_path, gene_list, dataset_class, eval_mode, device, use_cpu_fallback
        )
        
        # Combine results
        result_row = {
            **model_info,
            **validation_results,
            'eval_mode': eval_mode,
            'gene_list': gene_list
        }
        
        results.append(result_row)
        
        # Print immediate results
        if validation_results['success']:
            device_used = validation_results.get('device_used', 'unknown')
            successful_samples = validation_results.get('successful_samples', 'N/A')
            failed_samples = validation_results.get('failed_samples', 'N/A')
            print(f"  Mean Correlation: {validation_results['mean_correlation']:.4f} (on {device_used})")
            print(f"  Median Correlation: {validation_results['median_correlation']:.4f}")
            print(f"  Valid Genes: {validation_results['num_valid_genes']}/{validation_results['num_total_genes']}")
            print(f"  Samples: {successful_samples} successful, {failed_samples} failed")
        else:
            device_used = validation_results.get('device_used', 'unknown')
            print(f"  FAILED ({device_used}): {validation_results.get('error', 'Unknown error')}")
        
        # Show memory status after validation
        mem_info = get_gpu_memory_info()
        if mem_info:
            print(f"  Post-validation GPU memory - Free: {mem_info['free']:.2f}GB")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by mean correlation (descending)
    if not df_results.empty and 'mean_correlation' in df_results.columns:
        df_results = df_results.sort_values('mean_correlation', ascending=False)
        df_results['rank'] = range(1, len(df_results) + 1)
    
    # Save results if output file specified
    if output_file:
        # Remove complex columns before saving
        df_to_save = df_results.drop(columns=['correlations', 'p_values'], errors='ignore')
        df_to_save.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    return df_results


def print_summary(results_df, top_n=5):
    """Print a summary of the validation results"""
    
    if results_df.empty:
        print("No results to summarize.")
        return
    
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    # Overall statistics
    successful_models = results_df[results_df['success'] == True]
    failed_models = results_df[results_df['success'] == False]
    
    print(f"Total models evaluated: {len(results_df)}")
    print(f"Successful evaluations: {len(successful_models)}")
    print(f"Failed evaluations: {len(failed_models)}")
    
    if len(successful_models) > 0:
        print(f"\nOverall Statistics:")
        print(f"  Best Mean Correlation: {successful_models['mean_correlation'].max():.4f}")
        print(f"  Worst Mean Correlation: {successful_models['mean_correlation'].min():.4f}")
        print(f"  Average Mean Correlation: {successful_models['mean_correlation'].mean():.4f}")
        print(f"  Std Dev of Correlations: {successful_models['mean_correlation'].std():.4f}")
        
        print(f"\nTop {top_n} Performing Models:")
        print("-" * 80)
        for i, (_, row) in enumerate(successful_models.head(top_n).iterrows()):
            device_info = row.get('device_used', 'unknown')
            print(f"{i+1:2d}. {row['filename']:30s} | "
                  f"Mean: {row['mean_correlation']:.4f} | "
                  f"Device: {device_info}")
        
        # Find best model
        best_model = successful_models.iloc[0]
        print(f"\nBEST MODEL:")
        print(f"  File: {best_model['filename']}")
        print(f"  Model Number: {best_model['model_number']}")
        print(f"  Mean Correlation: {best_model['mean_correlation']:.4f}")
        print(f"  Median Correlation: {best_model['median_correlation']:.4f}")
        print(f"  Standard Deviation: {best_model['std_correlation']:.4f}")
        print(f"  25th Percentile: {best_model['percentile_25']:.4f}")
        print(f"  75th Percentile: {best_model['percentile_75']:.4f}")
        print(f"  Valid Genes: {best_model['num_valid_genes']}/{best_model['num_total_genes']}")
        if 'successful_samples' in best_model:
            print(f"  Successful Samples: {best_model['successful_samples']}")
            print(f"  Failed Samples: {best_model['failed_samples']}")
    
    if len(failed_models) > 0:
        print(f"\nFailed Models:")
        for _, row in failed_models.iterrows():
            print(f"  {row['filename']}: {row.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(description='Memory-safe validation of THItoGene model checkpoints')
    parser.add_argument('--model_dir', type=str, default='./model',
                        help='Directory containing model checkpoints')
    parser.add_argument('--gene_list', type=str, default='HER2ST', 
                        choices=list(GENE_LISTS.keys()),
                        help='Gene list used for training')
    parser.add_argument('--dataset', type=str, default='HEST1K',
                        choices=['HEST1K', 'HER2ST', 'SKIN'],
                        help='Dataset to use for validation')
    parser.add_argument('--eval_mode', type=str, default='val',
                        choices=['val', 'test'],
                        help='Evaluation mode: val for validation set, test for test set')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output CSV file to save results')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device to use')
    parser.add_argument('--model_pattern', type=str, default="THItoGene_her2st_*.ckpt",
                        help='Pattern to match checkpoint files')
    parser.add_argument('--top_n', type=int, default=5,
                        help='Number of top models to display in summary')
    parser.add_argument('--no_cpu_fallback', action='store_true',
                        help='Disable CPU fallback when GPU runs out of memory')
    parser.add_argument('--max_models', type=int, default=None,
                        help='Maximum number of models to validate (for testing)')
    
    args = parser.parse_args()
    
    # Select dataset class
    dataset_classes = {
        'HEST1K': ViT_HEST1K,
        'HER2ST': ViT_HER2ST,
        'SKIN': ViT_SKIN
    }
    dataset_class = dataset_classes[args.dataset]
    
    # Generate output filename if not provided
    if args.output_file is None:
        suffix = f"_max{args.max_models}" if args.max_models else ""
        args.output_file = f"validation_results_{args.gene_list}_{args.dataset}_{args.eval_mode}_memory_safe{suffix}.csv"
    
    print(f"Model Directory: {args.model_dir}")
    print(f"Gene List: {args.gene_list}")
    print(f"Dataset: {args.dataset}")
    print(f"Evaluation Mode: {args.eval_mode}")
    print(f"Model Pattern: {args.model_pattern}")
    print(f"Output File: {args.output_file}")
    print(f"GPU Device: {args.device}")
    print(f"CPU Fallback: {not args.no_cpu_fallback}")
    if args.max_models:
        print(f"Max Models: {args.max_models}")
    
    # Show initial GPU memory status
    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"Initial GPU Memory - Total: {mem_info['total']:.2f}GB, Free: {mem_info['free']:.2f}GB")
    
    # Validate all models
    results_df = validate_all_models_memory_safe(
        model_directory=args.model_dir,
        gene_list=args.gene_list,
        dataset_class=dataset_class,
        eval_mode=args.eval_mode,
        output_file=args.output_file,
        device=args.device,
        model_pattern=args.model_pattern,
        use_cpu_fallback=not args.no_cpu_fallback,
        max_models=args.max_models
    )
    
    # Print summary
    print_summary(results_df, top_n=args.top_n)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to validate multiple saved model checkpoints and find the best performing one.

This script will:
1. Load each checkpoint
2. Evaluate on validation and/or test sets
3. Calculate performance metrics (Pearson correlation)
4. Rank models by performance
5. Save results to CSV for analysis

Memory Management:
- Automatically clears GPU memory between model evaluations
- Falls back to CPU if GPU runs out of memory
- Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation
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

# Set environment variable for better memory management if not already set
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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


def validate_single_model(checkpoint_path, gene_list, dataset_class, eval_mode='val', device=0, use_cpu_fallback=True):
    """
    Validate a single model checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        gene_list: Gene list used for training
        dataset_class: Dataset class to use
        eval_mode: 'val' for validation set, 'test' for test set
        device: GPU device to use
        use_cpu_fallback: Whether to fallback to CPU if GPU runs out of memory
        
    Returns:
        Dictionary with validation results
    """
    # Clear GPU memory before starting
    clear_gpu_memory()
    
    # Log initial memory state
    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"  Initial GPU memory - Free: {mem_info['free']:.2f}GB, Allocated: {mem_info['allocated']:.2f}GB")
    
    try:
        # Get number of genes
        n_genes = GENE_LISTS[gene_list]["n_genes"]
        
        # Load model from checkpoint
        model = THItoGene.load_from_checkpoint(checkpoint_path, n_genes=n_genes)
        
        # Create dataset
        dataset = dataset_class(mode=eval_mode, gene_list=gene_list)
        data_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)  # Reduced num_workers
        
        # Setup trainer with GPU
        trainer = pl.Trainer(accelerator="gpu", devices=[device], logger=False, enable_progress_bar=False)
        
        # Run predictions
        predictions = trainer.predict(model, data_loader)
        pred, gt = predictions[0]
        
        # Calculate metrics
        R, p_val = get_R(pred, gt)
        
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
        del model, trainer, predictions, pred, gt
        clear_gpu_memory()
        
        return {
            'success': True,
            'device_used': 'gpu',
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
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) and use_cpu_fallback:
            print(f"  GPU out of memory, trying CPU fallback...")
            clear_gpu_memory()
            
            try:
                # Retry with CPU
                model = THItoGene.load_from_checkpoint(checkpoint_path, n_genes=n_genes)
                dataset = dataset_class(mode=eval_mode, gene_list=gene_list)
                data_loader = DataLoader(dataset, batch_size=1, num_workers=16, shuffle=False)  # Minimal workers for CPU
                
                # Setup trainer with CPU
                trainer = pl.Trainer(accelerator="gpu", devices=2, strategy = "ddp", logger=False, enable_progress_bar=False)
                
                # Run predictions
                predictions = trainer.predict(model, data_loader)
                pred, gt = predictions[0]
                
                # Calculate metrics
                R, p_val = get_R(pred, gt)
                
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
                del model, trainer, predictions, pred, gt
                clear_gpu_memory()
                
                print(f"  Successfully completed on CPU")
                
                return {
                    'success': True,
                    'device_used': 'cpu',
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
                
            except Exception as cpu_e:
                clear_gpu_memory()
                return {
                    'success': False,
                    'device_used': 'failed_cpu_fallback',
                    'error': f"GPU OOM, CPU fallback also failed: {str(cpu_e)}",
                    'original_gpu_error': str(e),
                    'mean_correlation': np.nan,
                    'median_correlation': np.nan,
                    'std_correlation': np.nan,
                    'percentile_25': np.nan,
                    'percentile_75': np.nan,
                    'num_valid_genes': 0,
                    'num_total_genes': 0
                }
        else:
            clear_gpu_memory()
            return {
                'success': False,
                'device_used': 'gpu_failed',
                'error': str(e),
                'mean_correlation': np.nan,
                'median_correlation': np.nan,
                'std_correlation': np.nan,
                'percentile_25': np.nan,
                'percentile_75': np.nan,
                'num_valid_genes': 0,
                'num_total_genes': 0
            }
            
    except Exception as e:
        clear_gpu_memory()
        return {
            'success': False,
            'device_used': 'unknown_error',
            'error': str(e),
            'mean_correlation': np.nan,
            'median_correlation': np.nan,
            'std_correlation': np.nan,
            'percentile_25': np.nan,
            'percentile_75': np.nan,
            'num_valid_genes': 0,
            'num_total_genes': 0
        }


def validate_all_models(
    model_directory,
    gene_list,
    dataset_class,
    eval_mode='val',
    output_file=None,
    device=0,
    model_pattern="THItoGene_her2st_*.ckpt",
    use_cpu_fallback=True
):
    """
    Validate all models in a directory
    
    Args:
        model_directory: Directory containing model checkpoints
        gene_list: Gene list used for training
        dataset_class: Dataset class to use
        eval_mode: 'val' for validation set, 'test' for test set
        output_file: Path to save results CSV (optional)
        device: GPU device to use
        model_pattern: Pattern to match checkpoint files
        use_cpu_fallback: Whether to fallback to CPU if GPU runs out of memory
        
    Returns:
        DataFrame with results for all models
    """
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(model_directory, model_pattern)
    checkpoint_files = glob.glob(checkpoint_pattern)
    checkpoint_files.sort()
    
    print(f"Found {len(checkpoint_files)} checkpoint files matching pattern: {model_pattern}")
    
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
        validation_results = validate_single_model(
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
            print(f"  Mean Correlation: {validation_results['mean_correlation']:.4f} (on {device_used})")
            print(f"  Median Correlation: {validation_results['median_correlation']:.4f}")
            print(f"  Valid Genes: {validation_results['num_valid_genes']}/{validation_results['num_total_genes']}")
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
        # Remove complex columns before saving (only correlations and p_values remain now)
        df_to_save = df_results.drop(columns=['correlations', 'p_values'], errors='ignore')
        df_to_save.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    return df_results


def print_summary(results_df, top_n=5):
    """Print a summary of the validation results"""
    
    if results_df.empty:
        print("No results to summarize.")
        return
    
    print(f"\\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    # Overall statistics
    successful_models = results_df[results_df['success'] == True]
    failed_models = results_df[results_df['success'] == False]
    
    print(f"Total models evaluated: {len(results_df)}")
    print(f"Successful evaluations: {len(successful_models)}")
    print(f"Failed evaluations: {len(failed_models)}")
    
    if len(successful_models) > 0:
        print(f"\\nOverall Statistics:")
        print(f"  Best Mean Correlation: {successful_models['mean_correlation'].max():.4f}")
        print(f"  Worst Mean Correlation: {successful_models['mean_correlation'].min():.4f}")
        print(f"  Average Mean Correlation: {successful_models['mean_correlation'].mean():.4f}")
        print(f"  Std Dev of Correlations: {successful_models['mean_correlation'].std():.4f}")
        
        print(f"\\nTop {top_n} Performing Models:")
        print("-" * 80)
        for i, (_, row) in enumerate(successful_models.head(top_n).iterrows()):
            print(f"{i+1:2d}. {row['filename']:30s} | "
                  f"Mean: {row['mean_correlation']:.4f} | "
                  f"Median: {row['median_correlation']:.4f} | "
                  f"Std: {row['std_correlation']:.4f}")
        
        # Find best model
        best_model = successful_models.iloc[0]
        print(f"\\nBEST MODEL:")
        print(f"  File: {best_model['filename']}")
        print(f"  Model Number: {best_model['model_number']}")
        print(f"  Mean Correlation: {best_model['mean_correlation']:.4f}")
        print(f"  Median Correlation: {best_model['median_correlation']:.4f}")
        print(f"  Standard Deviation: {best_model['std_correlation']:.4f}")
        print(f"  25th Percentile: {best_model['percentile_25']:.4f}")
        print(f"  75th Percentile: {best_model['percentile_75']:.4f}")
        print(f"  Valid Genes: {best_model['num_valid_genes']}/{best_model['num_total_genes']}")
    
    if len(failed_models) > 0:
        print(f"\\nFailed Models:")
        for _, row in failed_models.iterrows():
            print(f"  {row['filename']}: {row.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(description='Validate multiple THItoGene model checkpoints')
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
        args.output_file = f"validation_results_{args.gene_list}_{args.dataset}_{args.eval_mode}.csv"
    
    print(f"Model Directory: {args.model_dir}")
    print(f"Gene List: {args.gene_list}")
    print(f"Dataset: {args.dataset}")
    print(f"Evaluation Mode: {args.eval_mode}")
    print(f"Model Pattern: {args.model_pattern}")
    print(f"Output File: {args.output_file}")
    print(f"GPU Device: {args.device}")
    print(f"CPU Fallback: {not args.no_cpu_fallback}")
    
    # Show initial GPU memory status
    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"Initial GPU Memory - Total: {mem_info['total']:.2f}GB, Free: {mem_info['free']:.2f}GB")
    
    # Validate all models
    results_df = validate_all_models(
        model_directory=args.model_dir,
        gene_list=args.gene_list,
        dataset_class=dataset_class,
        eval_mode=args.eval_mode,
        output_file=args.output_file,
        device=args.device,
        model_pattern=args.model_pattern,
        use_cpu_fallback=not args.no_cpu_fallback
    )
    
    # Print summary
    print_summary(results_df, top_n=args.top_n)


if __name__ == "__main__":
    main()

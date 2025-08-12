"""
Simple test to validate a single model
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Set environment variable for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from dataset import ViT_HEST1K
from config import GENE_LISTS
from vis_model import THItoGene
from utils import get_R

def test_model(checkpoint_path="./model/THItoGene_her2st_6.ckpt"):
    """Test validation of a single model"""
    
    print("Testing single model validation...")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
  
    try:
        # Load model
        n_genes = GENE_LISTS["HER2ST"]["n_genes"]
        print(f"Loading model with {n_genes} genes...")
        model = THItoGene.load_from_checkpoint(checkpoint_path, n_genes=n_genes)
        print("✓ Model loaded")
        
        # Create dataset
        print("Creating dataset...")
        dataset = ViT_HEST1K(mode='test', gene_list='HER2ST', cancer_only=True)
        data_loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Get sample IDs directly from dataset
        sample_ids = dataset.sample_ids.copy()  # Get sample IDs from dataset
        print(f"✓ Got {len(sample_ids)} sample IDs from dataset")
        
        # Setup trainer
        print("Setting up trainer...")
        trainer = pl.Trainer(
            accelerator="gpu", 
            devices=[0], 
            logger=None, 
            enable_progress_bar=True
        )
        print("✓ Trainer created")
        
        # Run predictions
        print("Running predictions...")
        predictions = trainer.predict(model, data_loader)
        print(f"✓ Predictions completed, got {len(predictions)} batches")
        
        # Process predictions (they're just tensors, no metadata)
        all_preds = []
        for batch in predictions:
            all_preds.extend(batch)
        all_preds = torch.cat(all_preds, dim=0)
        print(f"✓ Processed predictions, total shape: {all_preds.shape}")
        
        # Verify sample count matches
        if len(sample_ids) != all_preds.shape[0]:
            print(f"Warning: Sample count mismatch - {len(sample_ids)} sample IDs vs {all_preds.shape[0]} predictions")
        
        # Save predictions with sample IDs
        output_data = {
            'sample_ids': sample_ids,
            'predictions': all_preds.cpu().numpy()
        }

        # Create output directory if it doesn't exist
        output_dir = "./results"
        os.makedirs(output_dir, exist_ok=True)

        # Save as numpy file
        output_path = os.path.join(output_dir, "predictions_with_sample_ids.npz")
        np.savez(output_path, **output_data)
        print(f"✓ Predictions saved to {output_path}")
        
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'trainer' in locals():
            del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
if __name__ == "__main__":
    test_model()
    print("Test completed successfully.")

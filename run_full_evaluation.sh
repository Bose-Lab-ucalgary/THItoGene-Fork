#!/bin/bash
"""
Batch script to validate and test HER2ST models
"""

echo "=========================================="
echo "HER2ST Model Validation and Testing Suite"
echo "=========================================="

# Step 1: Validate all models on validation set
echo "Step 1: Validating all 32 models on validation set..."
python validate_her2st_models.py

echo ""
echo "Validation completed. Check 'her2st_validation_results.csv' for detailed results."

# Step 2: Find the best model and test it
echo ""
echo "Step 2: Finding best model and testing on test set..."

# Extract best model filename from CSV (assuming it's the first row after header)
BEST_MODEL=$(tail -n +2 her2st_validation_results.csv | head -n 1 | cut -d',' -f2)

if [ ! -z "$BEST_MODEL" ]; then
    echo "Best model identified: $BEST_MODEL"
    echo "Testing on test set..."
    
    python validate_models.py \
        --model_dir ./model \
        --gene_list HER2ST \
        --dataset HEST1K \
        --eval_mode test \
        --model_pattern "$BEST_MODEL" \
        --output_file "best_model_test_results.csv"
    
    echo ""
    echo "Test results saved to: best_model_test_results.csv"
else
    echo "Could not identify best model from validation results."
fi

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "Files generated:"
echo "  - her2st_validation_results.csv (all models on validation set)"
echo "  - best_model_test_results.csv (best model on test set)"
echo "=========================================="

# test_data_loading.py
import torch
from train import get_data
import argparse

# Create test args
class TestArgs:
    def __init__(self):
        self.data_dir = "../data/imagenet_structured"  # or your converted path
        self.dataset = "imagenet"
        self.batch_size = 4
        self.workers = 2
        self.distributed = False

args = TestArgs()

try:
    train_loader, val_loader, num_classes = get_data(args)
    print(f"✅ Data loading successful!")
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test one batch
    for imgs, lbls in train_loader:
        print(f"Batch shape: {imgs.shape}, Labels shape: {lbls.shape}")
        break
        
except Exception as e:
    print(f"❌ Data loading failed: {e}")
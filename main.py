import argparse
import sys
import os
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# Training Function
def train():
    """start training the model"""
    print("starting training...")
    
    # Import 
    from models.train import train_model as train_main
    train_main()
    


# Data Builder Function
def build_data():
    """Build train, validation, and test datasets"""
    print("🔧 Building datasets...")
    
    # Import and run dataset split
    from utils.dataset_split import main as dataset_split_main
    dataset_split_main()
    
    print("✅ Datasets created successfully!")


# Main Function
def main():
    parser = argparse.ArgumentParser(description="Urdu Masra Generator")
    parser.add_argument("--mode", choices=["train","data-builder"], 
                       required=True, help="Mode to run")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    elif args.mode == "data-builder":
        build_data()

if __name__ == "__main__":
    main()
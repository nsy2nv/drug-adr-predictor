import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
import torch
from config import config

class ADRDataset(Dataset):
    """Dataset for ADR prediction"""
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'smiles': item['smiles'],
            'adr': item['adr'], 
            'label': item['label']
        }

class DataLoader:
    """Data loading and preprocessing"""
    
    @staticmethod
    def load_and_split_data():
        """Load data and create fixed partitions"""
        df = pd.read_csv(config.DATA_PATH)
        
        print(f"Loaded dataset with {len(df)} samples")
        print(f"Columns: {df.columns.tolist()}")
        
        # Clean data
        df = df.dropna(subset=['smiles', 'adr', 'label'])
        df['smiles'] = df['smiles'].astype(str)
        df['adr'] = df['adr'].astype(str)
        df['label'] = df['label'].astype(int)
        
        print(f"After cleaning: {len(df)} samples")
        print(f"Class distribution: {df['label'].value_counts().to_dict()}")
        
        # Convert to list of dictionaries
        data = df.to_dict('records')
        
        # Fixed partition: First split external validation
        train_val_data, external_val_data = train_test_split(
            data,
            test_size=config.EXTERNAL_VAL_RATIO,
            random_state=config.RANDOM_STATE,
            stratify=[d['label'] for d in data]
        )
        
        # Then split remaining into train/test
        train_data, test_data = train_test_split(
            train_val_data,
            test_size=config.TEST_RATIO,
            random_state=config.RANDOM_STATE,
            stratify=[d['label'] for d in train_val_data]
        )
        
        print(f"Data splits - Total: {len(data)}, "
              f"Train: {len(train_data)}, "
              f"Test: {len(test_data)}, "
              f"External Val: {len(external_val_data)}")
        
        return train_data, test_data, external_val_data
    
    @staticmethod
    def create_cv_splits(train_data, n_splits=5, random_state=42):
        """Create cross-validation splits"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        labels = [d['label'] for d in train_data]
        
        cv_splits = []
        for train_idx, val_idx in skf.split(train_data, labels):
            train_fold = [train_data[i] for i in train_idx]
            val_fold = [train_data[i] for i in val_idx]
            cv_splits.append((train_fold, val_fold))
        
        return cv_splits
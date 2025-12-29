import torch
import warnings
from transformers import logging

# Ignore warnings
logging.set_verbosity_error()
# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress specific Hugging Face download warnings
warnings.filterwarnings("ignore", message=".*resume_download.*")

class Config:
    """Configuration parameters for the experiment"""
    
    # Data parameters
    DATA_PATH = "C://Users//Nsikan//Documents//PYTHONFILES//combine_data_lstm_gnn.csv"
    EXTERNAL_VAL_RATIO = 0.20
    TEST_RATIO = 0.10
    TRAIN_RATIO = 0.70
    
    # Model parameters
    CHEM_MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
    SAP_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    GAT_HIDDEN_DIM = 128
    GAT_HEADS = 4
    GAT_OUTPUT_DIM = 256
    CLASSIFIER_HIDDEN_DIM = 512
    DROPOUT_RATE = 0.3
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    EPOCHS = 20 # 2, 5
    PATIENCE = 10
    
    # Cross-validation
    CV_FOLDS = 5
    CV_SEEDS = [42, 456, 999] # 123, 456, 789,
    CV_REPEATS = 1
    
    # Hyperparameter search
    HYPERPARAM_GRID = {
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'batch_size': [16, 32],
        'gat_hidden_dim': [64, 128],
        'gat_heads': [4, 8],
        'dropout_rate': [0.2, 0.3, 0.4]
    }
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random state
    RANDOM_STATE = 42

# Create a config instance for easy import
config = Config()
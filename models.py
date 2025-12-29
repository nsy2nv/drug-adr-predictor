import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from transformers import AutoModel
from config import config
from transformers import logging

# Set transformers to only show errors
logging.set_verbosity_error()

class HybridADRGNN(torch.nn.Module):
    """Hybrid ChemBERTa + SapBERT + GNN model"""
    
    def __init__(self, model_config=None):
        super().__init__()
        if model_config is None:
            model_config = config
        
        # Frozen language models
        self.sapbert = AutoModel.from_pretrained(model_config.SAP_MODEL_NAME)
        
        # Freeze language models
        for param in self.sapbert.parameters():
            param.requires_grad = False
        
        # GNN layers - updated for simpler atom features (5 dimensions)
        self.gat1 = GATv2Conv(
            in_channels=5,  # Simple atom features dimension
            out_channels=model_config.GAT_HIDDEN_DIM,
            heads=model_config.GAT_HEADS,
            concat=True
        )
        self.gat2 = GATv2Conv(
            in_channels=model_config.GAT_HIDDEN_DIM * model_config.GAT_HEADS,
            out_channels=model_config.GAT_OUTPUT_DIM,
            heads=1,
            concat=False
        )
        
        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(model_config.GAT_OUTPUT_DIM + 768, model_config.CLASSIFIER_HIDDEN_DIM),  # GNN + SapBERT
            torch.nn.ReLU(),
            torch.nn.Dropout(model_config.DROPOUT_RATE),
            torch.nn.Linear(model_config.CLASSIFIER_HIDDEN_DIM, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, graphs, adr_texts):
        # ADR embeddings from SapBERT
        adr_embeds = self.sapbert(**adr_texts).last_hidden_state[:, 0, :]
        
        # GNN forward pass
        x = F.leaky_relu(self.gat1(graphs.x, graphs.edge_index))
        x = F.leaky_relu(self.gat2(x, graphs.edge_index))
        x = global_mean_pool(x, graphs.batch)
        
        # Combine features
        combined = torch.cat([x, adr_embeds], dim=1)
        return self.classifier(combined).squeeze()
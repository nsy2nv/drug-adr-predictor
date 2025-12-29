import torch
from torch_geometric.data import Data, Batch
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer
from config import config
import numpy as np
from transformers import logging

# Set transformers to only show errors
logging.set_verbosity_error()

class GraphProcessor:
    """Handles molecular graph conversion and processing"""
    
    def __init__(self):
        self.chemberta_tokenizer = AutoTokenizer.from_pretrained(config.CHEM_MODEL_NAME)
        self.chemberta_model = AutoModel.from_pretrained(config.CHEM_MODEL_NAME)
        self.chemberta_model.eval()
        
        # Freeze ChemBERTa
        for param in self.chemberta_model.parameters():
            param.requires_grad = False
    
    def smiles_to_graph(self, smiles):
        """Convert SMILES to molecular graph with ChemBERTa features"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # print(f"Failed to parse SMILES: {smiles}")
                return None
            
            # For large dataset, use simpler features for speed
            atom_features = []
            
            # Use simpler atom features for large dataset
            for atom in mol.GetAtoms():
                # Basic atom features instead of ChemBERTa for speed
                features = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    atom.GetHybridization().real,
                    atom.GetIsAromatic()
                ]
                atom_features.append(features)
            
            if not atom_features:
                return None
                
            # Create edge indices
            edge_index = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_index.append([i, j])
                edge_index.append([j, i])  # Undirected graph
            
            if not edge_index:
                # Handle single-atom molecules
                if len(atom_features) == 1:
                    edge_index = [[0, 0]]
                else:
                    return None
            
            # Convert to tensors
            x = torch.tensor(atom_features, dtype=torch.float)
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            # Normalize features
            x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
            
            return Data(x=x, edge_index=edge_index)
            
        except Exception as e:
            # print(f"Error converting SMILES to graph: {e}")
            return None

def collate_fn(batch, graph_processor, sapbert_tokenizer):
    """Batch processing for graphs + text"""
    try:
        graphs = []
        valid_indices = []
        valid_adrs = []
        valid_labels = []
        
        for i, item in enumerate(batch):
            graph = graph_processor.smiles_to_graph(item['smiles'])
            if graph is not None:
                graphs.append(graph)
                valid_indices.append(i)
                valid_adrs.append(item['adr'])
                valid_labels.append(item['label'])
        
        if not graphs:
            return None, None, None
        
        # Tokenize ADR texts for valid samples
        adr_texts = sapbert_tokenizer(
            valid_adrs,
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors="pt"
        )
        
        labels = torch.tensor(valid_labels, dtype=torch.float)
        
        return Batch.from_data_list(graphs), adr_texts, labels
        
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        return None, None, None
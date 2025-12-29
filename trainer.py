import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from config import config

class Trainer:
    """Model training and evaluation"""
    
    def __init__(self, model, device=None):
        if device is None:
            device = config.DEVICE
        self.model = model.to(device)
        self.device = device
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        batches_processed = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            if batch[0] is None:  # Skip invalid batches
                continue
                
            graphs, adr_texts, labels = batch
            graphs = graphs.to(self.device)
            adr_texts = {k: v.to(self.device) for k, v in adr_texts.items()}
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(graphs, adr_texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches_processed += 1
        
        if batches_processed == 0:
            return 0.0
            
        return epoch_loss / batches_processed
    
    def evaluate(self, data_loader, criterion=None):
        """Evaluate model on given data loader"""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        batches_processed = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if batch[0] is None:
                    continue
                    
                graphs, adr_texts, labels = batch
                graphs = graphs.to(self.device)
                adr_texts = {k: v.to(self.device) for k, v in adr_texts.items()}
                labels = labels.to(self.device)
                
                outputs = self.model(graphs, adr_texts)
                
                if criterion:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    batches_processed += 1
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = {}
        if criterion and batches_processed > 0:
            metrics['loss'] = total_loss / batches_processed
        
        return np.array(all_preds), np.array(all_labels), metrics
    
    def train(self, train_loader, val_loader, epochs=config.EPOCHS, lr=1e-4, patience=config.PATIENCE):
        """Complete training procedure with early stopping"""
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr
        )
        criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            history['train_loss'].append(train_loss)
            
            # Validation
            _, _, val_metrics = self.evaluate(val_loader, criterion)
            val_loss = val_metrics.get('loss', float('inf'))
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}: " #epochs
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        try:
            self.model.load_state_dict(torch.load('best_model.pth'))
        except:
            print("Warning: Could not load best model, using final model")
        
        return history
# prediction_module.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import warnings
from typing import List, Dict, Optional, Tuple
import os

# Import your existing modules
from config import config
from data_loader import ADRDataset, DataLoader as DataLoaderClass
from models import HybridADRGNN
from graph_processor import GraphProcessor, collate_fn

warnings.filterwarnings("ignore")

class Predictor:
    """Simple predictor for your 10% test data"""
    
    def __init__(self, model_path: str = 'best_model.pth'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to your trained model (default: 'best_model.pth')
        """
        self.device = config.DEVICE
        
        # Initialize processors
        self.graph_processor = GraphProcessor()
        self.sapbert_tokenizer = AutoTokenizer.from_pretrained(config.SAP_MODEL_NAME)
        
        # Initialize model
        self.model = HybridADRGNN(config).to(self.device)
        
        # Load model with error handling
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model weights with error handling"""
        if not os.path.exists(model_path):
            print(f"⚠ Warning: Model file '{model_path}' not found.")
            print("   Using random weights (untrained model).")
            print("   To train a model, run: python main.py")
            return
        
        try:
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Remove unexpected keys (like position_ids from transformers)
            model_state_dict = self.model.state_dict()
            
            # Filter state dict to match model's keys
            filtered_state_dict = {}
            for key, value in state_dict.items():
                # Skip unexpected keys
                if key in model_state_dict:
                    filtered_state_dict[key] = value
                else:
                    print(f"   Skipping unexpected key: {key}")
            
            # Load filtered state dict
            if filtered_state_dict:
                self.model.load_state_dict(filtered_state_dict, strict=False)
                self.model.eval()
                print(f"✓ Loaded model from {model_path}")
                print(f"   Loaded {len(filtered_state_dict)}/{len(state_dict)} parameters")
            else:
                print("⚠ Could not load any parameters from the model file")
                
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            print("   Using random weights instead.")
    
    def make_predictions(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Main function to make predictions on the 10% test data
        
        Returns:
            Tuple of (all_predictions_df, correct_predictions_df, wrong_predictions_df)
        """
        print("=" * 60)
        print("MAKING PREDICTIONS ON 10% TEST DATA")
        print("=" * 60)
        
        # Initialize empty dataframes
        all_predictions_df = pd.DataFrame()
        correct_predictions_df = pd.DataFrame()
        wrong_predictions_df = pd.DataFrame()
        
        # 1. Load your data
        print("\nStep 1: Loading data...")
        try:
            data_loader = DataLoaderClass()
            train_data, test_data, external_val_data = data_loader.load_and_split_data()
            
            print(f"   Train samples: {len(train_data)}")
            print(f"   Test samples (10% reserved): {len(test_data)}")
            print(f"   External validation: {len(external_val_data)}")
            
            if len(test_data) == 0:
                print("❌ Error: No test data found!")
                return all_predictions_df, correct_predictions_df, wrong_predictions_df
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return all_predictions_df, correct_predictions_df, wrong_predictions_df
        
        # 2. Create test dataset
        print("\nStep 2: Preparing test data...")
        try:
            dataset = ADRDataset(test_data)
            
            test_loader = DataLoader(
                dataset,
                batch_size=min(config.BATCH_SIZE, 8),  # Smaller batch for safety
                shuffle=False,
                collate_fn=lambda batch: collate_fn(batch, self.graph_processor, self.sapbert_tokenizer),
                num_workers=0
            )
        except Exception as e:
            print(f"❌ Error preparing data: {e}")
            return all_predictions_df, correct_predictions_df, wrong_predictions_df
        
        # 3. Make predictions
        print("\nStep 3: Making predictions...")
        all_predictions = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch[0] is None:  # Skip invalid
                    print(f"   Skipping batch {batch_idx+1} (invalid)")
                    continue
                    
                try:
                    graphs, adr_texts, labels = batch
                    graphs = graphs.to(self.device)
                    adr_texts = {k: v.to(self.device) for k, v in adr_texts.items()}
                    
                    outputs = self.model(graphs, adr_texts)
                    predictions = outputs.cpu().numpy()
                    
                    all_predictions.extend(predictions)
                    all_labels.extend(labels.cpu().numpy())
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"   Processed {batch_idx + 1} batches...")
                        
                except Exception as e:
                    print(f"   Error in batch {batch_idx+1}: {e}")
                    continue
        
        if len(all_predictions) == 0:
            print("❌ No predictions were made!")
            return all_predictions_df, correct_predictions_df, wrong_predictions_df
            
        print(f"   ✓ Made predictions for {len(all_predictions)} samples")
        
        # 4. Create dataframes
        print("\nStep 4: Creating prediction tables...")
        try:
            # Create all predictions dataframe
            all_predictions_df = self._create_predictions_table(test_data, all_predictions, all_labels)
            
            # Split into correct and wrong predictions
            correct_predictions_df, wrong_predictions_df = self._split_correct_wrong(all_predictions_df)
            
            # Save to CSV files
            self._save_prediction_tables(all_predictions_df, correct_predictions_df, wrong_predictions_df)
            
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            return all_predictions_df, correct_predictions_df, wrong_predictions_df
        
        # 5. Show summary
        print("\nStep 5: Prediction Summary")
        print("-" * 50)
        
        # Calculate statistics
        total = len(all_predictions_df)
        correct = len(correct_predictions_df)
        wrong = len(wrong_predictions_df)
        
        print(f"Total predictions: {total}")
        print(f"Correct predictions: {correct} ({correct/total*100:.1f}%)")
        print(f"Wrong predictions: {wrong} ({wrong/total*100:.1f}%)")
        
        if total > 0:
            # Breakdown of wrong predictions
            if len(wrong_predictions_df) > 0:
                false_positives = len(wrong_predictions_df[wrong_predictions_df['error_type'] == 'False Positive'])
                false_negatives = len(wrong_predictions_df[wrong_predictions_df['error_type'] == 'False Negative'])
                print(f"\nWrong prediction breakdown:")
                print(f"  False Positives: {false_positives}")
                print(f"  False Negatives: {false_negatives}")
        
        # Calculate metrics if labels available
        if len(all_labels) > 0:
            try:
                from evaluator import Evaluator
                evaluator = Evaluator()
                metrics = evaluator.calculate_metrics(all_labels, all_predictions)
                print(f"\nModel Performance on Test Data:")
                print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1-Score: {metrics['f1']:.4f}")
                
                # Save metrics
                metrics_df = pd.DataFrame([metrics])
                metrics_df.to_csv('test_metrics.csv', index=False)
                print(f"  ✓ Metrics saved to 'test_metrics.csv'")
                
            except Exception as e:
                print(f"  Could not calculate metrics: {e}")
        
        return all_predictions_df, correct_predictions_df, wrong_predictions_df
    
    def _create_predictions_table(self, test_data, predictions, true_labels=None):
        """Create a dataframe with all predictions"""
        rows = []
        for i, item in enumerate(test_data[:len(predictions)]):
            pred_prob = float(predictions[i])
            pred_binary = 1 if pred_prob > 0.5 else 0
            
            row = {
                'index': i,
                'smiles': str(item['smiles'])[:100],
                'adr': str(item['adr'])[:200],
                'prediction_probability': pred_prob,
                'prediction_binary': pred_binary,
                'confidence': abs(pred_prob - 0.5) * 2  # 0 to 1 scale
            }
            
            if true_labels is not None and i < len(true_labels):
                true_label = int(true_labels[i])
                row['true_label'] = true_label
                row['is_correct'] = 1 if pred_binary == true_label else 0
                
                # Determine error type
                if pred_binary == true_label:
                    row['error_type'] = 'Correct'
                elif pred_binary == 1 and true_label == 0:
                    row['error_type'] = 'False Positive'
                else:  # pred_binary == 0 and true_label == 1
                    row['error_type'] = 'False Negative'
                
                # Add confidence labels
                if row['confidence'] >= 0.7:
                    row['confidence_level'] = 'High'
                elif row['confidence'] >= 0.4:
                    row['confidence_level'] = 'Medium'
                else:
                    row['confidence_level'] = 'Low'
            else:
                row['true_label'] = None
                row['is_correct'] = None
                row['error_type'] = 'Unknown'
                row['confidence_level'] = 'Unknown'
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _split_correct_wrong(self, predictions_df):
        """Split predictions into correct and wrong dataframes"""
        if 'is_correct' not in predictions_df.columns:
            return pd.DataFrame(), pd.DataFrame()
        
        # Correct predictions
        correct_df = predictions_df[predictions_df['is_correct'] == 1].copy()
        
        # Wrong predictions
        wrong_df = predictions_df[predictions_df['is_correct'] == 0].copy()
        
        # Sort wrong predictions by confidence (lowest confidence first)
        if not wrong_df.empty:
            wrong_df = wrong_df.sort_values('confidence', ascending=True)
        
        # Sort correct predictions by confidence (highest confidence first)
        if not correct_df.empty:
            correct_df = correct_df.sort_values('confidence', ascending=False)
        
        return correct_df, wrong_df
    
    def _save_prediction_tables(self, all_df, correct_df, wrong_df):
        """Save all prediction tables to CSV files"""
        # Save all predictions
        if not all_df.empty:
            all_df.to_csv('all_predictions.csv', index=False)
            print(f"✓ All predictions saved to 'all_predictions.csv' ({len(all_df)} rows)")
        
        # Save correct predictions
        if not correct_df.empty:
            correct_df.to_csv('correct_predictions.csv', index=False)
            print(f"✓ Correct predictions saved to 'correct_predictions.csv' ({len(correct_df)} rows)")
            
            # Save sample of correct predictions
            correct_df.head(20).to_csv('sample_correct_predictions.csv', index=False)
        
        # Save wrong predictions
        if not wrong_df.empty:
            wrong_df.to_csv('wrong_predictions.csv', index=False)
            print(f"✓ Wrong predictions saved to 'wrong_predictions.csv' ({len(wrong_df)} rows)")
            
            # Save sample of wrong predictions
            wrong_df.head(20).to_csv('sample_wrong_predictions.csv', index=False)
        
        # Save summary statistics
        self._save_summary_statistics(all_df, correct_df, wrong_df)
    
    def _save_summary_statistics(self, all_df, correct_df, wrong_df):
        """Save summary statistics to a CSV file"""
        if all_df.empty:
            return
        
        stats = {
            'total_predictions': len(all_df),
            'correct_predictions': len(correct_df),
            'wrong_predictions': len(wrong_df),
            'accuracy': len(correct_df) / len(all_df) if len(all_df) > 0 else 0,
            'average_confidence': all_df['confidence'].mean() if 'confidence' in all_df.columns else 0,
            'average_prediction_probability': all_df['prediction_probability'].mean()
        }
        
        # Error type breakdown if available
        if 'error_type' in all_df.columns:
            error_counts = all_df['error_type'].value_counts()
            for error_type, count in error_counts.items():
                stats[f'count_{error_type.lower().replace(" ", "_")}'] = count
        
        # Confidence level breakdown
        if 'confidence_level' in all_df.columns:
            confidence_counts = all_df['confidence_level'].value_counts()
            for level, count in confidence_counts.items():
                stats[f'count_{level.lower()}_confidence'] = count
        
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv('prediction_summary.csv', index=False)
        print(f"✓ Summary statistics saved to 'prediction_summary.csv'")
    
    def show_prediction_tables(self, n_samples=10):
        """Display samples from all prediction tables"""
        print("\n" + "=" * 80)
        print("PREDICTION TABLES SUMMARY")
        print("=" * 80)
        
        # Check files exist
        files_exist = all(os.path.exists(f) for f in ['all_predictions.csv', 'correct_predictions.csv', 'wrong_predictions.csv'])
        
        if not files_exist:
            print("Prediction files not found. Run make_predictions() first.")
            return
        
        # Load dataframes
        all_df = pd.read_csv('all_predictions.csv')
        correct_df = pd.read_csv('correct_predictions.csv')
        wrong_df = pd.read_csv('wrong_predictions.csv')
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total predictions: {len(all_df)}")
        print(f"   Correct predictions: {len(correct_df)} ({len(correct_df)/len(all_df)*100:.1f}%)")
        print(f"   Wrong predictions: {len(wrong_df)} ({len(wrong_df)/len(all_df)*100:.1f}%)")
        
        # Show sample correct predictions
        print(f"\n✅ SAMPLE CORRECT PREDICTIONS (Top {min(n_samples, len(correct_df))}):")
        print("-" * 80)
        if not correct_df.empty:
            for i, (_, row) in enumerate(correct_df.head(n_samples).iterrows()):
                print(f"{i+1}. Smiles: {row['smiles'][:40]}...")
                print(f"   ADR: {row['adr'][:50]}...")
                print(f"   Prediction: {row['prediction_probability']:.3f} "
                      f"({'Positive' if row['prediction_binary'] == 1 else 'Negative'})")
                print(f"   True Label: {'Positive' if row['true_label'] == 1 else 'Negative'}")
                print(f"   Confidence: {row['confidence']:.3f} ({row.get('confidence_level', 'N/A')})")
                print()
        else:
            print("No correct predictions")
        
        # Show sample wrong predictions
        print(f"\n❌ SAMPLE WRONG PREDICTIONS (Top {min(n_samples, len(wrong_df))}):")
        print("-" * 80)
        if not wrong_df.empty:
            for i, (_, row) in enumerate(wrong_df.head(n_samples).iterrows()):
                print(f"{i+1}. Smiles: {row['smiles'][:40]}...")
                print(f"   ADR: {row['adr'][:50]}...")
                print(f"   Prediction: {row['prediction_probability']:.3f} "
                      f"({'Positive' if row['prediction_binary'] == 1 else 'Negative'})")
                print(f"   True Label: {'Positive' if row['true_label'] == 1 else 'Negative'}")
                print(f"   Error Type: {row.get('error_type', 'Unknown')}")
                print(f"   Confidence: {row['confidence']:.3f} ({row.get('confidence_level', 'N/A')})")
                print()
        else:
            print("No wrong predictions")
        
        # Show error type breakdown
        if not wrong_df.empty and 'error_type' in wrong_df.columns:
            error_counts = wrong_df['error_type'].value_counts()
            print(f"\n📈 ERROR BREAKDOWN:")
            for error_type, count in error_counts.items():
                percentage = count / len(wrong_df) * 100
                print(f"   {error_type}: {count} ({percentage:.1f}%)")
        
        return all_df, correct_df, wrong_df


# Simple function to run everything
def run_predictions() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One function to run all predictions and return tables"""
    print("Starting predictions on 10% test data...")
    predictor = Predictor()
    
    # Get prediction tables
    all_df, correct_df, wrong_df = predictor.make_predictions()
    
    if not all_df.empty:
        # Show tables
        predictor.show_prediction_tables(5)
        
        print("\n" + "=" * 60)
        print("✅ PREDICTIONS COMPLETE!")
        print("=" * 60)
        print("\n📁 Files created:")
        print("1. 'all_predictions.csv' - All predictions with details")
        print("2. 'correct_predictions.csv' - Only correctly predicted samples")
        print("3. 'wrong_predictions.csv' - Only incorrectly predicted samples")
        print("4. 'sample_correct_predictions.csv' - Sample of correct predictions")
        print("5. 'sample_wrong_predictions.csv' - Sample of wrong predictions")
        print("6. 'prediction_summary.csv' - Summary statistics")
        print("7. 'test_metrics.csv' - Performance metrics")
    else:
        print("\n❌ No predictions were made.")
    
    return all_df, correct_df, wrong_df


if __name__ == "__main__":
    # When you run this file directly, it will make predictions
    all_predictions, correct_predictions, wrong_predictions = run_predictions()
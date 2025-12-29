import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix, 
    average_precision_score, precision_recall_curve, roc_curve
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

class Evaluator:
    """Model evaluation and metrics calculation"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        y_pred_class = (y_pred > threshold).astype(int)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred_class)
        precision = precision_score(y_true, y_pred_class, zero_division=0)
        recall = recall_score(y_true, y_pred_class, zero_division=0)
        f1 = f1_score(y_true, y_pred_class, zero_division=0)
        
        # AUC scores
        auc_roc = roc_auc_score(y_true, y_pred)
        auc_pr = average_precision_score(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_class)
        
        # Additional metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'fpr': fpr,
            'confusion_matrix': cm,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
    
    @staticmethod
    def save_metrics_to_csv(metrics_dict: Dict[str, Dict], filename: str = 'evaluation_metrics.csv'):
        """Save metrics to CSV file"""
        metrics_data = []
        for model_name, metrics in metrics_dict.items():
            row = {'model': model_name}
            row.update(metrics)
            metrics_data.append(row)
        
        df = pd.DataFrame(metrics_data)
        df.to_csv(filename, index=False)
        print(f"Metrics saved to {filename}")
        return df
    
    @staticmethod
    def plot_comparison(metrics_dict: Dict[str, Dict], output_file: str = 'metrics_comparison.png'):
        """Plot comparison of metrics across different models/architectures"""
        metrics_names = ['auc_roc', 'auc_pr', 'f1', 'accuracy', 'precision', 'recall']
        model_names = list(metrics_dict.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics_names):
            values = [metrics_dict[model][metric] for model in model_names]
            axes[i].bar(model_names, values, color=['skyblue', 'lightcoral', 'lightgreen'])
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_roc_curves(roc_data: Dict[str, tuple], output_file: str = 'roc_curves.png'):
        """Plot ROC curves for multiple models"""
        plt.figure(figsize=(8, 6))
        
        for model_name, (fpr, tpr, auc) in roc_data.items():
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_pr_curves(pr_data: Dict[str, tuple], output_file: str = 'pr_curves.png'):
        """Plot Precision-Recall curves for multiple models"""
        plt.figure(figsize=(8, 6))
        
        for model_name, (precision, recall, auc_pr) in pr_data.items():
            plt.plot(recall, precision, label=f'{model_name} (AUC = {auc_pr:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
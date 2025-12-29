# simple_predictor.py
"""
Simple script for users to make predictions with their own data
Usage: python simple_predictor.py --file your_data.csv
"""

import argparse
import pandas as pd
import sys
import os

sys.path.append('.')

def main():
    print("=" * 60)
    print("DRUG-ADR PREDICTION SYSTEM")
    print("=" * 60)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='Input CSV file')
    parser.add_argument('--output', default='predictions.csv', help='Output file')
    args = parser.parse_args()
    
    # Check file exists
    if not os.path.exists(args.file):
        print(f" Error: File '{args.file}' not found")
        return
    
    try:
        # Load user data
        print(f"📁 Loading {args.file}...")
        df = pd.read_csv(args.file)
        
        # Check required columns
        if 'smiles' not in df.columns or 'adr' not in df.columns:
            print(" Error: File must have 'smiles' and 'adr' columns")
            print("   See examples/sample_data.csv for format")
            return
        
        print(f" Found {len(df)} records")
        
        # Import and run predictions
        from prediction_module import Predictor
        predictor = Predictor('models/best_model.pth')
        
        print(" Making predictions...")
        
        # Convert to list format
        data = []
        for _, row in df.iterrows():
            data.append({
                'smiles': str(row['smiles']),
                'adr': str(row['adr']),
                'label': 0 
            })
        
        # Get predictions
        from data_loader import ADRDataset
        from torch.utils.data import DataLoader
        
        dataset = ADRDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(
                batch, 
                predictor.graph_processor, 
                predictor.sapbert_tokenizer
            )
        )
        
        predictions = predictor.predict_batch(loader)
        
        # Add predictions to dataframe
        df['prediction_probability'] = predictions
        df['prediction'] = (predictions > 0.5).astype(int)
        df['prediction_label'] = df['prediction'].map({0: 'Negative', 1: 'Positive'})
        
        # Save results
        df.to_csv(args.output, index=False)
        
        print(f"\nPREDICTIONS COMPLETE!")
        print(f"Results saved to: {args.output}")
        print(f"Summary: {len(df)} predictions made")
        
    except Exception as e:
        print(f" Error: {e}")

def collate_fn(batch, graph_processor, sapbert_tokenizer):
    from graph_processor import collate_fn as original_collate_fn
    return original_collate_fn(batch, graph_processor, sapbert_tokenizer)

if __name__ == '__main__':
    main()
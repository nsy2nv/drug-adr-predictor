# Drug-ADR Prediction System

Predict Adverse Drug Reactions from drug SMILES strings using hybrid GNN + Transformer model.

## 🚀 Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/nsy2nv/drug-adr-predictor.git
cd drug-adr-predictor
pip install -r requirements.txt

Test Installation
# bash
python test_installation.py
python examples/test_prediction.py

Make Predictions with Your Data
# bash

# 1. Create CSV file with columns: smiles, adr
#    Example: examples/sample_data.csv

# 2. Run predictions
python simple_predictor.py --file your_data.csv

# 3. Results saved to predictions.csv

Input Format
Your CSV file needs:

smiles: Drug SMILES string

adr: Adverse reaction text

E.g
smiles,adr
CC(=O)OC1=CC=CC=C1C(=O)O,gastrointestinal bleeding
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,insomnia
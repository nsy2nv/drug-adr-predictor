# test_installation.py
"""
Test if all dependencies are installed
"""
import importlib

required = [
    'torch',
    'pandas',
    'numpy',
    'sklearn',
    'transformers',
    'rdkit'
]

print("Testing dependencies...")
for package in required:
    try:
        importlib.import_module(package)
        print(f" {package}")
    except:
        print(f" {package} - Install with: pip install {package}")

print("\nIf all , installation successful!")
print("Run: python examples/test_prediction.py")
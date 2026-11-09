# MIF Framework: Machine Intelligence for Finance

This framework is a professional implementation of the methods described in the *Machine Intelligence for Finance* research. It transforms fragmented educational notebooks into a cohesive, production-ready library for financial machine learning.

## 🌟 Key Features

- **Information Time Sampling**: Implements Tick, Volume, and Dollar bars to eliminate chronological noise.
- **Fractional Differentiation**: Achieves stationarity while preserving long-term memory of price action.
- **Triple Barrier Labeling**: A volatility-aware labeling method for binary and ternary classification.
- **Purged Cross-Validation**: Prevents data leakage in time-series ML using purging and embargoing.
- **Hierarchical Risk Parity (HRP)**: A robust portfolio allocation method based on hierarchical clustering.

## 🚀 Quick Start

### Installation
Ensure you have `pandas`, `numpy`, and `scipy` installed.

### Usage
```python
from mif_framework.pipeline import MIFPipeline
import pandas as pd

# Load your data (must have 'price' and 'volume' columns)
df = pd.read_csv("your_market_data.csv")

# Initialize and run the pipeline
pipeline = MIFPipeline(sampling_type='dollar', sampling_threshold=10000)
results = pipeline.run(df)

print(results['weights']) # View HRP weights
```

## 🏗️ Architecture

- `data_preprocessing/`: Sampling and Differentiation.
- `labeling/`: Triple Barrier logic.
- `validation/`: Purged CV logic.
- `portfolio/`: Risk parity and allocation.
- `pipeline.py`: The integrated entry point.

## 📜 License
BSD 3-Clause License.

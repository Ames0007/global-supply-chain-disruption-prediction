# Global Supply Chain Neural Networks

A Kaggle notebook exploring global supply chain risk and logistics data and using a neural network to predict whether a supply chain disruption occurs.

## Notebook workflow

- Exploratory data analysis and visualization
- Distribution and correlation analysis
- Data preprocessing and feature transformation
- Categorical feature encoding
- Date/year feature extraction
- Train/test split and feature scaling
- Neural network classification with TensorFlow/Keras
- Early stopping during training
- Model evaluation with accuracy, F1 score, classification report, and confusion matrix

## Dataset

The notebook uses the Kaggle dataset **Global Supply Chain Risk and Logistics 2024–2026** and loads:

`global_supply_chain_risk_2026.csv`

The original Kaggle path used in the notebook is:

```python
df = pd.read_csv("/kaggle/input/global-supply-chain-risk-and-logistics-2024-2026/global_supply_chain_risk_2026.csv")
```

## Run on Kaggle

Open `global-supply-chain-neural-networks.ipynb` in Kaggle and attach the **Global Supply Chain Risk and Logistics 2024–2026** dataset before running the notebook.

# Hybrid NIDS Project Walkthrough

The ML pipeline preprocessing bug has been successfully resolved and the new models have been trained and evaluated in an isolated Python 3.10 environment to fix the TensorFlow compatibility issue.

## 1. Issues Identified and Corrected
1. **Preprocessing Bug (Categorical Mismatch)**: 
   - *Issue*: The original Jupyter notebooks trained the model using One-Hot Encoding (`pd.get_dummies`) for categorical features, which expanded the feature set to over 122 columns. However, the FastAPI backend explicitly expects exactly 41 features and uses Label Encoding (`LabelEncoder`).
   - *Correction*: I wrote a unified `train_model.py` script that enforces `LabelEncoder` across the pipeline, ensuring the trained Autoencoder and Isolation Forest strictly accept 41 features.
2. **Dataset Confusion**: 
   - *Issue*: The Jupyter notebooks accidentally swapped the Train and Test datasets and used hardcoded absolute paths (`C:/AI_datasets/...`) that would break on other machines.
   - *Correction*: The new training script uses relative paths to point to the `data/` folder and assigns `KDDTrain+_20Percent.txt` to Train and `KDDTest+.txt` to Test correctly.
3. **Python Environment Error (TensorFlow Incompatibility)**: 
   - *Issue*: The backend setup failed because TensorFlow does not yet support Python 3.14 (which is installed globally on the system).
   - *Correction*: I bypassed this by using `uv` to dynamically create a `.venv` with Python 3.10, installing all requirements safely inside it.

## 2. Model Evaluation Results
The unified `train_model.py` script was executed on the NSL-KDD test set. The models (Autoencoder + Isolation Forest) achieved the following results using the optimal ROC threshold (`0.1160`):

| Metric | Score |
| --- | --- |
| **Accuracy** | 79.55% |
| **Precision** | 88.56% |
| **Recall** | 73.58% |
| **F1 Score** | 80.38% |
| **False Positive Rate (FPR)** | 12.56% |

### Confusion Matrix
| | Predicted Normal | Predicted Anomaly |
| --- | --- | --- |
| **True Normal** | 8,491 (TN) | 1,220 (FP) |
| **True Anomaly** | 3,390 (FN) | 9,443 (TP) |

> [!NOTE]
> The Hybrid Model's precision (88.5%) is quite strong, meaning when it flags an anomaly, it is highly likely to be a real attack (minimizing false alerts). However, the recall (73.5%) indicates some attacks are still slipping through as normal traffic. This is typical for unsupervised/hybrid baselines on NSL-KDD.

## 3. How to Launch the Project Now

Since we created a `.venv` to bypass the Python 3.14 issue, use the following commands to launch your backend:

**Terminal 1: Start the Backend (FastAPI)**
```powershell
cd "d:\hybrid nids main\hybrid-nids-main\src\backend"
..\..\.venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

**Terminal 2: Start the Frontend (React)**
```powershell
cd "d:\hybrid nids main\hybrid-nids-main\src\frontend"
npm install
npm run dev
```

# binary_cross_validation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("STRATIFIED 5-FOLD CROSS VALIDATION - BINARY CLASSIFICATION")
print("="*70)

# ==================== Load & Preprocess Dataset ====================
print("Loading dataset")
df = pd.read_csv('Obfuscated-MalMem2022.csv', sep=';')

# Fix European number format
print("Fixing number format...")
object_cols = [col for col in df.select_dtypes(include=['object']).columns 
               if col not in ['Class', 'Category', 'Class4', 'ClassBinary']]

for col in object_cols:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    df[col] = df[col].str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values
df = df.dropna(subset=['Class']).copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('Class', errors='ignore')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

X = df.drop(['Class', 'Category', 'Class4', 'ClassBinary'], axis=1, errors='ignore')
y = df['Class']

print(f"Total samples: {len(X)}")
print(f"Class distribution:\n{y.value_counts()}")

# ==================== Stratified 5-Fold CV ====================
print("\nStarting Stratified 5-Fold Cross Validation")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
fold = 1

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    print(f"\nFold {fold}/5")
    
    model = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.1,
        loss_function='Logloss',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    accuracies.append(acc)
    
    print(f"Accuracy Fold {fold}: {acc*100:.4f}%")
    fold += 1

# ==================== Final Results ====================
print("\n" + "="*60)
print("CROSS VALIDATION RESULTS")
print("="*60)
print(f"Average Accuracy      : {np.mean(accuracies)*100:.4f}%")
print(f"Standard Deviation    : {np.std(accuracies)*100:.4f}%")
print(f"Minimum Accuracy      : {np.min(accuracies)*100:.4f}%")
print(f"Maximum Accuracy      : {np.max(accuracies)*100:.4f}%")

# Save results
with open('binary_cross_validation.txt', 'w') as f:
    f.write("=== STRATIFIED 5-FOLD CROSS VALIDATION - BINARY ===\n\n")
    f.write(f"Average Accuracy      : {np.mean(accuracies)*100:.4f}%\n")
    f.write(f"Standard Deviation    : {np.std(accuracies)*100:.4f}%\n")
    f.write(f"Minimum Accuracy      : {np.min(accuracies)*100:.4f}%\n")
    f.write(f"Maximum Accuracy      : {np.max(accuracies)*100:.4f}%\n")

print("\nCross Validation results saved to: binary_cross_validation.txt")
print("Completed.")
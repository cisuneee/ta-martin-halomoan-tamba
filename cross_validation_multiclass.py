import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

print("=== STRATIFIED 5-FOLD CROSS VALIDATION - MULTICLASS ===")

# Load and Clean Dataset
print("Loading and cleaning dataset")
df = pd.read_csv("Obfuscated-MalMem2022.csv", sep=';')

# Data Cleaning (European numeric format)
numeric_cols = [col for col in df.columns if col not in ['Category', 'Class']]
for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("Data cleaning completed.")

# Mapping to 4 Classes
def map_to_4classes(cat):
    cat = str(cat).strip()
    if cat == 'Benign': return 'Benign'
    elif cat.startswith('Ransomware'): return 'Ransomware'
    elif cat.startswith('Spyware'): return 'Spyware'
    elif cat.startswith('Trojan'): return 'Trojan'
    else: return 'Benign'

df['Class4'] = df['Category'].apply(map_to_4classes)

X = df.drop(columns=['Category', 'Class', 'Class4'])
y = df['Class4']

# Stratified 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
fold = 1

print("Starting Stratified 5-Fold Cross Validation\n")

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.1,
        depth=8,
        loss_function='MultiClass',
        verbose=0,
        random_seed=42
    )
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    accuracies.append(acc)
    
    print(f"Fold {fold} - Accuracy: {acc:.4f}")
    fold += 1

# Final Results
print("\n" + "="*65)
print("STRATIFIED 5-FOLD CROSS VALIDATION RESULTS")
print("="*65)
print(f"Average Accuracy       : {np.mean(accuracies):.4f}")
print(f"Standard Deviation     : {np.std(accuracies):.4f}")
print(f"Minimum Accuracy       : {np.min(accuracies):.4f}")
print(f"Maximum Accuracy       : {np.max(accuracies):.4f}")
print("="*65)
# cross_validation_individual_class.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print("STRATIFIED 5-FOLD CROSS VALIDATION - INDIVIDUAL MALWARE (16 Classes)")
print("="*90)

# ==================== Load & Preprocess ====================
print("Loading dataset")
df = pd.read_csv('Obfuscated-MalMem2022.csv', sep=';')

# Fix number format
object_cols = [col for col in df.select_dtypes(include=['object']).columns 
               if col not in ['Class', 'Category', 'Class4', 'ClassBinary']]

for col in object_cols:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    df[col] = df[col].str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['Category']).copy()

# ==================== 16 Individual Classes Mapping ====================
def map_to_16_classes(cat):
    c = str(cat)
    if 'Benign' in c:
        return 'Benign'
    # Ransomware
    elif 'Ransomware-Shade' in c: return 'Ransomware-Shade'
    elif 'Ransomware-Ako' in c: return 'Ransomware-Ako'
    elif 'Ransomware-Conti' in c: return 'Ransomware-Conti'
    elif 'Ransomware-Maze' in c: return 'Ransomware-Maze'
    elif 'Ransomware-Pysa' in c: return 'Ransomware-Pysa'
    # Spyware
    elif 'Spyware-Gator' in c: return 'Spyware-Gator'
    elif 'Spyware-Transponder' in c: return 'Spyware-Transponder'
    elif 'Spyware-180solutions' in c: return 'Spyware-180solutions'
    elif 'Spyware-CWS' in c: return 'Spyware-CWS'
    elif 'Spyware-TIBS' in c: return 'Spyware-TIBS'
    # Trojan
    elif 'Trojan-Refroso' in c: return 'Trojan-Refroso'
    elif 'Trojan-Scar' in c: return 'Trojan-Scar'
    elif 'Trojan-Emotet' in c: return 'Trojan-Emotet'
    elif 'Trojan-Zeus' in c: return 'Trojan-Zeus'
    elif 'Trojan-Reconyc' in c: return 'Trojan-Reconyc'
    else:
        return 'Other'

df['Category_16'] = df['Category'].apply(map_to_16_classes)
df = df[df['Category_16'] != 'Other'].copy()

print(f"Final samples: {len(df)}")
print(f"Number of classes: {len(df['Category_16'].unique())}")
print("\nClass distribution:\n", df['Category_16'].value_counts().sort_values(ascending=False))

X = df.drop(['Class', 'Category', 'Class4', 'ClassBinary', 'Category_16'], axis=1, errors='ignore')
y = df['Category_16']

# ==================== Stratified 5-Fold ====================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
fold = 1

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    print(f"\nFold {fold}/5 training")
    
    model = CatBoostClassifier(
        iterations=600,
        depth=8,
        learning_rate=0.1,
        loss_function='MultiClass',
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
print("\n" + "="*70)
print("CROSS VALIDATION RESULTS - 16 CLASSES")
print("="*70)
print(f"Average Accuracy      : {np.mean(accuracies)*100:.4f}%")
print(f"Standard Deviation    : {np.std(accuracies)*100:.4f}%")
print(f"Minimum Accuracy      : {np.min(accuracies)*100:.4f}%")
print(f"Maximum Accuracy      : {np.max(accuracies)*100:.4f}%")

with open('cv_16classes_results.txt', 'w') as f:
    f.write("=== STRATIFIED 5-FOLD CV - 16 CLASSES INDIVIDUAL MALWARE ===\n\n")
    f.write(f"Average Accuracy      : {np.mean(accuracies)*100:.4f}%\n")
    f.write(f"Standard Deviation    : {np.std(accuracies)*100:.4f}%\n")
    f.write(f"Minimum Accuracy      : {np.min(accuracies)*100:.4f}%\n")
    f.write(f"Maximum Accuracy      : {np.max(accuracies)*100:.4f}%\n")

print("\nResults saved to: cv_16classes_results.txt")
print("Cross Validation Completed!")
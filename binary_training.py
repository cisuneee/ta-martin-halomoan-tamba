# binary_training_final.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("START BINARY CLASSIFICATION TRAINING")
print("="*70)

# ==================== 1. Load Dataset ====================
print("Loading dataset")
df = pd.read_csv('Obfuscated-MalMem2022.csv', sep=';')
print(f"Original samples: {len(df)}")

# ==================== 2. FIX NUMBER FORMAT (Hanya kolom numerik) ====================
print("Fixing European number format")

# Jangan ubah kolom target dan kategori
target_cols = ['Class', 'Category', 'Class4', 'ClassBinary']
object_cols = [col for col in df.select_dtypes(include=['object']).columns if col not in target_cols]

for col in object_cols:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    df[col] = df[col].str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("Number format fixed.")

# ==================== 3. Handle Missing Values ====================
print("Handling missing values")

# Drop only if target is NaN
df = df.dropna(subset=['Class']).copy()

# Fill NaN only in feature columns
feature_cols = [col for col in df.columns if col not in target_cols]
numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns

if len(numeric_features) > 0:
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

print(f"Final samples: {len(df)}")

# ==================== 4. Preprocessing ====================
X = df.drop(target_cols, axis=1, errors='ignore')
y = df['Class']

print(f"Features: {X.shape[1]}")
print(f"Class distribution:\n{y.value_counts()}")

# ==================== 5. Train-Test Split ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set : {X_train.shape[0]} samples")
print(f"Test set     : {X_test.shape[0]} samples")

# ==================== 6. Training ====================
print("\nTraining CatBoost Binary Model...")

model_binary = CatBoostClassifier(
    iterations=800,
    depth=8,
    learning_rate=0.1,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

model_binary.fit(X_train, y_train, eval_set=(X_test, y_test), plot=False)

# ==================== 7. Evaluation ====================
print("\n" + "="*60)
print("BINARY CLASSIFICATION RESULTS")
print("="*60)

y_pred = model_binary.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['Benign', 'Malware']))

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc*100:.4f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malware'],
            yticklabels=['Benign', 'Malware'])
plt.title('Confusion Matrix - Binary Classification')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_binary.png', dpi=300, bbox_inches='tight')
print("Confusion Matrix saved as: confusion_matrix_binary.png")

model_binary.save_model('model_binary.cbm')
print("Model saved as: model_binary.cbm")

print("\nBinary Training Completed Successfully!")
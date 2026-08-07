# test_pi_binary.py

import pandas as pd
import numpy as np
import time
import psutil
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("RASPBERRY PI BINARY CLASSIFICATION TESTING")
print("="*70)

# ==================== Load Model ====================
print("Loading binary model...")
model = CatBoostClassifier()
model.load_model('model_binary.cbm')
print("Model binary loaded successfully!")

# ==================== Load Dataset ====================
print("Loading dataset for testing")
df = pd.read_csv('Obfuscated-MalMem2022.csv', sep=';')

# Same preprocessing as training
target_cols = ['Class', 'Category', 'Class4', 'ClassBinary']
object_cols = [col for col in df.select_dtypes(include=['object']).columns if col not in target_cols]

for col in object_cols:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    df[col] = df[col].str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Use only features
X = df.drop(target_cols, axis=1, errors='ignore')
y = df['Class']

# ==================== Stratified Split (80:20) ====================
print("Performing stratified split (80:20)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Test samples: {len(X_test)}")

# ==================== Testing on Raspberry Pi ====================
print("\nStarting inference on Raspberry Pi")

start_time = time.time()
memory_before = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

y_pred = model.predict(X_test)

end_time = time.time()
memory_after = psutil.Process().memory_info().rss / (1024 * 1024)

total_time = end_time - start_time
avg_inference_time = (total_time / len(X_test)) * 1000  # ms

print("\n" + "="*60)
print("BINARY TESTING RESULTS ON RASPBERRY PI")
print("="*60)
print(f"Total Inference Time : {total_time:.2f} seconds")
print(f"Average Inference Time: {avg_inference_time:.4f} ms per sample")
print(f"Memory Usage         : {memory_after - memory_before:.2f} MB")

# ==================== Evaluation ====================
from sklearn.metrics import classification_report, accuracy_score

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malware']))

acc = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {acc*100:.4f}%")

# Save results
with open('hasil_binary_pi.txt', 'w') as f:
    f.write("=== BINARY CLASSIFICATION RESULTS ON RASPBERRY PI ===\n\n")
    f.write(f"Accuracy: {acc*100:.4f}%\n")
    f.write(f"Total Time: {total_time:.2f} seconds\n")
    f.write(f"Avg Inference Time: {avg_inference_time:.4f} ms/sample\n")
    f.write(f"Memory Usage: {memory_after - memory_before:.2f} MB\n\n")
    f.write(classification_report(y_test, y_pred, target_names=['Benign', 'Malware']))

print("\nResults saved to: hasil_binary_pi.txt")
print("Binary Testing Completed!")
import time
import psutil
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

print("=== RASPBERRY PI FULL 4-CLASSES TESTING (TEST SET) ===")

# Load Model
print("Loading model...")
model = CatBoostClassifier()
model.load_model("model_malware_4classes.cbm")

# Load Dataset
print("Loading dataset...")
df = pd.read_csv("Obfuscated-MalMem2022.csv", sep=';')

# Data Cleaning
numeric_cols = [col for col in df.columns if col not in ['Category', 'Class']]
for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Mapping to 4 Classes
def map_to_4classes(cat):
    cat = str(cat).strip()
    if cat == 'Benign': return 'Benign'
    elif cat.startswith('Ransomware'): return 'Ransomware'
    elif cat.startswith('Spyware'): return 'Spyware'
    elif cat.startswith('Trojan'): return 'Trojan'
    else: return 'Benign'

df['Class4'] = df['Category'].apply(map_to_4classes)

# Split Data (same as training)
X = df.drop(columns=['Category', 'Class', 'Class4'])
y = df['Class4']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Test set ready: {len(X_test)} samples")

# Monitoring Resource + Inference
mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
start_time = time.time()

predictions = model.predict(X_test)

end_time = time.time()
mem_after = psutil.Process().memory_info().rss / (1024 * 1024)

# Results
accuracy = accuracy_score(y_test, predictions) * 100
total_time = end_time - start_time
avg_time = total_time / len(X_test) * 1000

print("\n" + "="*70)
print("FULL 4-CLASSES TESTING RESULTS ON RASPBERRY PI")
print("="*70)
print(f"Model Accuracy         : {accuracy:.4f}%")
print(f"Total Inference Time   : {total_time:.2f} seconds")
print(f"Avg Inference Time     : {avg_time:.4f} ms per sample")
print(f"Additional Memory Usage: {mem_after - mem_before:.1f} MB")
print("="*70)

print("\nClassification Report:")
print(classification_report(y_test, predictions, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save Results
with open("hasil_pengujian_pi.txt", "w", encoding="utf-8") as f:
    f.write("=== FULL 4-CLASSES TESTING RESULTS (TEST SET) ===\n")
    f.write(f"Accuracy     : {accuracy:.4f}%\n")
    f.write(f"Total Time   : {total_time:.2f} seconds\n")
    f.write(f"Avg Time     : {avg_time:.4f} ms/sample\n")
    f.write(f"Memory Usage : {mem_after - mem_before:.1f} MB\n")

print("\nResults saved to: hasil_pengujian_pi.txt")
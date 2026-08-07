import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

print("=== START MULTICLASS TRAINING (4 Classes) - FINAL ===")

# 1. Load Dataset with Cleaning
print("Loading and cleaning dataset...")
df = pd.read_csv("Obfuscated-MalMem2022.csv", sep=';')

numeric_cols = [col for col in df.columns if col not in ['Category', 'Class']]
for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("Data cleaning completed.")

# ==================== LABELING ====================

# A. Binary Classification (Benign vs Malware)
df['ClassBinary'] = df['Category'].apply(lambda x: 'Benign' if str(x).strip() == 'Benign' else 'Malware')

# B. 4 Classes (Malware Families)
def map_to_4classes(cat):
    cat = str(cat).strip()
    if cat == 'Benign': return 'Benign'
    elif cat.startswith('Ransomware'): return 'Ransomware'
    elif cat.startswith('Spyware'): return 'Spyware'
    elif cat.startswith('Trojan'): return 'Trojan'
    else: return 'Benign'

df['Class4'] = df['Category'].apply(map_to_4classes)

print("\nClass Distribution:")
print("Binary:", df['ClassBinary'].value_counts())
print("4 Classes:", df['Class4'].value_counts())

# Features & Target
X = df.drop(columns=['Category', 'Class', 'Class4', 'ClassBinary'])
y = df['Class4']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTraining set : {X_train.shape[0]} samples")
print(f"Test set     : {X_test.shape[0]} samples")

# Training Model
print("\nTraining CatBoost Multiclass Model...")
model = CatBoostClassifier(
    iterations=800,
    learning_rate=0.1,
    depth=8,
    loss_function='MultiClass',
    verbose=100,
    random_seed=42
)

model.fit(X_train, y_train, eval_set=(X_test, y_test))

# Evaluation
print("\n=== EVALUATION RESULTS (4 Classes) ===")
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n")
print(classification_report(y_test, predictions, digits=4))

cm4 = confusion_matrix(y_test, predictions)

# ==================== GENERATE 2 CONFUSION MATRIX IMAGES ====================

class_labels_4 = ['Benign', 'Ransomware', 'Spyware', 'Trojan']

# 1. Binary Classification
print("\nGenerating Binary Confusion Matrix...")
y_binary_true = df.loc[X_test.index, 'ClassBinary']
pred_binary = ['Benign' if p[0] == 'Benign' else 'Malware' for p in predictions]
cm_binary = confusion_matrix(y_binary_true, pred_binary, labels=['Benign', 'Malware'])

plt.figure(figsize=(6, 5))
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'])
plt.title('Confusion Matrix - Binary Classification\n(Benign vs Malware)', fontsize=14, pad=20)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_binary.png', dpi=300, bbox_inches='tight')
print("Binary Confusion Matrix saved: confusion_matrix_binary.png")

# 2. 4 Classes (Malware Families)
print("Generating 4 Classes Confusion Matrix...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm4, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels_4, yticklabels=class_labels_4)
plt.title('Confusion Matrix - Malware Families Detection\n(4 Classes)', fontsize=14, pad=20)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_4classes.png', dpi=300, bbox_inches='tight')
print("4 Classes Confusion Matrix saved: confusion_matrix_4classes.png")

# Save Model
model.save_model("model_malware_4classes.cbm")
print("\nModel successfully saved: model_malware_4classes.cbm")
print("All Confusion Matrix images have been successfully generated!")
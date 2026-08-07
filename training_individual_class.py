# training_individual_class.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TRAINING 16 CLASSES")
print("="*80)

# Load Dataset
df = pd.read_csv('Obfuscated-MalMem2022.csv', sep=';')

# Fix number format
object_cols = [col for col in df.select_dtypes(include=['object']).columns 
               if col not in ['Class', 'Category', 'Class4', 'ClassBinary']]

for col in object_cols:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    df[col] = df[col].str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['Category']).copy()

# ==================== CLEAN MAPPING TO 16 CLASSES ====================
def clean_family_name(category):
    cat = str(category)
    if 'Benign' in cat:
        return 'Benign'
    # Ransomware
    elif 'Ransomware-Shade' in cat:
        return 'Ransomware-Shade'
    elif 'Ransomware-Ako' in cat:
        return 'Ransomware-Ako'
    elif 'Ransomware-Conti' in cat:
        return 'Ransomware-Conti'
    elif 'Ransomware-Maze' in cat:
        return 'Ransomware-Maze'
    elif 'Ransomware-Pysa' in cat:
        return 'Ransomware-Pysa'
    # Spyware
    elif 'Spyware-Transponder' in cat:
        return 'Spyware-Transponder'
    elif 'Spyware-Gator' in cat:
        return 'Spyware-Gator'
    elif 'Spyware-180solutions' in cat:
        return 'Spyware-180solutions'
    elif 'Spyware-CWS' in cat:
        return 'Spyware-CWS'
    elif 'Spyware-TIBS' in cat:
        return 'Spyware-TIBS'
    # Trojan
    elif 'Trojan-Refroso' in cat:
        return 'Trojan-Refroso'
    elif 'Trojan-Scar' in cat:
        return 'Trojan-Scar'
    elif 'Trojan-Emotet' in cat:
        return 'Trojan-Emotet'
    elif 'Trojan-Zeus' in cat:
        return 'Trojan-Zeus'
    elif 'Trojan-Reconyc' in cat:
        return 'Trojan-Reconyc'
    else:
        return 'Other'

df['Category_16'] = df['Category'].apply(clean_family_name)
df = df[df['Category_16'] != 'Other'].copy()

print(f"Final samples: {len(df)}")
print(f"Number of classes: {len(df['Category_16'].unique())}")
print("\nClass distribution:\n", df['Category_16'].value_counts().sort_index())

# ==================== Training ====================
X = df.drop(['Class', 'Category', 'Class4', 'ClassBinary', 'Category_16'], axis=1, errors='ignore')
y = df['Category_16']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = CatBoostClassifier(
    iterations=1200,           
    depth=8,
    learning_rate=0.08,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=100,   
    task_type="CPU"
)

model.fit(X_train, y_train, eval_set=(X_test, y_test))

# ==================== Evaluation ====================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nOverall Accuracy: {acc*100:.4f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==================== Confusion Matrix ====================
plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_test, y_pred)
classes = sorted(y.unique())
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Individual Malware Classification (16 Classes)', fontsize=14, pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('confusion_matrix_16classes.png', dpi=300, bbox_inches='tight')
print("Confusion Matrix saved as: confusion_matrix_16classes.png")

model.save_model('model_16classes.cbm')
print("Model saved as: model_16classes.cbm")
print("Training Completed!")
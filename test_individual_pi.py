import pandas as pd
import time
import psutil
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("RASPBERRY PI - INDIVIDUAL MALWARE TESTING (16 Classes)")
print("="*80)

# Load Model
model = CatBoostClassifier()
model.load_model('model_16classes.cbm')
print("Model loaded successfully!")

# Load Dataset
df = pd.read_csv('Obfuscated-MalMem2022.csv', sep=';')

# Preprocessing
object_cols = [col for col in df.select_dtypes(include=['object']).columns 
               if col not in ['Class', 'Category', 'Class4', 'ClassBinary']]

for col in object_cols:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False)
    df[col] = df[col].str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['Category']).copy()

# Mapping yang SAMA PERSIS dengan training
def clean_family_name(category):
    cat = str(category)
    if 'Benign' in cat:
        return 'Benign'
    elif 'Ransomware-Shade' in cat: return 'Ransomware-Shade'
    elif 'Ransomware-Ako' in cat: return 'Ransomware-Ako'
    elif 'Ransomware-Conti' in cat: return 'Ransomware-Conti'
    elif 'Ransomware-Maze' in cat: return 'Ransomware-Maze'
    elif 'Ransomware-Pysa' in cat: return 'Ransomware-Pysa'
    elif 'Spyware-Transponder' in cat: return 'Spyware-Transponder'
    elif 'Spyware-Gator' in cat: return 'Spyware-Gator'
    elif 'Spyware-180solutions' in cat: return 'Spyware-180solutions'
    elif 'Spyware-CWS' in cat: return 'Spyware-CWS'
    elif 'Spyware-TIBS' in cat: return 'Spyware-TIBS'
    elif 'Trojan-Refroso' in cat: return 'Trojan-Refroso'
    elif 'Trojan-Scar' in cat: return 'Trojan-Scar'
    elif 'Trojan-Emotet' in cat: return 'Trojan-Emotet'
    elif 'Trojan-Zeus' in cat: return 'Trojan-Zeus'
    elif 'Trojan-Reconyc' in cat: return 'Trojan-Reconyc'
    else:
        return 'Other'

df['Category_16'] = df['Category'].apply(clean_family_name)
df = df[df['Category_16'] != 'Other'].copy()

X = df.drop(['Class', 'Category', 'Class4', 'ClassBinary', 'Category_16'], axis=1, errors='ignore')
y = df['Category_16']

# Split dengan random_state yang SAMA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Test samples: {len(X_test)}")

# Inference
start_time = time.time()
memory_before = psutil.Process().memory_info().rss / (1024 * 1024)

y_pred = model.predict(X_test)

end_time = time.time()
memory_after = psutil.Process().memory_info().rss / (1024 * 1024)

total_time = end_time - start_time
avg_inference = (total_time / len(X_test)) * 1000

print(f"\nOverall Accuracy     : {accuracy_score(y_test, y_pred)*100:.4f}%")
print(f"Avg Inference Time   : {avg_inference:.4f} ms per sample")
print(f"Memory Usage         : {memory_after - memory_before:.2f} MB")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
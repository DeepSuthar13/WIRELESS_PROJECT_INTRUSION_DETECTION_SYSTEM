import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load existing CSI dataset
# Example format: columns = ['amplitude', 'phase', 'label']
# label = 1 (human), 0 (small_animal)
data = pd.read_csv("csi_data.csv")

# Extract basic features
def extract_features(df, window=50):
    features = []
    for i in range(0, len(df) - window, window):
        chunk = df.iloc[i:i+window]
        amp_var = np.var(chunk['amplitude'])
        amp_mean = np.mean(chunk['amplitude'])
        amp_std = np.std(chunk['amplitude'])
        features.append([amp_mean, amp_var, amp_std, chunk['label'].iloc[0]])
    return pd.DataFrame(features, columns=['mean', 'var', 'std', 'label'])

feature_data = extract_features(data)

# Step 3: Train classifier
X = feature_data[['mean', 'var', 'std']]
y = feature_data['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Real-time detection
def detect_intrusion(new_csi_amplitude):
    mean = np.mean(new_csi_amplitude)
    var = np.var(new_csi_amplitude)
    std = np.std(new_csi_amplitude)

    features = scaler.transform([[mean, var, std]])
    prediction = model.predict(features)[0]

    if prediction == 1:
        print("🚨 Human Intrusion Detected!")
    else:
        print("🐀 Small movement ignored.")

# incoming CSI window
sample_data = np.random.normal(0, 1, 50)
detect_intrusion(sample_data)

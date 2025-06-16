# =============================================================================
# Problem Statement
# =============================================================================
"""
Objective: Predict student depression status (0/1) using academic and lifestyle factors
Dataset: 87 students with 21 features including:
- Academic: CGPA, academic workload
- Lifestyle: Sleep patterns, sports engagement
- Mental Health: Anxiety, isolation scores
- Stress Relief Activities: Multi-label categorical data
"""

# 📌 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report  # Add this import

# 📌 Step 2: Load Dataset
file_path = r"C:\Users\HP\Downloads\MentalHealthSurvey.xlsx"  # Change if needed
df = pd.read_excel(file_path)

# 📌 Step 3: Show Basic Info
print(" Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(" Data Types:\n", df.dtypes)
print("Missing Values:\n", df.isnull().sum())
print(" First Rows:\n", df.head())

# 📌 Step 4: Convert Range Columns
def convert_range_to_avg(val):
    try:
        if isinstance(val, str) and '-' in val:
            low, high = map(float, val.split('-'))
            return (low + high) / 2
        return float(val)
    except:
        return np.nan

df['cgpa'] = df['cgpa'].apply(convert_range_to_avg)
df['average_sleep'] = df['average_sleep'].apply(convert_range_to_avg)

# 📌 Step 4.1: Handle Outliers (IQR method)
def handle_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    return df

df = handle_outliers(df, ['cgpa', 'average_sleep'])

# 📌 Step 4.2: Handle Missing Values Properly
# Numeric missing values
df[['cgpa', 'average_sleep']] = df[['cgpa', 'average_sleep']].fillna(df[['cgpa', 'average_sleep']].median())

# Categorical missing values
df.fillna(df.mode().iloc[0], inplace=True)

# 📌 Step 5: Binary Conversion for Depression & Anxiety
df['depression'] = df['depression'].apply(lambda x: 1 if x >= 3 else 0)
df['anxiety'] = df['anxiety'].apply(lambda x: 1 if x >= 3 else 0)

# 📌 Step 6: Encode Categorical Variables
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('stress_relief_activities')  # we'll handle this separately
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 📌 Step 7: Encode Multi-label Stress Relief Activities
df['stress_relief_activities'] = df['stress_relief_activities'].fillna('')
activities = df['stress_relief_activities'].str.get_dummies(sep=', ')
df = pd.concat([df, activities], axis=1)
df.drop('stress_relief_activities', axis=1, inplace=True)

# 📌 Step 8: Normalize Numeric Columns (excluding target)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'depression' in numerical_cols:
    numerical_cols.remove('depression')  # exclude target
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 📌 Step 9: Correlation and Feature Selection
correlation = df.corr()['depression'].sort_values(ascending=False)
selected_features = correlation[1:11].index.tolist()
final_df = df[selected_features + ['depression']]

# ✅ Confirm
print("\n Final dataset exported.")
print("Selected Features:", selected_features)
print("Shape:", final_df.shape)
print(final_df.head())

# ---------------- VISUALIZATIONS ----------------

# Set global style
sns.set(style="whitegrid")

# Plot 1: Depression Class Count
plt.figure(figsize=(6, 4))
sns.countplot(x='depression', data=df)
plt.title(" Depression Class Distribution")
plt.xlabel("Depression (0 = No, 1 = Yes)")
plt.ylabel("Number of Students")
plt.show()

# Plot 2: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm', linewidths=0.5)
plt.title("🔗 Correlation Heatmap")
plt.show()

# Plot 3: CGPA vs Depression
plt.figure(figsize=(6, 4))
sns.boxplot(x='depression', y='cgpa', data=df)
plt.title("CGPA by Depression Level")
plt.xlabel("Depression")
plt.ylabel("Normalized CGPA")
plt.show()

# Plot 4: Average Sleep vs Depression
plt.figure(figsize=(6, 4))
sns.violinplot(x='depression', y='average_sleep', data=df)
plt.title(" Average Sleep by Depression Level")
plt.xlabel("Depression")
plt.ylabel("Normalized Sleep Duration")
plt.show()

# Plot 5: Stress Relief Activity Usage (Top 5)
activity_counts = activities.sum().sort_values(ascending=False).head(5)
plt.figure(figsize=(8, 4))
sns.barplot(x=activity_counts.values, y=activity_counts.index, palette="viridis")
plt.title(" Top 5 Stress Relief Activities")
plt.xlabel("Number of Students")
plt.ylabel("Activity")
plt.show()

# ------------------ MODELS ---------------------

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = final_df.drop('depression', axis=1)
y = final_df['depression']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# 📌 Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Depression', 'Depressed'])
disp.plot(cmap='Blues')
plt.title(" Confusion Matrix: Depression Prediction")
plt.show()

from sklearn.tree import DecisionTreeClassifier, plot_tree

tree_model = DecisionTreeClassifier(max_depth=4)
tree_model.fit(X_train, y_train)

plt.figure(figsize=(15, 8))
plot_tree(tree_model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree for Depression Prediction")
plt.show()

# ------------------ ANN from Scratch ---------------------

# Convert y to proper shape and one-hot encode for binary classification
y_train_reshaped = y_train.values.reshape(-1, 1)
y_test_reshaped = y_test.values.reshape(-1, 1)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-10
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

# Set architecture
input_size = X_train.shape[1]
hidden_size = 8  # You can adjust
output_size = 1  # Binary classification
learning_rate = 0.1
epochs = 1000

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Convert X_train and X_test to NumPy arrays with float type
X_train = X_train.astype(np.float64).values  # Convert to NumPy array
X_test = X_test.astype(np.float64).values    # Convert to NumPy array

# Training loop
loss_history = []
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Compute loss
    loss = binary_cross_entropy(y_train_reshaped, a2)
    loss_history.append(loss)

    # Backward pass
    dz2 = a2 - y_train_reshaped
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X_train.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Prediction on test data
hidden_output = sigmoid(np.dot(X_test, W1) + b1)
output = sigmoid(np.dot(hidden_output, W2) + b2)
predicted_labels = (output > 0.5).astype(int)

# Accuracy
ann_accuracy = np.mean(predicted_labels.flatten() == y_test_reshaped.flatten())
print(f"\n🔍 ANN Accuracy: {ann_accuracy * 100:.2f}%")

# Enhanced Evaluation: Classification Report
print("\nClassification Report:")
print(classification_report(y_test_reshaped, predicted_labels))

# Loss curve plotting
plt.plot(loss_history)
plt.title("Training Loss Progress")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Actual vs Predicted
print("\nActual vs Predicted (ANN):")
for i in range(len(y_test)):
    print(f"Sample {i+1}: Actual = {y_test_reshaped[i][0]}, Predicted = {predicted_labels[i][0]}")

# 📌 Step 10: Support Vector Machine (SVM) Implementation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ------------------ KERNEL FUNCTIONS ------------------

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def rbf_kernel(x1, x2, gamma=0.5):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

def compute_kernel_matrix(X, kernel='linear', gamma=0.5):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if kernel == 'linear':
                K[i, j] = linear_kernel(X[i], X[j])
            elif kernel == 'rbf':
                K[i, j] = rbf_kernel(X[i], X[j], gamma)
    return K

# ------------------ SVM FROM SCRATCH ------------------

class SVM_Scratch:
    def __init__(self, C=1.0, kernel='linear', gamma=0.5, epochs=100):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.epochs = epochs

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.X = X
        self.y = y.copy()
        self.y[self.y == 0] = -1  # Convert labels to -1 and 1
        self.alpha = np.zeros(n_samples)
        self.b = 0

        # Compute Gram matrix
        K = compute_kernel_matrix(X, self.kernel, self.gamma)

        # Simplified optimization
        for _ in range(self.epochs):
            for i in range(n_samples):
                f_i = np.sum(self.alpha * self.y * K[:, i]) + self.b
                if self.y[i] * f_i < 1:
                    self.alpha[i] += self.C * (1 - self.y[i] * f_i)
                else:
                    self.alpha[i] -= self.C * self.y[i] * f_i

        # Extract support vectors
        self.support_indices = self.alpha > 1e-4
        self.support_alpha = self.alpha[self.support_indices]
        self.support_vectors = X[self.support_indices]
        self.support_y = self.y[self.support_indices]

    def project(self, X):
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            s = 0
            for alpha, sv_y, sv in zip(self.support_alpha, self.support_y, self.support_vectors):
                if self.kernel == 'linear':
                    k = linear_kernel(X[i], sv)
                elif self.kernel == 'rbf':
                    k = rbf_kernel(X[i], sv, self.gamma)
                s += alpha * sv_y * k
            result[i] = s
        return result + self.b

    def predict(self, X):
        return (self.project(X) > 0).astype(int)

# ------------------ TRAIN & EVALUATE ------------------

# Train and evaluate Linear SVM
print("\n📌 SVM from Scratch - Linear Kernel")
svm_linear = SVM_Scratch(kernel='linear', C=0.01, epochs=200)
svm_linear.fit(X_train, y_train.values)
pred_linear = svm_linear.predict(X_test)

acc_linear = accuracy_score(y_test, pred_linear)
print(f"🎯 Accuracy (Linear): {acc_linear * 100:.2f}%")
print(classification_report(y_test, pred_linear))
ConfusionMatrixDisplay.from_predictions(y_test, pred_linear, display_labels=['No Depression', 'Depressed'], cmap='Blues')
plt.title("Confusion Matrix - SVM Scratch (Linear)")
plt.show()

# Train and evaluate RBF SVM
print("\n📌 SVM from Scratch - RBF Kernel")
svm_rbf = SVM_Scratch(kernel='rbf', gamma=0.5, C=0.01, epochs=200)
svm_rbf.fit(X_train, y_train.values)
pred_rbf = svm_rbf.predict(X_test)

acc_rbf = accuracy_score(y_test, pred_rbf)
print(f"🎯 Accuracy (RBF): {acc_rbf * 100:.2f}%")
print(classification_report(y_test, pred_rbf))
ConfusionMatrixDisplay.from_predictions(y_test, pred_rbf, display_labels=['No Depression', 'Depressed'], cmap='Greens')
plt.title("Confusion Matrix - SVM Scratch (RBF)")
plt.show()

# ------------------ FINAL COMPARISON ------------------

print("\n====================== SVM Kernel Comparison ======================")
print(f"SVM (Linear Kernel) Accuracy: {acc_linear * 100:.2f}%")
print(f"SVM (RBF Kernel) Accuracy: {acc_rbf * 100:.2f}%")

if acc_linear > acc_rbf:
    print(f"\n✅ Final Conclusion: Linear SVM performed better with {acc_linear * 100:.2f}% accuracy.")
elif acc_rbf > acc_linear:
    print(f"\n✅ Final Conclusion: RBF SVM performed better with {acc_rbf * 100:.2f}% accuracy.")
else:
    print("\n✅ Final Conclusion: Both kernels performed equally.")

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

print("\n🔍 Columns available in final_df:")
print(final_df.columns.tolist())

# ======================= KNN FROM SCRATCH =======================
print("\n🔍 Running KNN from Scratch...")

X_train_knn = X_train.astype(np.float64)
X_test_knn = X_test.astype(np.float64)
y_train_knn = y_train
y_test_knn = y_test

# --- Euclidean Distance ---
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))  # 📌 Explicit distance metric used: Euclidean

# --- KNN Prediction Function ---
def knn_predict(X_train, y_train, x_test, k=3):
    distances = [(euclidean_distance(x, x_test), label) for x, label in zip(X_train, y_train)]
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    votes = [label for _, label in neighbors]
    return max(set(votes), key=votes.count)

# --- Experiment with different k values ---
print("🔁 KNN Results for different values of k:\n")
for k_val in [1, 3, 5, 7]:
    knn_predictions = [knn_predict(X_train_knn, y_train_knn, x, k=k_val) for x in X_test_knn]
    accuracy = accuracy_score(y_test_knn, knn_predictions)
    print(f"✅ k = {k_val} | Accuracy = {accuracy:.4f}")
    print(classification_report(y_test_knn, knn_predictions, target_names=['No Depression', 'Depressed']))

# --- Final Confusion Matrix for k=3 (best default) ---
final_k = 3
final_predictions = [knn_predict(X_train_knn, y_train_knn, x, k=final_k) for x in X_test_knn]

cm_knn = confusion_matrix(y_test_knn, final_predictions)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=['No Depression', 'Depressed'])
disp_knn.plot(cmap='Purples')
plt.title(f"Confusion Matrix: KNN (k={final_k}) from Scratch")
plt.show()

# =================== K-MEANS CLUSTERING FROM SCRATCH ===================
print("\n🔁 Running K-Means Clustering from Scratch...")

class KMeans:
    def __init__(self, n_clusters, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iters):
            labels = self._assign_labels(X)
            new_centroids = self._update_centroids(X, labels)
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        self.labels_ = labels

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        new_centroids = []
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroid = cluster_points.mean(axis=0)
            else:
                new_centroid = X[np.random.randint(0, X.shape[0])]
            new_centroids.append(new_centroid)
        return np.array(new_centroids)

# --- Prepare Data ---
X_kmeans = final_df.drop('depression', axis=1).values.astype(float)
y_kmeans = final_df['depression'].values

# --- Determine k (number of clusters) ---
print("\n🔢 Assuming binary classification, setting K=2 for depression classification.")

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_kmeans)
kmeans_clusters = kmeans.labels_

# --- Map clusters to true labels ---
def map_clusters_to_labels(clusters, true_labels):
    label_map = {}
    for cluster in np.unique(clusters):
        indices = np.where(clusters == cluster)[0]
        assigned_labels = true_labels[indices]
        most_common = Counter(assigned_labels).most_common(1)[0][0]
        label_map[cluster] = most_common
    return np.array([label_map[c] for c in clusters])

kmeans_predictions = map_clusters_to_labels(kmeans_clusters, y_kmeans)

# --- Evaluate on test portion ---
test_indices = y_test.index
kmeans_test_predictions = kmeans_predictions[test_indices]

kmeans_accuracy = accuracy_score(y_test, kmeans_test_predictions)
print(f"✅ K-Means Accuracy (from scratch): {kmeans_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, kmeans_test_predictions, target_names=['No Depression', 'Depressed']))

# --- Confusion Matrix ---
cm_kmeans = confusion_matrix(y_test, kmeans_test_predictions)
disp_kmeans = ConfusionMatrixDisplay(confusion_matrix=cm_kmeans, display_labels=['No Depression', 'Depressed'])
disp_kmeans.plot(cmap='Greens')
plt.title("Confusion Matrix: K-Means from Scratch")
plt.show()

# ===================== KNN & K-MEANS VISUALIZATION =====================

# Using 'anxiety' and 'isolation' as features for visualization
feature1 = 'anxiety'
feature2 = 'isolation'

print(f"\n⚠️ Features '{feature1}' and '{feature2}' used for visualization.")

X_vis = final_df[[feature1, feature2]].values.astype(float)

plt.figure(figsize=(14, 6))

# --- K-Means Clustering Visualization ---
plt.subplot(1, 2, 1)
for cluster_id in np.unique(kmeans_clusters):
    cluster_mask = (kmeans_clusters == cluster_id)
    plt.scatter(X_vis[cluster_mask, 0], X_vis[cluster_mask, 1], label=f'Cluster {cluster_id}', alpha=0.6)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=200, marker='X', c='black', label='Centroids')
plt.title("K-Means Clustering")
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend()
plt.grid(True)

# --- KNN Decision Boundary Visualization ---
plt.subplot(1, 2, 2)
x_min, x_max = X_vis[:, 0].min() - 0.1, X_vis[:, 0].max() + 0.1
y_min, y_max = X_vis[:, 1].min() - 0.1, X_vis[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

# Use only 2D features for prediction (matching visualization features)
mesh_preds = [knn_predict(X_train_knn[:, [final_df.columns.get_loc(feature1), final_df.columns.get_loc(feature2)]], 
                         y_train_knn, np.array([x, y]), k=3)
              for x, y in zip(xx.ravel(), yy.ravel())]
Z = np.array(mesh_preds).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_test_knn[:, final_df.columns.get_loc(feature1)], X_test_knn[:, final_df.columns.get_loc(feature2)], 
            c=y_test, cmap='coolwarm', edgecolor='k', s=50)
plt.title("KNN Decision Regions")
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.grid(True)

plt.tight_layout()
plt.show()



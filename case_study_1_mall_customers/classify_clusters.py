import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- Reload and recreate the clustering (so this file can run independently) ---
df = pd.read_csv("mall_customers.csv")

def get_age_group(age):
    if age <= 25:
        return "18-25"
    elif age <= 35:
        return "26-35"
    elif age <= 50:
        return "36-50"
    else:
        return "51+"

df['Age Group'] = df['Age'].apply(get_age_group)

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Estimated Savings (k$)', 'Credit Score']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# --- Now the actual classification task ---
# Features stay the same; target is now the Cluster label
y = df['Cluster']

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- Naive Bayes ---
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_preds)

print("=== Naive Bayes Classifier ===")
print(f"Accuracy: {nb_accuracy:.3f}")
print(classification_report(y_test, nb_preds))

# --- Decision Tree ---
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_preds)

print("\n=== Decision Tree Classifier ===")
print(f"Accuracy: {dt_accuracy:.3f}")
print(classification_report(y_test, dt_preds))

# --- Save the better model + scaler for reuse ---
best_model = nb_model if nb_accuracy >= dt_accuracy else dt_model
best_model_name = "Naive Bayes" if nb_accuracy >= dt_accuracy else "Decision Tree"

joblib.dump(best_model, "cluster_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")
print(f"\nBest model ({best_model_name}) saved as cluster_classifier.pkl")




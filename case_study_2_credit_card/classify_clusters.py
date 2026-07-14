import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# reloading and recreating the clustering pipeline so this file can run on its own
df = pd.read_csv("CC GENERAL.csv")

df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())
df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())

features = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
            'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
            'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
            'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.90, random_state=42)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_pca)




# now predicting the cluster label from the original scaled features
y = df['Cluster']

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)



# naive bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_preds)

print("=== Naive Bayes Classifier ===")
print(f"Accuracy: {nb_accuracy:.3f}")
print(classification_report(y_test, nb_preds))



# decision tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_preds)

print("\n=== Decision Tree Classifier ===")
print(f"Accuracy: {dt_accuracy:.3f}")
print(classification_report(y_test, dt_preds))



# saving whichever model performed better, along with the scaler
best_model = nb_model if nb_accuracy >= dt_accuracy else dt_model
best_model_name = "Naive Bayes" if nb_accuracy >= dt_accuracy else "Decision Tree"

joblib.dump(best_model, "cluster_classifier.pkl")
joblib.dump(scaler, "scaler.pkl")
print(f"\nBest model ({best_model_name}) saved as cluster_classifier.pkl")
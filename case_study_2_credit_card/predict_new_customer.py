import joblib
import numpy as np

model = joblib.load("cluster_classifier.pkl")
scaler = joblib.load("scaler.pkl")

cluster_names = {
    0: "High-Value Active User",
    1: "Standard/Light User"
}

feature_names = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
                  'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
                  'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
                  'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']

def predict_cluster(values):
    input_data = np.array([values])
    input_scaled = scaler.transform(input_data)
    cluster = model.predict(input_scaled)[0]
    return cluster, cluster_names.get(cluster, "Unknown")

if __name__ == "__main__":
    print("💳 Credit Card Customer Segment Predictor")
    print("Enter a new customer's details:\n")

    while True:
        try:
            values = []
            for fname in feature_names:
                val = float(input(f"{fname}: "))
                values.append(val)

            cluster_num, cluster_name = predict_cluster(values)
            print(f"\n➡️ Predicted Segment: Cluster {cluster_num} — {cluster_name}\n")

        except ValueError:
            print("Please enter valid numbers.\n")

        again = input("Try another customer? (y/n): ")
        if again.lower() != 'yes':
            break

    print("Done!")
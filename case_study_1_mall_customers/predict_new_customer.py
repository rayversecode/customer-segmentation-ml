import joblib
import numpy as np



# load the trained model and scaler
model = joblib.load("cluster_classifier.pkl")
scaler = joblib.load("scaler.pkl")



# cluster interpretations based on the earlier analysis
cluster_names = {
    0: "Mature Moderate Spenders",
    1: "Young Average Earners",
    2: "Young High Spenders (Low Credit)",
    3: "Premium VIP Customers",
    4: "Wealthy Savers"
}

def predict_cluster(age, income, spending_score, savings, credit_score):
    input_data = np.array([[age, income, spending_score, savings, credit_score]])
    input_scaled = scaler.transform(input_data)
    cluster = model.predict(input_scaled)[0]
    return cluster, cluster_names.get(cluster, "Unknown")


if __name__ == "__main__":
    print("Customer Segment Predictor")
    print("Enter a new customer's details to find their segment.\n")

    while True:
        try:
            age = float(input("Age: "))
            income = float(input("Annual Income (k$): "))
            spending = float(input("Spending Score (1-100): "))
            savings = float(input("Estimated Savings (k$): "))
            credit = float(input("Credit Score: "))

            cluster_num, cluster_name = predict_cluster(age, income, spending, savings, credit)
            print(f"\nPredicted Segment: Cluster {cluster_num} - {cluster_name}\n")

        except ValueError:
            print("Please enter valid numbers.\n")

        again = input("Try another customer? (y/n): ")
        if again.lower() != 'y':
            break

    print("Done!")
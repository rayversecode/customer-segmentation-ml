# Customer Segmentation using Machine Learning

This project applies unsupervised and supervised machine learning techniques to segment customers into meaningful groups, demonstrated across two different real-world domains: retail and banking.

## Project Structure

- **case_study_1_mall_customers/** — Segments mall customers based on demographics and spending behavior
- **case_study_2_credit_card/** — Segments credit card customers based on financial usage behavior

## Methodology (applied in both case studies)

1. **Data Exploration** — cleaning, handling missing values, understanding feature distributions
2. **Feature Scaling** — standardizing features for fair distance-based clustering
3. **K-Means Clustering** — with the Elbow Method to determine optimal cluster count
4. **Cluster Evaluation** — Silhouette Score used to quantitatively validate cluster quality across multiple k values
5. **Hierarchical Clustering** — used as a comparison algorithm (Case Study 1)
6. **PCA (Principal Component Analysis)** — used to reduce dimensionality and validate clustering results (Case Study 2)
7. **Classification** — Naive Bayes and Decision Tree classifiers trained to predict cluster membership for new customers
8. **Interactive Prediction Tool** — command-line tool to classify a new customer into a segment instantly

## Key Findings

- **Case Study 1 (Mall Customers):** Data revealed 5 distinct, nuanced customer segments (e.g., Premium VIPs, Wealthy Savers, Young High Spenders), with a silhouette score of 0.409.
- **Case Study 2 (Credit Card Customers):** Data revealed 2 dominant behavioral groups (High-Value Active Users vs Standard/Light Users), with a silhouette score of 0.381 — a result confirmed even after PCA, showing this is a genuine property of the data rather than an artifact of high dimensionality.

This comparison highlights an important principle in applied machine learning: cluster structure depends on the nature of the data itself, and a rigorous approach (testing multiple k values, validating with multiple metrics and algorithms) is necessary rather than assuming a fixed number of segments will always emerge.

## Tech Stack
Python, pandas, scikit-learn, matplotlib, scipy

## How to Run
Each case study folder is self-contained. Navigate into either folder and run:
```
python explore_data.py       # data exploration
python clustering.py         # clustering + evaluation
python classify_clusters.py  # train classifiers
python predict_new_customer.py  # interactive prediction tool
```
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

### Algorithms Compared
Three clustering algorithms were applied and evaluated across both case studies:
1. **K-Means** — partition-based, requires specifying cluster count upfront
2. **Hierarchical (Agglomerative) Clustering** — builds a tree of nested clusters
3. **DBSCAN** — density-based, automatically detects outliers, no cluster count needed

### Case Study 1: Mall Customers (Retail)
- **K-Means (k=5):** Silhouette score 0.409 — revealed 5 distinct, business-relevant segments (e.g., Premium VIPs, Wealthy Savers, Young High Spenders with low credit)
- **Hierarchical Clustering (k=5):** Silhouette score 0.388 — confirmed similar structure to K-Means
- **DBSCAN:** Found 8 tight micro-clusters covering 42.5% of customers with a high silhouette score of 0.601 on that subset, but classified 57.5% of customers as noise — excellent at finding pockets of very similar customers, but impractical for segmenting the full customer base

**Conclusion:** K-Means offers the best balance of interpretability and full coverage for business use, though DBSCAN reveals interesting tight sub-groups worth investigating separately (e.g., for highly targeted micro-campaigns).

### Case Study 2: Credit Card Customers (Banking)
- **K-Means (k=2):** Silhouette score 0.381 — revealed 2 dominant behavioral groups (High-Value Active Users vs Standard/Light Users), confirmed even after PCA dimensionality reduction
- **DBSCAN:** Silhouette score only 0.143, with 32.6% of customers classified as noise and only one dominant cluster identified — density-based clustering struggled to find meaningful structure in this dataset

**Conclusion:** This dataset's natural structure is a broad 2-group split rather than several nuanced segments. Both PCA and DBSCAN results reinforce that this is a genuine property of the data, not a limitation of any single algorithm.

### Overall Takeaway
This project demonstrates that clustering results are highly dependent on the nature of the underlying data:
- **Demographic/behavioral retail data** (mall customers) naturally separates into multiple nuanced groups, and all three algorithms broadly agree, with DBSCAN excelling at finding tight micro-segments.
- **High-dimensional financial usage data** (credit card customers) has a simpler, dominant 2-group structure, and density-based methods like DBSCAN are less effective here.

Rather than assuming one "correct" number of clusters or one best algorithm, this project applied rigorous validation (Elbow Method, Silhouette Score across multiple k values, cross-algorithm comparison, and PCA verification) to let the data determine the right approach for each domain — a key principle of applied, honest data science.

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
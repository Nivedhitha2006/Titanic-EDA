
# 🛳️ Titanic Dataset - Exploratory Data Analysis (EDA)

This project performs **Exploratory Data Analysis (EDA)** on the classic **Titanic dataset**, using Python libraries such as **Pandas, NumPy, Matplotlib, and Seaborn**.  
Visualizations and statistical insights are included to understand the patterns related to survival, passenger characteristics, correlations, and outliers.

---

## 📌 **Objectives of the Project**

This EDA aims to:

- Understand dataset structure (nulls, datatypes, distributions)
- Generate summary statistics (mean, median, std, percentiles)
- Visualize numerical features using histograms & boxplots
- Explore relationships using pairplots and correlation matrices
- Identify patterns, trends, and anomalies
- Infer survival-related insights from the data

---

## 📂 **Dataset Source**

The dataset is fetched **directly from Seaborn**:

```python
df = sns.load_dataset("titanic")

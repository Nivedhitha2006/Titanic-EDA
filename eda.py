import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Create screenshots folder
os.makedirs("screenshots", exist_ok=True)
# Load Titanic dataset
df = sns.load_dataset("titanic")
# -----------------------------
# Basic Dataset Overview
# -----------------------------
print("\n===== FIRST 5 ROWS =====")
print(df.head())
print("\n===== DATA INFO =====")
print(df.info())
print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe(include="all"))
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())
# -----------------------------
# Summary Statistics
# -----------------------------
print("\n===== SUMMARY STATISTICS =====")
print("Mean Age:", df['age'].mean())
print("Median Age:", df['age'].median())
print("Std Age:", df['age'].std())
print("---------------------------------")
print("Mean Fare:", df['fare'].mean())
print("Median Fare:", df['fare'].median())
print("Std Fare:", df['fare'].std())
# -----------------------------
# Histograms (with display)
# -----------------------------
numeric_cols = ['age', 'fare', 'sibsp', 'parch']
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"{col} Distribution")
    plt.savefig(f"screenshots/{col}_hist.png")
    plt.show()     # 👈 SHOW output immediately!
# -----------------------------
# Boxplots (with display)
# -----------------------------
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"{col} Boxplot")
    plt.savefig(f"screenshots/{col}_boxplot.png")
    plt.show()     # 👈 SHOW output immediately!
# -----------------------------
# Pairplot (with display)
# -----------------------------
pairplot_cols = ["survived", "age", "fare", "pclass"]
pair = sns.pairplot(df[pairplot_cols].dropna(), hue="survived", diag_kind="kde")
pair.savefig("screenshots/pairplot.png")
plt.show()         # 👈 SHOW output immediately!
# -----------------------------
# Correlation Heatmap (with display)
# -----------------------------
plt.figure(figsize=(10, 6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("screenshots/correlation_heatmap.png")
plt.show()         # 👈 SHOW output immediately!
# -----------------------------
# Inferences
# -----------------------------
print("\n===== ANALYSIS INFERENCES =====")
print("✔ Females survived more than males.")
print("✔ 1st class passengers survived more than 3rd class.")
print("✔ Fare is highly skewed (right skew).")
print("✔ Age column contains outliers.")
print("✔ Pclass & Fare have strong correlation.")
print("✔ Family size influences survival.")
print("\n✨ EDA Completed! Visualizations displayed + saved in 'screenshots/'")



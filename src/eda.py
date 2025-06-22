import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure artifacts directory exists
os.makedirs("artifacts", exist_ok=True)

# Load Data
df = pd.read_csv(r'C:\Users\IfeomaAugustaAdigwe\Desktop\creditscore\data\clean_data.csv')
print(df.head())

# Shape of the dataset
print("Dataset Shape:", df.shape)

# Basic Info
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

# Univariate Analysis (Distribution of Each Feature)
sns.set_style("whitegrid")
features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, feature in enumerate(features):
    sns.histplot(df[feature], bins=30, kde=True, ax=axes[i], color='royalblue')
    axes[i].set_title(f'Distribution of {feature}', fontsize=14)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig("artifacts/univariate_distribution.png")
plt.close()

# Correlation matrix
numerical_columns = df.select_dtypes(include=['int', 'float'])
corr_matrix = numerical_columns.corr()
plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.savefig("artifacts/correlation_heatmap.png")
plt.close()

# Relationship Between Income and Loan Amount
if 'Num_of_Loan' in df.columns:
    y_col = 'Num_of_Loan'
elif 'Loan_Amount' in df.columns:
    y_col = 'Loan_Amount'
else:
    y_col = df.columns[df.columns.str.contains('Loan', case=False)][0]
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Annual_Income'], y=df[y_col], alpha=0.7, color='green')
plt.title('Income vs. Number of Loans', fontsize=14)
plt.xlabel('Annual Income')
plt.ylabel('Number of Loans')
plt.grid()
plt.savefig("artifacts/income_vs_num_loans.png")
plt.close()

sns.scatterplot(x=df['Num_Credit_Inquiries'], y=df['Num_of_Delayed_Payment'], color='blue')
plt.title("Credit Inquiries vs. Delayed Payments", fontsize=14)
plt.xlabel("Number of Credit Inquiries")
plt.ylabel("Delayed Payments")
plt.savefig("artifacts/credit_inquiries_vs_delayed_payments.png")
plt.close()

# Top 10 loan type distribution
count_loan_type = df['Type_of_Loan'].value_counts()[:10]
print(count_loan_type)
plt.figure(figsize=(10, 5))
sns.barplot(x=count_loan_type.values, y=count_loan_type.index, palette="Blues_r")
plt.title("Top 10 Loan Types", fontsize=14)
plt.xlabel("Count")
plt.ylabel("Loan Type")
plt.savefig("artifacts/top_10_loan_types.png")
plt.close()

# Outlier Detection
features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt']
plt.figure(figsize=(14, 8))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[feature], color="orange")
    plt.title(f"Boxplot of {feature}", fontsize=12)
plt.tight_layout()
plt.savefig("artifacts/outlier_boxplots.png")
plt.close()

# Hypothesis Testing
from scipy.stats import ttest_ind, chi2_contingency, f_oneway, kruskal

# Hypothesis 1: Does Annual Income Differ Based on Loan Default?
default_group = df[df['Num_of_Delayed_Payment'] > 0]['Annual_Income']
non_default_group = df[df['Num_of_Delayed_Payment'] == 0]['Annual_Income']
t_stat, p_value = ttest_ind(default_group, non_default_group, nan_policy='omit')
print(f"T-Statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
if p_value < 0.05:
    print("Reject Null Hypothesis: Annual Income differs between defaulters and non-defaulters.")
else:
    print("Fail to Reject Null Hypothesis: No significant difference in Annual Income.")

# Hypothesis 2: Is There a Relationship Between Age and Loan Defaults?
contingency_table = pd.crosstab(df['Age'], df['Num_of_Delayed_Payment'])
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2_stat:.3f}, P-value: {p_value:.3f}")
if p_value < 0.05:
    print("Reject Null Hypothesis: Age and Loan Defaults are dependent.")
else:
    print("Fail to Reject Null Hypothesis: No strong relationship between Age and Loan Defaults.")

# Income vs. Categorical Credit Score Analysis
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Credit_Score'], y=df['Annual_Income'], palette='coolwarm')
plt.title('Income Distribution by Credit Score Category')
plt.xlabel('Credit Score Category')
plt.ylabel('Annual Income')
plt.savefig("artifacts/income_by_credit_score.png")
plt.close()

# Proportion of Credit Score Categories in Each Income Bracket
df['Income_Bracket'] = pd.cut(df['Annual_Income'], bins=[0, 30000, 60000, 100000, float('inf')],
                              labels=['Low', 'Mid', 'Upper-Mid', 'High'])
plt.figure(figsize=(8, 6))
sns.countplot(x=df['Income_Bracket'], hue=df['Credit_Score'], palette='coolwarm')
plt.title('Credit Score Distribution by Income Bracket')
plt.xlabel('Income Bracket')
plt.ylabel('Count')
plt.legend(title="Credit Score")
plt.savefig("artifacts/credit_score_by_income_bracket.png")
plt.close()

print(df['Credit_Score'].unique())

# ANOVA and Kruskal-Wallis tests
good_income = df[df['Credit_Score'] == 'Good']['Annual_Income']
standard_income = df[df['Credit_Score'] == 'Standard']['Annual_Income']
poor_income = df[df['Credit_Score'] == 'Poor']['Annual_Income']
anova_result = f_oneway(good_income, standard_income, poor_income)
print(f"ANOVA Test Result: p-value = {anova_result.pvalue}")
kruskal_result = kruskal(good_income, standard_income, poor_income)
print(f"Kruskal-Wallis Test Result: p-value = {kruskal_result.pvalue}")

# Feature Engineering for Credit Scoring
df['Debt_to_Income_Ratio'] = df['Outstanding_Debt'] / df['Annual_Income']
df['Missed_Payment_Frequency'] = df['Num_of_Delayed_Payment'] / (df['Credit_History_Age'] * 12)
df['Loan-to-Income Ratio'] = df['Total_EMI_per_month'] / df['Monthly_Inhand_Salary']
df['Credit Usage Efficiency'] = df['Credit_Utilization_Ratio'] * df['Num_Credit_Card']
print(df.head())

# Descriptive statistics for engineered features
print(df[['Debt_to_Income_Ratio', 'Missed_Payment_Frequency', 'Credit_Utilization_Ratio', 'Loan-to-Income Ratio', 'Credit Usage Efficiency']].describe())

# Histogram Distribution of Engineered Features
features = ['Debt_to_Income_Ratio', 'Missed_Payment_Frequency', 'Credit_Utilization_Ratio']
plt.figure(figsize=(12, 4))
for i, col in enumerate(features, 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[col], bins=30, kde=True, color='orange')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig("artifacts/engineered_features_hist.png")
plt.close()

# Scatter Plots to Identify Risky Patterns
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=df['Debt_to_Income_Ratio'], y=df['Credit_Utilization_Ratio'], alpha=0.5)
plt.title("Debt-to-Income vs. Credit Utilization")
plt.subplot(1, 2, 2)
sns.scatterplot(x=df['Debt_to_Income_Ratio'], y=df['Missed_Payment_Frequency'], alpha=0.5)
plt.title("Debt-to-Income vs. Missed Payments")
plt.tight_layout()
plt.savefig("artifacts/risky_patterns_scatter.png")
plt.close()

# Distribution of Financial Indicators by Credit Score
features = ['Debt_to_Income_Ratio', 'Credit_Utilization_Ratio', 'Missed_Payment_Frequency', 'Annual_Income', 'Credit_History_Age', 'Num_Credit_Inquiries']
plt.figure(figsize=(12, 8))
for i, col in enumerate(features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df['Credit_Score'], y=df[col], palette="Set2")
    plt.title(f'{col} vs Credit Score')
plt.tight_layout()
plt.savefig("artifacts/financial_indicators_by_credit_score.png")
plt.close()

# Correlation Analysis of Financial Indicators
numerical_features = ['Debt_to_Income_Ratio', 'Credit_Utilization_Ratio', 'Missed_Payment_Frequency', 'Annual_Income', 'Credit_History_Age', 'Num_Credit_Inquiries']
corr_matrix = df[numerical_features].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.savefig("artifacts/financial_indicators_corr_heatmap.png")
plt.close()

# Income vs Credit Utilization (Financial Stability)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Annual_Income'], y=df['Credit_Utilization_Ratio'], hue=df['Credit_Score'], alpha=0.6)
plt.title("Annual Income vs Credit Utilization by Credit Score")
plt.savefig("artifacts/income_vs_credit_utilization.png")
plt.close()

# Risk Segmentation using Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = df[['Debt_to_Income_Ratio', 'Credit_Utilization_Ratio', 'Missed_Payment_Frequency', 'Annual_Income']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Risk_Cluster'] = kmeans.fit_predict(X_scaled)

columns = ['Debt_to_Income_Ratio', 'Credit_Utilization_Ratio', 'Missed_Payment_Frequency', 'Annual_Income']
fig, ax = plt.subplots(2, 2, figsize=(10, 6))
for name, axis in zip(columns, ax.ravel()):
    sns.kdeplot(data=df, hue='Risk_Cluster', x=name, fill=True, ax=axis)
plt.tight_layout()
plt.savefig("artifacts/risk_cluster_kde.png")
plt.close()

# Categorize Risk Cluster
risk_mapping = {0: "Safe", 1: "Moderate Risk", 2: "High Risk"}
df["Risk_Cluster"] = df["Risk_Cluster"].map(risk_mapping)
print(df["Risk_Cluster"].unique())

# Aggregate total annual income per risk category
risk_income = df.groupby("Risk_Cluster")["Annual_Income"].sum()
plt.figure(figsize=(8, 6))
plt.pie(risk_income, labels=risk_income.index, autopct='%1.1f%%', startangle=140, colors=['green', 'orange', 'red'])
plt.title("Contribution of Risk Groups to Total Annual Income")
plt.savefig("artifacts/risk_group_income_pie.png")
plt.close()

# Income distribution by credit score
fig, ax = plt.subplots(figsize=(12, 6))
grouped = df.groupby(['Credit_Score', 'Income_Bracket'])['Annual_Income'].count().unstack(fill_value=0)
grouped.plot(kind='bar', ax=ax, cmap='viridis')
ax.set_title('Income Distribution by Credit Score')
ax.set_xlabel('Credit Score')
ax.set_ylabel('Count')
ax.legend(title='Income')
plt.xticks(rotation=45)
plt.subplots_adjust(bottom=0.2)
plt.savefig("artifacts/income_distribution_by_credit_score.png")
plt.close()

# Loan-to-Income Ratio vs Credit Utilization Ratio
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(x='Loan-to-Income Ratio', y='Credit_Utilization_Ratio', data=df, ax=ax)
ax.set_title('Relationship between Loan-to-Income Ratio and Credit Utilization Ratio')
ax.set_xlabel('Loan-to-Income Ratio')
ax.set_ylabel('Credit Utilization Ratio')
plt.savefig("artifacts/loan_income_vs_credit_utilization.png")
plt.close()

# Credit Age vs Credit Score
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Credit_History_Age', y='Credit_Score', data=df, ax=ax)
ax.set_title('Relationship between Credit Age and Credit Score')
ax.set_xlabel('Credit Age (months)')
ax.set_ylabel('Credit Score')
plt.savefig("artifacts/credit_age_vs_credit_score.png")
plt.close()

# Annual Income vs Missed Payment Frequency
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(x='Annual_Income', y='Missed_Payment_Frequency', data=df, ax=ax)
ax.set_title('Relationship between Annual Income and Missed Payment Frequency')
ax.set_xlabel('Annual Income')
ax.set_ylabel('Missed Payment Frequency')
plt.savefig("artifacts/income_vs_missed_payment_frequency.png")
plt.close()

# Loan-to-Income Ratio vs Credit Usage Efficiency
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(x='Loan-to-Income Ratio', y='Credit Usage Efficiency', data=df, ax=ax)
ax.set_title('Relationship between Loan-to-Income Ratio and Credit Usage Efficiency')
ax.set_xlabel('Loan-to-Income Ratio')
ax.set_ylabel('Credit Usage Efficiency')
plt.savefig("artifacts/loan_income_vs_credit_usage_efficiency.png")
plt.close()

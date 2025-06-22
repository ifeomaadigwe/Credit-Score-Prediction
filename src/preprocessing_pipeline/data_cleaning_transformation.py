import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

def import_data(file_path):
    """
    Import data from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def visualize_missing_values(df, output_path):
    plt.figure(figsize=(20, 8))
    sns.heatmap(df.isnull(), cbar=True, cmap='coolwarm')
    plt.title('Visualizing Missing Values')
    plt.savefig(output_path)
    plt.close()

def remove_special_characters(df):
    return df.replace(to_replace=r'[^\w\s.-]', value='', regex=True)

def convert_credit_history_age(text):
    match = re.search(r'(\d+)\sYears?\sand\s(\d+)\sMonths?', str(text))
    if match:
        return int(match.group(1)) * 12 + int(match.group(2))
    return np.nan

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].copy()
    df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = df[column].median()
    return df

def clean_credit_score_data(input_path, output_path, artifacts_dir):
    # Import data
    df = import_data(input_path)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Visualize missing values
    os.makedirs(artifacts_dir, exist_ok=True)
    visualize_missing_values(df, os.path.join(artifacts_dir, 'missing_values_heatmap.png'))

    # Remove leading/trailing spaces
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Remove special characters
    df = remove_special_characters(df)

    # Fix Data Types and handle nulls
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(0).astype(int)
    df['Annual_Income'] = pd.to_numeric(df['Annual_Income'], errors='coerce')
    df['Num_of_Loan'] = pd.to_numeric(df['Num_of_Loan'], errors='coerce').fillna(0).astype(int)
    df['Num_of_Delayed_Payment'] = pd.to_numeric(df['Num_of_Delayed_Payment'], errors='coerce').fillna(0).astype(int)
    df['Num_Credit_Inquiries'] = pd.to_numeric(df['Num_Credit_Inquiries'], errors='coerce').fillna(0).astype(int)
    df['Outstanding_Debt'] = pd.to_numeric(df['Outstanding_Debt'], errors='coerce')
    df['Outstanding_Debt'].fillna(df['Outstanding_Debt'].median(), inplace=True)
    df['Changed_Credit_Limit'] = pd.to_numeric(df['Changed_Credit_Limit'].replace('_', np.nan), errors='coerce')
    df['Changed_Credit_Limit'].fillna(df['Changed_Credit_Limit'].median(), inplace=True)
    df['Amount_invested_monthly'] = pd.to_numeric(df['Amount_invested_monthly'], errors='coerce')
    df.loc[df['Amount_invested_monthly'] > 1e6, 'Amount_invested_monthly'] = np.nan
    df['Amount_invested_monthly'].fillna(df['Amount_invested_monthly'].median(), inplace=True)
    df['Monthly_Balance'] = pd.to_numeric(df['Monthly_Balance'], errors='coerce')
    df['Monthly_Balance'].fillna(df['Monthly_Balance'].median(), inplace=True)
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', np.nan).fillna('Unknown')
    df['Annual_Income'] = df['Annual_Income'].fillna(df['Annual_Income'].median())
    df['Monthly_Inhand_Salary'] = df['Monthly_Inhand_Salary'].fillna(df['Monthly_Inhand_Salary'].median())

    # Convert Credit History Age to Months
    df['Credit_History_Age'] = df['Credit_History_Age'].astype(str).apply(convert_credit_history_age)

    # Replace inconsistent values
    if 'Payment_Behaviour' in df.columns:
        mode_payment_behaviour = df['Payment_Behaviour'].mode()[0]
        df['Payment_Behaviour'] = df['Payment_Behaviour'].str.replace('98', mode_payment_behaviour)
    if 'Name' in df.columns:
        df['Name'] = df['Name'].fillna('Unknown')
    if 'Credit_Mix' in df.columns:
        df['Credit_Mix'] = df['Credit_Mix'].replace('_', 'Unknown')
    if 'Occupation' in df.columns:
        df['Occupation'] = df['Occupation'].str.replace('_______', 'Unknown')

    # Fill NaN in Credit_History_Age
    median_value = df['Credit_History_Age'].median()
    df['Credit_History_Age'].fillna(median_value, inplace=True)

    # Replace negative values in selected columns
    cols_to_fix = ['Age', 'Num_Bank_Accounts', 'Num_of_Loan']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] < 0, col] = np.nan
            df[col] = df[col].fillna(df[col].median()).astype(int)

    # Handle outliers
    for col in ['Age', 'Annual_Income', 'Monthly_Inhand_Salary']:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)

    # Replace nulls in Type_of_Loan and Credit_History_Age
    if 'Type_of_Loan' in df.columns:
        loan_type_mode_value = df['Type_of_Loan'].mode()[0]
        df['Type_of_Loan'] = df['Type_of_Loan'].fillna(loan_type_mode_value).astype(str)
    if 'Credit_History_Age' in df.columns:
        credit_history_age_median = df['Credit_History_Age'].median()
        df['Credit_History_Age'] = df['Credit_History_Age'].fillna(credit_history_age_median).astype(int)

    # Save cleaned data
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    # Correct file paths
    input_path = os.path.join('data', r'C:\Users\IfeomaAugustaAdigwe\Desktop\creditscore\data\credit_score.csv')
    output_path = os.path.join('data', r'C:\Users\IfeomaAugustaAdigwe\Desktop\creditscore\data\clean_data.csv')
    artifacts_dir = 'artifacts'
    clean_credit_score_data(input_path, output_path, artifacts_dir)
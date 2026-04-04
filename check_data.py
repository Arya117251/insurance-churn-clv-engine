import pandas as pd
import os

# Load the raw customer file
customer_path = r'E:\data_warehouse\insurance_churn\data\raw\customer.csv'
df = pd.read_csv(customer_path)

print("Columns before:", list(df.columns))

# Drop SSN immediately — PII, never belongs in a model or repo
df = df.drop(columns=['SOCIAL_SECURITY_NUMBER'])

print("Columns after:", list(df.columns))

# Overwrite the original file
df.to_csv(customer_path, index=False)
print("\nDone. SSN column permanently removed from customer.csv")
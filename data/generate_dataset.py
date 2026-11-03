"""
BrightMart Retail Dataset Generator
Creates a realistic retail dataset with natural business patterns and intentional
data quality quirks to demonstrate end-to-end data preparation and ML modelling.
"""

import numpy as np
import pandas as pd

def generate_brightmart_dataset(n_samples=250, random_state=42):
    np.random.seed(random_state)
    
    # 1. Customer IDs
    customer_ids = [f"C{str(i+1).zfill(3)}" for i in range(n_samples)]
    
    # 2. Age: Realistic retail demographic (18 - 70)
    ages = np.random.randint(18, 71, size=n_samples)
    
    # 3. Annual Income & Spending Score: 4 distinct natural customer archetypes for K-Means
    # Group 1: High Income, High Spending (Affluent Spenders) ~ 25%
    # Group 2: High Income, Low Spending (Careful Savers) ~ 25%
    # Group 3: Low Income, High Spending (Bargain/Aspirational Spenders) ~ 25%
    # Group 4: Low Income, Low Spending (Budget Conscious) ~ 25%
    n_per_group = n_samples // 4
    remainder = n_samples - (n_per_group * 4)
    
    # Group 1: Income 700-1100, Score 65-95
    inc_g1 = np.random.normal(loc=850, scale=80, size=n_per_group)
    score_g1 = np.random.normal(loc=80, scale=8, size=n_per_group)
    
    # Group 2: Income 750-1150, Score 15-40
    inc_g2 = np.random.normal(loc=880, scale=85, size=n_per_group)
    score_g2 = np.random.normal(loc=28, scale=7, size=n_per_group)
    
    # Group 3: Income 250-500, Score 60-90
    inc_g3 = np.random.normal(loc=380, scale=60, size=n_per_group)
    score_g3 = np.random.normal(loc=75, scale=9, size=n_per_group)
    
    # Group 4: Income 200-480, Score 15-45
    inc_g4 = np.random.normal(loc=340, scale=55, size=n_per_group + remainder)
    score_g4 = np.random.normal(loc=30, scale=8, size=n_per_group + remainder)
    
    annual_incomes = np.concatenate([inc_g1, inc_g2, inc_g3, inc_g4])
    spending_scores = np.concatenate([score_g1, score_g2, score_g3, score_g4])
    
    # Clip to realistic ranges
    annual_incomes = np.round(np.clip(annual_incomes, 180, 1300), 1)
    spending_scores = np.round(np.clip(spending_scores, 5, 98)).astype(int)
    
    # Shuffle so groups are mixed across customer IDs
    shuffle_indices = np.random.permutation(n_samples)
    annual_incomes = annual_incomes[shuffle_indices]
    spending_scores = spending_scores[shuffle_indices]
    
    # 4. Purchase Frequency (purchases per year / quarter): 1 to 24
    # Correlated with spending score
    freq_base = (spending_scores / 100.0) * 14 + np.random.poisson(lam=4, size=n_samples)
    purchase_frequency = np.clip(np.round(freq_base), 1, 25).astype(int)
    
    # 5. Discount Usage: High, Medium, Low
    discount_probs = []
    for inc, scr in zip(annual_incomes, spending_scores):
        if inc < 450:
            discount_probs.append(['High', 'Medium', 'Low'][np.random.choice([0, 1, 2], p=[0.60, 0.30, 0.10])])
        elif inc > 800 and scr < 40:
            discount_probs.append(['High', 'Medium', 'Low'][np.random.choice([0, 1, 2], p=[0.45, 0.35, 0.20])])
        elif scr > 70:
            discount_probs.append(['High', 'Medium', 'Low'][np.random.choice([0, 1, 2], p=[0.25, 0.45, 0.30])])
        else:
            discount_probs.append(['High', 'Medium', 'Low'][np.random.choice([0, 1, 2], p=[0.33, 0.34, 0.33])])
    discount_usage = np.array(discount_probs)
    
    # 6. Advertising Spend (₹ thousands): 10 to 50
    ad_spend = np.round(np.random.uniform(10, 50, size=n_samples), 1)
    
    # 7. Sales (₹ thousands): Strong linear relationship with ad_spend
    # Sales = 45.0 + 4.8 * Ad_Spend + normal noise
    sales_noise = np.random.normal(loc=0, scale=12.5, size=n_samples)
    sales = np.round(45.0 + 4.8 * ad_spend + sales_noise, 1)
    sales = np.clip(sales, 30, 350)
    
    # 8. Customer_Segment (Pre-existing business segment)
    customer_segments = []
    for inc, scr in zip(annual_incomes, spending_scores):
        if inc >= 750 and scr >= 55:
            customer_segments.append("Premium")
        elif inc < 400 and scr < 40:
            customer_segments.append("Low-value")
        else:
            customer_segments.append("Regular")
            
    # 9. Campaign_Response: 'Yes' or 'No'
    # Probabilistic logit based on customer attributes
    # High spending score, higher frequency, medium/high discount usage increase likelihood
    disc_weights = {'High': 0.6, 'Medium': 0.4, 'Low': -0.3}
    disc_num = np.array([disc_weights[d] for d in discount_usage])
    
    logit = (
        -3.5 
        + 0.045 * spending_scores 
        + 0.15 * purchase_frequency 
        + 0.0012 * annual_incomes 
        - 0.015 * ages 
        + disc_num 
        + np.random.normal(0, 0.5, size=n_samples)
    )
    prob_response = 1 / (1 + np.exp(-logit))
    campaign_response = np.where(prob_response >= 0.50, "Yes", "No")
    
    df = pd.DataFrame({
        'Customer_ID': customer_ids,
        'Age': ages,
        'Annual_Income': annual_incomes,
        'Spending_Score': spending_scores,
        'Purchase_Frequency': purchase_frequency,
        'Discount_Usage': discount_usage,
        'Ad_Spend': ad_spend,
        'Sales': sales,
        'Campaign_Response': campaign_response,
        'Customer_Segment': customer_segments
    })
    
    # --- Inject Realistic Data Quality Quirks for Task A Demonstration ---
    raw_df = df.copy()
    
    # A) Missing values (4 values)
    raw_df.loc[12, 'Annual_Income'] = np.nan
    raw_df.loc[45, 'Annual_Income'] = np.nan
    raw_df.loc[88, 'Spending_Score'] = np.nan
    raw_df.loc[142, 'Ad_Spend'] = np.nan
    
    # B) Invalid boundary values (3 values to detect & clean)
    raw_df.loc[18, 'Age'] = -4                  # Impossible negative age
    raw_df.loc[72, 'Spending_Score'] = 135        # Spending score > 100
    raw_df.loc[110, 'Sales'] = -25.0             # Impossible negative sales
    
    # C) Duplicate records (3 duplicate rows)
    dup_rows = raw_df.iloc[[5, 25, 60]].copy()
    raw_df = pd.concat([raw_df, dup_rows], ignore_index=True)
    
    return raw_df, df

if __name__ == "__main__":
    raw_df, clean_truth_df = generate_brightmart_dataset(n_samples=250, random_state=42)
    raw_path = r"c:\Users\RITICK\OneDrive\Desktop\BrightMart-Retail-ML-Analytics\data\BrightMart_ML_Dataset.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"Dataset generated successfully at: {raw_path}")
    print(f"Total rows: {len(raw_df)}, Columns: {len(raw_df.columns)}")
    print(f"Class distribution of Campaign_Response:\n{raw_df['Campaign_Response'].value_counts(dropna=False)}")

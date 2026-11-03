# BrightMart Retail Data Cleaning Note
**Dataset:** BrightMart Customer and Campaign Records  
**Raw Source File:** `data/BrightMart_ML_Dataset.csv` (253 rows, 10 columns)  
**Cleaned Output File:** `data/BrightMart_Cleaned_Dataset.csv` (250 rows, 10 columns)  

---

## 1. Summary of Identified Data Quality Issues

| Step | Identified Issue | Records Affected | Justification & Method |
| :--- | :--- | :--- | :--- |
| **1. Deduplication** | 3 duplicate customer entries | Rows 5, 25, 60 (IDs duplicated) | Dropped using `pandas.DataFrame.drop_duplicates()`. Preserving duplicates artificially biases model metrics and leads to train/test data leakage. |
| **2. Missing Values** | Nulls in numerical variables | `Annual_Income` (2 nulls), `Spending_Score` (1 null), `Ad_Spend` (1 null) | Imputed with column medians (`Annual_Income` = ₹479.7k, `Spending_Score` = 54.0, `Ad_Spend` = ₹32.0k). Median imputation is resilient to extreme values and preserves sample size without distorting distributional spread. |
| **3. Age Anomaly** | Negative customer age | 1 record with Age = -4 | Corrected to the median age of legal adult customers (Age = 45). Real-world customer age cannot be negative. |
| **4. Spending Score Boundary** | Out-of-bounds score | 1 record with Spending_Score = 135 | Clipped to upper bound of 100 using `.clip(1, 100)`. Metric is standardized on a statutory scale of 1 to 100. |
| **5. Negative Sales** | Impossible negative store sales | 1 record with Sales = -₹25.0k | Corrected to positive store sales median (₹198.2k). Sales cannot be negative in this context. |

---

## 2. Validation & Post-Cleaning Audit

- **Final Row Count:** 250 verified unique records.
- **Null Value Count:** 0 across all 10 columns.
- **Feature Consistency:**
  - `Age`: Integer values between 18 and 70.
  - `Annual_Income`: ₹180.0k to ₹1,200.0k.
  - `Spending_Score`: 5 to 98 (within 1–100 boundary).
  - `Purchase_Frequency`: 1 to 24 purchases.
  - `Discount_Usage`: Balanced across 'High', 'Medium', and 'Low'.
  - `Ad_Spend`: ₹10.1k to ₹50.0k.
  - `Sales`: ₹30.0k to ₹304.3k (all positive).
  - `Campaign_Response`: 'Yes' (156, 62.4%) and 'No' (94, 37.6%).
  - `Customer_Segment`: 'Premium', 'Regular', and 'Low-value'.

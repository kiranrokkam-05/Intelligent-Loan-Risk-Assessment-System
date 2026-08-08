import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

def preprocess_and_save(input_path="loan_dataset.csv", output_path="preprocessed_loan_dataset.csv"):
    # 1. Load the dataset
    print(f"Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path)
    
    # 2. Separate features (X) and target (y)
    # We drop application_id as it is a unique identifier and not a predictive feature
    X = df.drop(columns=["application_id", "defaulted_within_12_months"])
    y = df["defaulted_within_12_months"].reset_index(drop=True)
    
    # 3. Identify feature types
    numeric_features = [
        "age",
        "emp_exp_years",
        "annual_income",
        "additional_income",
        "loan_amount",
        "loan_term_months",
        "existing_debt",
        "monthly_emi",
        "credit_score",
        "previous_defaults"
    ]
    
    categorical_features = ["emp_type", "loan_purpose"]
    ordinal_features = ["repayment_status"]
    
    # Define order for ordinal column: repayment_status (poor -> fair -> good -> excellent)
    repayment_order = ["poor", "fair", "good", "excellent"]
    
    # 4. Create Preprocessing ColumnTransformer
    # StandardScaler handles numerical scaling
    # OneHotEncoder handles categorical text (drop='first' avoids multi-collinearity)
    # OrdinalEncoder handles ranked text
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features),
            ("ord", OrdinalEncoder(categories=[repayment_order]), ordinal_features)
        ]
    )
    
    # 5. Fit and transform all features
    print("Fitting and transforming features...")
    X_transformed = preprocessor.fit_transform(X)
    
    # 6. Retrieve proper feature names post-transformation
    # Get feature names from OneHotEncoder
    onehot_encoder = preprocessor.named_transformers_["cat"]
    onehot_features = onehot_encoder.get_feature_names_out(categorical_features)
    
    # Combined feature names list
    all_feature_names = list(numeric_features) + list(onehot_features) + list(ordinal_features)
    
    # 7. Reconstruct DataFrame
    df_preprocessed = pd.DataFrame(X_transformed, columns=all_feature_names)
    
    # Re-attach target variable
    df_preprocessed["defaulted_within_12_months"] = y
    
    # 8. Save to CSV
    df_preprocessed.to_csv(output_path, index=False)
    print(f"Preprocessed dataset successfully saved to: {output_path}")
    print(f"Preprocessed dataset shape: {df_preprocessed.shape}")
    print(f"Columns in output: {list(df_preprocessed.columns)}")

if __name__ == "__main__":
    preprocess_and_save()

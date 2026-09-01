import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class FeatureEngineer:
    """
    Feature engineering pipeline for ReturnShield AI.
    Handles numerical imputation, categorical encoding, and engineering
    risk features for model training and inference.
    """
    def __init__(self, categorical_cols=None):
        if categorical_cols is None:
            self.categorical_cols = ['product_category', 'payment_method', 'item_condition', 'return_reason']
        else:
            self.categorical_cols = categorical_cols
            
        # Define predefined categories to handle unseen values gracefully during inference
        self.predefined_categories = {
            'product_category': ['Electronics', 'Clothing', 'Beauty', 'Home', 'Books'],
            'payment_method': ['Credit Card', 'Debit Card', 'UPI', 'Netbanking', 'COD'],
            'item_condition': ['New', 'Damaged', 'Wrong Item', 'Used'],
            'return_reason': ['Size issues', 'Defective', 'Not as described', 'Late delivery', 'Change of mind']
        }
        
        self.encoders = {}
        self.medians_ = {}
        self.fitted_ = False
        self.feature_names_ = []
        self.numeric_cols_ = []

    def fit(self, X, y=None):
        """
        Fits the feature engineering pipeline on the dataset X.
        Calculates numerical medians for imputation and fits encoders for categorical columns.
        """
        # Validate input columns
        self._validate_input(X)
        
        # Identify numerical columns in X
        all_numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude metadata and target columns from numerical features
        self.numeric_cols_ = [
            col for col in all_numeric 
            if col not in ['is_abusive_return', 'order_id', 'customer_id']
        ]
        
        # Calculate medians for numerical columns for robust imputation
        for col in self.numeric_cols_:
            self.medians_[col] = float(X[col].median())
            
        # Fit encoders for categorical columns that are present in X
        for col in self.categorical_cols:
            if col in X.columns:
                categories = self.predefined_categories.get(col, 'auto')
                enc = OneHotEncoder(
                    categories=[categories] if isinstance(categories, list) else 'auto',
                    handle_unknown='ignore',
                    sparse_output=False
                )
                # Fill missing values with 'Unknown'
                enc.fit(X[[col]].fillna('Unknown'))
                self.encoders[col] = enc
                
        # Determine the final list of feature names
        feature_names = list(self.numeric_cols_)
        
        # Engineered numeric feature names
        engineered_names = [
            'previous_return_frequency', 'return_to_delivery_days_ratio',
            'total_return_delay_days', 'refund_to_order_value_ratio',
            'returns_velocity_ratio_7d_30d', 'recent_returns_to_orders_ratio_7d',
            'recent_returns_to_orders_ratio_30d', 'payment_failure_rate',
            'address_change_rate', 'chargeback_rate',
            'orders_velocity_ratio_7d_30d', 'order_frequency_days',
            'high_value_return_interaction'
        ]
        feature_names.extend(engineered_names)
        
        # Categorical one-hot encoded columns
        for col in self.categorical_cols:
            if col in self.encoders:
                enc = self.encoders[col]
                encoded_cats = enc.categories_[0]
                for cat in encoded_cats:
                    feature_names.append(f"{col}_{cat}")
                    
        self.feature_names_ = feature_names
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Transforms the dataset X: imputes numerical columns, computes engineered features,
        and encodes categorical features.
        """
        if not self.fitted_:
            raise RuntimeError("FeatureEngineer must be fitted before calling transform.")
            
        # Validate columns
        self._validate_input(X)
        
        # Copy DataFrame to avoid target leakage/mutation
        df_out = X.copy()
        
        # 1. Impute and check numerical features
        for col in self.numeric_cols_:
            median_val = self.medians_.get(col, 0.0)
            df_out[col] = df_out[col].fillna(median_val)
            # Clip to valid ranges (all these features are logically non-negative)
            df_out[col] = df_out[col].clip(lower=0.0)
                
        # 2. Engineer features
        eps = 1e-5
        
        df_out['previous_return_frequency'] = df_out['previous_returns'] / (df_out['previous_orders'] + 1)
        df_out['return_to_delivery_days_ratio'] = df_out['return_days_after_delivery'] / (df_out['delivery_days'] + eps)
        df_out['total_return_delay_days'] = df_out['delivery_days'] + df_out['return_days_after_delivery']
        df_out['refund_to_order_value_ratio'] = df_out['order_amount'] / (df_out['average_order_value'] + eps)
        
        df_out['returns_velocity_ratio_7d_30d'] = df_out['returns_last_7_days'] / (df_out['returns_last_30_days'] + 1)
        df_out['recent_returns_to_orders_ratio_7d'] = df_out['returns_last_7_days'] / (df_out['orders_last_7_days'] + 1)
        df_out['recent_returns_to_orders_ratio_30d'] = df_out['returns_last_30_days'] / (df_out['orders_last_30_days'] + 1)
        
        df_out['payment_failure_rate'] = df_out['payment_failures'] / (df_out['previous_orders'] + 1)
        df_out['address_change_rate'] = df_out['address_change_count'] / (df_out['previous_orders'] + 1)
        df_out['chargeback_rate'] = df_out['previous_chargebacks'] / (df_out['previous_orders'] + 1)
        df_out['orders_velocity_ratio_7d_30d'] = df_out['orders_last_7_days'] / (df_out['orders_last_30_days'] + 1)
        df_out['order_frequency_days'] = df_out['customer_age_days'] / (df_out['previous_orders'] + 1)
        
        df_out['high_value_return_interaction'] = df_out['is_high_value_order'] * df_out['customer_return_rate']
        
        # Clean any infinite or NaN values introduced during feature calculations
        cols_to_clean = [
            'previous_return_frequency', 'return_to_delivery_days_ratio',
            'total_return_delay_days', 'refund_to_order_value_ratio',
            'returns_velocity_ratio_7d_30d', 'recent_returns_to_orders_ratio_7d',
            'recent_returns_to_orders_ratio_30d', 'payment_failure_rate',
            'address_change_rate', 'chargeback_rate',
            'orders_velocity_ratio_7d_30d', 'order_frequency_days',
            'high_value_return_interaction'
        ]
        for col in cols_to_clean:
            df_out[col] = df_out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
        # 3. Categorical encoding
        encoded_dfs = []
        for col in self.categorical_cols:
            if col in self.encoders:
                enc = self.encoders[col]
                col_data = df_out[[col]].fillna('Unknown')
                encoded_arr = enc.transform(col_data)
                encoded_cols = [f"{col}_{cat}" for cat in enc.categories_[0]]
                encoded_df = pd.DataFrame(encoded_arr, columns=encoded_cols, index=df_out.index)
                encoded_dfs.append(encoded_df)
                
        # 4. Construct final features dataframe
        base_features = self.numeric_cols_ + cols_to_clean
        X_trans = df_out[base_features].copy()
        
        if encoded_dfs:
            X_trans = pd.concat([X_trans] + encoded_dfs, axis=1)
            
        # Ensure all columns are in exactly the fitted feature order
        X_trans = X_trans.reindex(columns=self.feature_names_, fill_value=0.0)
        
        return X_trans

    def fit_transform(self, X, y=None):
        """
        Fits on X and returns transformed X.
        """
        return self.fit(X, y).transform(X)

    def _validate_input(self, X):
        """
        Ensures input X is a DataFrame and contains all required numeric columns.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame")
            
        required_raw = [
            'order_amount', 'customer_age_days', 'previous_orders', 'previous_returns',
            'customer_return_rate', 'orders_last_7_days', 'orders_last_30_days',
            'returns_last_7_days', 'returns_last_30_days', 'average_order_value',
            'discount_percentage', 'delivery_days', 'return_days_after_delivery',
            'address_change_count', 'payment_failures', 'previous_chargebacks',
            'is_first_order', 'is_high_value_order'
        ]
        
        missing = [col for col in required_raw if col not in X.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required columns: {missing}")

def load_and_preprocess_data(csv_path_or_df):
    """
    Loads raw returns data from path or DataFrame, and separates the features and target.
    
    Parameters:
    -----------
    csv_path_or_df : str or pandas.DataFrame
        Path to the csv file or the loaded DataFrame.
        
    Returns:
    --------
    X : pandas.DataFrame
        The input features (still containing raw features and IDs before transformation).
    y : pandas.Series or None
        The target column 'is_abusive_return' if present, else None.
    """
    if isinstance(csv_path_or_df, str):
        if not os.path.exists(csv_path_or_df):
            raise FileNotFoundError(f"Dataset file not found at: {csv_path_or_df}")
        df = pd.read_csv(csv_path_or_df)
    elif isinstance(csv_path_or_df, pd.DataFrame):
        df = csv_path_or_df.copy()
    else:
        raise TypeError("Input must be a CSV file path or a pandas DataFrame")
        
    # Extract target label if present
    y = None
    if 'is_abusive_return' in df.columns:
        y = df['is_abusive_return'].copy()
        
    return df, y

def prepare_data_splits(df, test_size=0.2, split_method="chronological", random_state=42):
    """
    Splits the dataset into train and test sets to prepare for modeling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame.
    test_size : float
        Proportion of dataset to include in the test split.
    split_method : str
        Method to split the data: 'chronological' (default) or 'random'.
    random_state : int
        Random seed for 'random' split method.
        
    Returns:
    --------
    train_df, test_df : pandas.DataFrame
        The training and testing DataFrames.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
        
    if split_method == "chronological":
        # Sort chronologically by timestamp if timestamp exists
        if 'timestamp' in df.columns:
            df_sorted = df.copy()
            df_sorted['timestamp_parsed'] = pd.to_datetime(df_sorted['timestamp'])
            df_sorted = df_sorted.sort_values(by='timestamp_parsed').drop(columns=['timestamp_parsed'])
        else:
            df_sorted = df.copy()
            
        split_idx = int(len(df_sorted) * (1 - test_size))
        train_df = df_sorted.iloc[:split_idx].reset_index(drop=True)
        test_df = df_sorted.iloc[split_idx:].reset_index(drop=True)
        return train_df, test_df
        
    elif split_method == "random":
        from sklearn.model_selection import train_test_split
        
        stratify = None
        if 'is_abusive_return' in df.columns:
            stratify = df['is_abusive_return']
            
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
        
    else:
        raise ValueError(f"Unknown split_method: {split_method}. Choose 'chronological' or 'random'.")

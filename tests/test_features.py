import os
# pyrefly: ignore [missing-import]
import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import (
    FeatureEngineer,
    load_and_preprocess_data,
    prepare_data_splits
)

@pytest.fixture
def sample_raw_data():
    """
    Generates a small mock dataset matching the columns in data/returns.csv
    to perform fast, isolated testing.
    """
    np.random.seed(42)
    n_records = 10
    
    data = {
        'order_id': [f"ORD_{i}" for i in range(n_records)],
        'customer_id': [f"CUST_{i % 3}" for i in range(n_records)],
        'timestamp': pd.date_range(start='2025-01-01', periods=n_records, freq='D').strftime('%Y-%m-%d %H:%M:%S'),
        'order_amount': [100.0, 50.0, 200.0, 15.0, 300.0, 80.0, 120.0, 25.0, 90.0, 160.0],
        'product_category': ['Electronics', 'Clothing', 'Beauty', 'Home', 'Books', 'Electronics', 'Clothing', 'Beauty', 'Home', 'Books'],
        'payment_method': ['Credit Card', 'Debit Card', 'UPI', 'Netbanking', 'COD', 'Credit Card', 'Debit Card', 'UPI', 'Netbanking', 'COD'],
        'customer_age_days': [100, 200, 30, 450, 600, 120, 250, 45, 480, 610],
        'previous_orders': [2, 5, 0, 10, 15, 3, 6, 1, 11, 16],
        'previous_returns': [1, 2, 0, 3, 4, 1, 2, 0, 3, 5],
        'customer_return_rate': [0.5, 0.4, 0.0, 0.3, 0.267, 0.333, 0.333, 0.0, 0.273, 0.313],
        'orders_last_7_days': [1, 2, 0, 3, 4, 1, 2, 0, 3, 4],
        'orders_last_30_days': [2, 4, 0, 8, 12, 3, 5, 1, 9, 13],
        'returns_last_7_days': [0, 1, 0, 1, 2, 0, 1, 0, 1, 2],
        'returns_last_30_days': [1, 2, 0, 2, 3, 1, 2, 0, 2, 3],
        'average_order_value': [85.0, 45.0, 0.0, 110.0, 150.0, 90.0, 48.0, 20.0, 105.0, 155.0],
        'discount_percentage': [10.0, 0.0, 20.0, 15.0, 5.0, 12.0, 0.0, 25.0, 14.0, 6.0],
        'delivery_days': [3, 2, 5, 4, 3, 3, 2, 4, 3, 3],
        'return_days_after_delivery': [5, 10, 2, 1, 7, 6, 12, 3, 1, 8],
        'address_change_count': [0, 1, 0, 2, 3, 0, 1, 0, 2, 3],
        'payment_failures': [1, 0, 0, 2, 3, 1, 0, 0, 2, 3],
        'previous_chargebacks': [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        'is_first_order': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'is_high_value_order': [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        'is_abusive_return': [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    }
    return pd.DataFrame(data)

def test_load_and_preprocess_raw_data():
    """
    Verifies that the raw dataset is loaded successfully from data/returns.csv
    and that target column is correctly isolated.
    """
    csv_path = "data/returns.csv"
    if os.path.exists(csv_path):
        X_raw, y = load_and_preprocess_data(csv_path)
        assert isinstance(X_raw, pd.DataFrame)
        assert len(X_raw) > 0
        assert y is not None
        assert len(y) == len(X_raw)
    else:
        pytest.skip("data/returns.csv not found, skipping integration load test.")

def test_pipeline_target_separation(sample_raw_data):
    """
    Verifies that FeatureEngineer fits and transforms the data correctly,
    and that the output X does not contain target-related columns.
    """
    df = sample_raw_data.copy()
    X_raw, y = load_and_preprocess_data(df)
    
    fe = FeatureEngineer()
    X_trans = fe.fit_transform(X_raw, y)
    
    assert isinstance(X_trans, pd.DataFrame)
    # Target and metadata columns must not be in X_trans
    assert 'is_abusive_return' not in X_trans.columns
    assert 'order_id' not in X_trans.columns
    assert 'customer_id' not in X_trans.columns
    assert 'timestamp' not in X_trans.columns

def test_engineered_features_exist(sample_raw_data):
    """
    Verifies that all expected engineered features are present in the output.
    """
    fe = FeatureEngineer()
    X_trans = fe.fit_transform(sample_raw_data)
    
    expected_features = [
        'previous_return_frequency', 'return_to_delivery_days_ratio',
        'total_return_delay_days', 'refund_to_order_value_ratio',
        'returns_velocity_ratio_7d_30d', 'recent_returns_to_orders_ratio_7d',
        'recent_returns_to_orders_ratio_30d', 'payment_failure_rate',
        'address_change_rate', 'chargeback_rate',
        'orders_velocity_ratio_7d_30d', 'order_frequency_days',
        'high_value_return_interaction'
    ]
    for col in expected_features:
        assert col in X_trans.columns

def test_categorical_encoding(sample_raw_data):
    """
    Verifies that categorical columns are correctly one-hot encoded
    according to predefined categories.
    """
    fe = FeatureEngineer()
    X_trans = fe.fit_transform(sample_raw_data)
    
    # Check for categories in product_category
    for cat in ['Electronics', 'Clothing', 'Beauty', 'Home', 'Books']:
        col_name = f"product_category_{cat}"
        assert col_name in X_trans.columns
        assert X_trans[col_name].isin([0.0, 1.0]).all()
        
    # Check for categories in payment_method
    for cat in ['Credit Card', 'Debit Card', 'UPI', 'Netbanking', 'COD']:
        col_name = f"payment_method_{cat}"
        assert col_name in X_trans.columns
        assert X_trans[col_name].isin([0.0, 1.0]).all()

def test_no_nan_or_inf_values(sample_raw_data):
    """
    Verifies that there are no missing or infinite values in the output features.
    """
    # Insert some NaNs and Infs manually to verify robust handling
    df = sample_raw_data.copy()
    df.loc[0, 'order_amount'] = np.nan
    df.loc[1, 'delivery_days'] = 0 # would trigger division by zero in return_to_delivery_days_ratio
    df.loc[2, 'average_order_value'] = 0.0 # would trigger division by zero in refund_to_order_value_ratio
    
    fe = FeatureEngineer()
    X_trans = fe.fit_transform(df)
    
    assert not X_trans.isnull().any().any()
    assert not np.isinf(X_trans).any().any()

def test_reproducibility(sample_raw_data):
    """
    Verifies that running the transform twice with the same fitted engineer
    on the same data produces identical results.
    """
    fe = FeatureEngineer()
    fe.fit(sample_raw_data)
    
    X_trans_1 = fe.transform(sample_raw_data)
    X_trans_2 = fe.transform(sample_raw_data)
    
    pd.testing.assert_frame_equal(X_trans_1, X_trans_2)

def test_data_splits_reproducible(sample_raw_data):
    """
    Verifies that splitting data works as expected.
    """
    # Test chronological split
    train_c1, test_c1 = prepare_data_splits(sample_raw_data, test_size=0.3, split_method="chronological")
    train_c2, test_c2 = prepare_data_splits(sample_raw_data, test_size=0.3, split_method="chronological")
    
    assert len(train_c1) == 7
    assert len(test_c1) == 3
    pd.testing.assert_frame_equal(train_c1, train_c2)
    pd.testing.assert_frame_equal(test_c1, test_c2)
    
    # Check that chronological split preserves order
    # Our sample data is already in order of index (dates increase by 1 day)
    # The split should split exactly at index 7
    pd.testing.assert_frame_equal(train_c1, sample_raw_data.iloc[:7].reset_index(drop=True))
    pd.testing.assert_frame_equal(test_c1, sample_raw_data.iloc[7:].reset_index(drop=True))
    
    # Test random split
    train_r1, test_r1 = prepare_data_splits(sample_raw_data, test_size=0.3, split_method="random", random_state=42)
    train_r2, test_r2 = prepare_data_splits(sample_raw_data, test_size=0.3, split_method="random", random_state=42)
    
    assert len(train_r1) == 7
    assert len(test_r1) == 3
    pd.testing.assert_frame_equal(train_r1, train_r2)
    pd.testing.assert_frame_equal(test_r1, test_r2)

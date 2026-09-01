import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(seed=42, num_returns_target=100000):
    """
    Generates a synthetic dataset of ecommerce returns and flags them for return abuse risk.
    Ensures chronological order and no information leakage.
    """
    np.random.seed(seed)
    
    # 1. Define customer pool (e.g., 25,000 customers to generate ~100k returns)
    num_customers = 25000
    customer_ids = [f"CUST_{i:06d}" for i in range(num_customers)]
    
    # Customer baseline features
    customer_profiles = {}
    for cust_id in customer_ids:
        # Account creation date: between 1 and 3 years before start of simulation
        account_age_days_start = np.random.randint(30, 1000)
        
        # Base return rate propensity (prob of returning any order)
        # Average is around 20%, but with wide variance
        base_return_rate = np.clip(np.random.beta(2, 8), 0.05, 0.85)
        
        # Base payment failure propensity
        base_pay_fail_rate = np.clip(np.random.beta(1, 15), 0.0, 0.4)
        
        # Base chargeback propensity
        base_chargeback_rate = np.clip(np.random.beta(0.1, 20), 0.0, 0.1)
        
        # Base address change frequency
        address_change_prob = np.clip(np.random.beta(1, 10), 0.0, 0.3)
        
        # Average order value profile
        avg_order_value_profile = np.random.exponential(80) + 20
        
        customer_profiles[cust_id] = {
            "account_age_days_start": account_age_days_start,
            "base_return_rate": base_return_rate,
            "base_pay_fail_rate": base_pay_fail_rate,
            "base_chargeback_rate": base_chargeback_rate,
            "address_change_prob": address_change_prob,
            "avg_order_value_profile": avg_order_value_profile,
            "orders": [], # will store history of order dicts
        }
    
    # 2. Simulate orders over 12 months
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    total_days = (end_date - start_date).days
    
    # To get ~100k returns with an average return rate of ~20%, we need ~500k orders
    total_orders_to_simulate = int(num_returns_target * 5.2)
    
    print(f"Simulating {total_orders_to_simulate} orders to reach target of {num_returns_target} returns...")
    
    # Generate timestamps for all orders
    # To keep simulation simple and memory efficient, we can assign orders to customers randomly,
    # and order each customer's timeline chronologically.
    
    # Assign orders to customers with a power-law-like distribution (some customers order a lot)
    customer_order_counts = np.random.zipf(1.8, size=total_orders_to_simulate)
    customer_order_counts = np.clip(customer_order_counts, 1, 100)
    
    # Let's map orders to customers
    cust_choices = np.random.choice(customer_ids, size=total_orders_to_simulate)
    
    # Order details helper values
    categories = ["Electronics", "Clothing", "Beauty", "Home", "Books"]
    category_probs = [0.15, 0.40, 0.20, 0.15, 0.10]
    category_abuse_weight = {
        "Electronics": 1.5,
        "Clothing": 1.2,
        "Beauty": 0.8,
        "Home": 0.5,
        "Books": 0.2
    }
    
    pmts = ["Credit Card", "Debit Card", "UPI", "Netbanking", "COD"]
    pmt_probs = [0.35, 0.20, 0.30, 0.05, 0.10]
    pmt_abuse_weight = {
        "COD": 1.4,
        "Netbanking": 1.1,
        "UPI": 0.9,
        "Debit Card": 0.8,
        "Credit Card": 0.7
    }
    
    # We will generate orders for each customer
    for idx, cust_id in enumerate(cust_choices):
        profile = customer_profiles[cust_id]
        
        # Determine order timestamp (randomly spread across the 12 months)
        day_offset = np.random.uniform(0, total_days)
        order_time = start_date + timedelta(days=day_offset)
        
        # Order amount centered around customer profile with some noise
        order_amount = max(5.0, np.random.normal(profile["avg_order_value_profile"], profile["avg_order_value_profile"] * 0.25))
        
        # Product category & payment method
        category = np.random.choice(categories, p=category_probs)
        payment_method = np.random.choice(pmts, p=pmt_probs)
        
        # Address changes up to now
        address_change = 1 if np.random.rand() < profile["address_change_prob"] else 0
        
        # Payment failure up to now
        pay_fail = 1 if np.random.rand() < profile["base_pay_fail_rate"] else 0
        
        # Chargeback up to now
        chargeback = 1 if np.random.rand() < profile["base_chargeback_rate"] else 0
        
        # Does the customer return this order?
        is_returned = np.random.rand() < profile["base_return_rate"]
        
        profile["orders"].append({
            "timestamp": order_time,
            "order_amount": round(order_amount, 2),
            "product_category": category,
            "payment_method": payment_method,
            "address_change": address_change,
            "payment_failure": pay_fail,
            "chargeback": chargeback,
            "is_returned": is_returned,
        })
        
    # 3. Sort orders chronologically for each customer and build features at prediction time
    return_records = []
    
    for cust_id, profile in customer_profiles.items():
        # Sort customer orders by timestamp
        profile["orders"].sort(key=lambda x: x["timestamp"])
        
        # Running customer state
        running_orders = 0
        running_returns = 0
        running_pay_failures = 0
        running_chargebacks = 0
        running_address_changes = 0
        running_order_values_sum = 0.0
        
        orders_history = []
        
        for idx, order in enumerate(profile["orders"]):
            order_time = order["timestamp"]
            order_amount = order["order_amount"]
            
            # Account age at time of order
            account_age_days = profile["account_age_days_start"] + (order_time - start_date).days
            
            # Compute sliding window features (last 7 days, last 30 days) prior to this order
            o_7 = sum(1 for prev_o in orders_history if order_time - timedelta(days=7) <= prev_o["timestamp"] < order_time)
            o_30 = sum(1 for prev_o in orders_history if order_time - timedelta(days=30) <= prev_o["timestamp"] < order_time)
            r_7 = sum(1 for prev_o in orders_history if prev_o["is_returned"] and order_time - timedelta(days=7) <= prev_o["timestamp"] < order_time)
            r_30 = sum(1 for prev_o in orders_history if prev_o["is_returned"] and order_time - timedelta(days=30) <= prev_o["timestamp"] < order_time)
            
            # Customer return rate at time of order
            cust_return_rate = (running_returns / running_orders) if running_orders > 0 else 0.0
            avg_order_val = (running_order_values_sum / running_orders) if running_orders > 0 else 0.0
            
            is_first_order = 1 if running_orders == 0 else 0
            is_high_value_order = 1 if order_amount > 150.0 else 0
            discount_percentage = round(np.clip(np.random.beta(1.5, 5) * 100, 0, 70), 1)
            
            # Shipping delivery days
            delivery_days = int(np.random.poisson(3) + 1)
            
            # Return details (only if returned)
            if order["is_returned"]:
                # Days to initiate return after delivery (usually 0 to 30 days)
                return_days_after_delivery = int(np.random.exponential(7))
                return_days_after_delivery = np.clip(return_days_after_delivery, 0, 30)
                
                # Probabilistic risk modeling for is_abusive_return
                # Risk factors weight definition
                coef_const = -3.5 # baseline low rate of abuse
                coef_return_rate = 4.0 # high return rates strongly correlate
                coef_chargebacks = 2.5 # prior chargebacks strongly correlate
                coef_pay_failures = 1.0 # minor correlation
                coef_address_changes = 1.5 # profile updates
                coef_new_acct = 1.2 if account_age_days < 90 else 0.0
                coef_cat = category_abuse_weight[order["product_category"]]
                coef_pmt = pmt_abuse_weight[order["payment_method"]]
                
                # Combine factors
                z = (
                    coef_const
                    + coef_return_rate * cust_return_rate
                    + coef_chargebacks * min(3, running_chargebacks)
                    + coef_pay_failures * min(5, running_pay_failures)
                    + coef_address_changes * min(3, running_address_changes)
                    + coef_new_acct
                    + coef_cat
                    + coef_pmt
                    + np.random.normal(0, 0.8) # noise
                )
                
                # Sigmoid probability
                prob_abuse = 1.0 / (1.0 + np.exp(-z))
                is_abusive = 1 if np.random.rand() < prob_abuse else 0
                
                return_records.append({
                    "order_id": f"ORD_{np.random.randint(100000, 999999)}_{len(return_records)}",
                    "customer_id": cust_id,
                    "timestamp": order_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "order_amount": order_amount,
                    "product_category": order["product_category"],
                    "payment_method": order["payment_method"],
                    "customer_age_days": account_age_days,
                    "previous_orders": running_orders,
                    "previous_returns": running_returns,
                    "customer_return_rate": round(cust_return_rate, 4),
                    "orders_last_7_days": o_7,
                    "orders_last_30_days": o_30,
                    "returns_last_7_days": r_7,
                    "returns_last_30_days": r_30,
                    "average_order_value": round(avg_order_val, 2),
                    "discount_percentage": discount_percentage,
                    "delivery_days": delivery_days,
                    "return_days_after_delivery": return_days_after_delivery,
                    "address_change_count": running_address_changes,
                    "payment_failures": running_pay_failures,
                    "previous_chargebacks": running_chargebacks,
                    "is_first_order": is_first_order,
                    "is_high_value_order": is_high_value_order,
                    "is_abusive_return": is_abusive
                })
            
            # Update customer history state
            running_orders += 1
            running_order_values_sum += order_amount
            running_pay_failures += order["payment_failure"]
            running_chargebacks += order["chargeback"]
            running_address_changes += order["address_change"]
            if order["is_returned"]:
                running_returns += 1
                
            orders_history.append(order)

    # 4. Convert to DataFrame and sort chronologically overall
    df = pd.DataFrame(return_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Limit to required target count
    if len(df) > num_returns_target:
        df = df.head(num_returns_target)
        
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Return Abuse Data Generator")
    parser.parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--target-size", type=int, default=100000, help="Target number of return records")
    parser.add_argument("--output", type=str, default="C:\\Users\\rohit\\Documents\\ReturnShield_ai\\data\\returns.csv", help="Output CSV path")
    
    args = parser.parse_known_args()[0]
    
    print(f"Generating synthetic dataset (Seed: {args.seed}, Target Size: {args.target_size})...")
    df_generated = generate_synthetic_data(seed=args.seed, num_returns_target=args.target_size)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_generated.to_csv(args.output, index=False)
    print(f"Dataset generated and saved successfully to: {args.output}")

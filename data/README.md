# ReturnShield AI Synthetic Returns Dataset

This directory contains the synthetic ecommerce return-risk dataset generated for ReturnShield AI development.

## Dataset Overview

* **File**: `returns.csv`
* **Size**: 100,000 records, 24 columns
* **Domain**: Synthetic ecommerce return requests
* **Time Span**: ~12 months (2025-01-01 to 2025-12-16)
* **Goal**: Target classification (`is_abusive_return`) predicting high-risk/abusive returns.

> [!NOTE]
> This dataset is **100% synthetic** and programmatically generated. It does not contain any real customer records, PII, or real transaction data.

---

## Features & Business Meanings

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `order_id` | String | Unique identifier for the order/return transaction. |
| `customer_id` | String | Unique identifier for the customer. |
| `timestamp` | String | The datetime when the return was requested. |
| `order_amount` | Float | Total transaction value of the order. |
| `product_category` | Categorical | Category of the returned product (`Electronics`, `Clothing`, `Beauty`, `Home`, `Books`). |
| `payment_method` | Categorical | Payment method used (`Credit Card`, `Debit Card`, `UPI`, `Netbanking`, `COD`). |
| `customer_age_days` | Integer | Account age in days at the time the order was placed. |
| `previous_orders` | Integer | Total orders placed by the customer prior to this transaction. |
| `previous_returns` | Integer | Total returns initiated by the customer prior to this transaction. |
| `customer_return_rate` | Float | Running return rate (`previous_returns / previous_orders`). |
| `orders_last_7_days` | Integer | Number of orders placed by the customer in the last 7 days. |
| `orders_last_30_days` | Integer | Number of orders placed by the customer in the last 30 days. |
| `returns_last_7_days` | Integer | Number of returns initiated by the customer in the last 7 days. |
| `returns_last_30_days` | Integer | Number of returns initiated by the customer in the last 30 days. |
| `average_order_value` | Float | Historical average order value of the customer before this order. |
| `discount_percentage` | Float | The discount applied to this order (0 to 70%). |
| `delivery_days` | Integer | Days between order creation and package delivery. |
| `return_days_after_delivery` | Integer | Days between delivery and return initiation. |
| `address_change_count` | Integer | Running count of delivery address modifications by the customer. |
| `payment_failures` | Integer | Running count of payment failures prior to this order. |
| `previous_chargebacks` | Integer | Running count of payment chargebacks initiated by the customer. |
| `is_first_order` | Binary (0/1) | Whether this order was the customer's very first purchase. |
| `is_high_value_order` | Binary (0/1) | Whether the current order amount exceeds \$150. |
| `is_abusive_return` | Binary (0/1) | **Target Variable**: 1 if the return is abusive/high-risk, 0 if normal. |

---

## Target Generation Logic

The target variable `is_abusive_return` is generated programmatically using a logistic function based on multiple behavioral variables plus random Gaussian noise:

1. **Risk Coefficients**:
   * Baseline intercept: `-3.5` (represents a low baseline rate of abuse).
   * High customer return rate: `+4.0`
   * Historical chargebacks (capped at 3): `+2.5` per chargeback
   * Address changes (capped at 3): `+1.5` per change
   * Payment failures (capped at 5): `+1.0` per failure
   * New Account risk (age < 90 days): `+1.2`
   * Product Category risk weight (e.g., Electronics `+1.5`, Books `+0.2`)
   * Payment Method risk weight (e.g., Cash on Delivery `+1.4`, Credit Card `+0.7`)

2. **Probability Calculation**:
   The risk factors are summed to compute a log-odds risk score \( z \):
   \[
   z = z_{base} + \text{weights} \cdot \text{features} + \epsilon
   \]
   where \( \epsilon \sim N(0, 0.8) \). The score is then passed through a sigmoid function to yield a probability \( p \):
   \[
   p = \frac{1}{1 + e^{-z}}
   \]

3. **Bernoulli Sampling**:
   `is_abusive_return` is drawn from a Bernoulli distribution with parameter \( p \).

This probabilistic design ensures realistic overlap between normal and abusive instances, producing a non-trivial classification boundary (with some high-risk profiles occasionally behaving normally, and vice versa).

---

## Data Leakage Prevention Rules

To guarantee strict real-world deployability, the dataset conforms to the following guidelines:
* **Point-in-Time Features Only**: All historical counters (`previous_orders`, `previous_returns`, `payment_failures`, `previous_chargebacks`) represent only events that occurred *prior* to the timestamp of the current transaction.
* **No Target Derivation**: No features are derived directly from the `is_abusive_return` outcome.
* **No Future Information**: The dataset does not include subsequent transaction activity or chargebacks that occur after the return initiation date.

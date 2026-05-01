select
  order_id,
  amount,
  double_amount,
  (double_amount - amount) as delta
from {{ ref('customer_features') }}

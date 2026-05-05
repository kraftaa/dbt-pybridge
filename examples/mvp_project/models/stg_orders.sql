{{ config(materialized='table') }}

select
  cast(order_id as bigint) as order_id,
  cast(amount as numeric(12,2)) as amount
from {{ ref('orders') }}

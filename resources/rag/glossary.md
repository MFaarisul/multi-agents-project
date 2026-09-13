# Enterprise Glossary

## Revenue
Revenue = SUM(orders.amount) WHERE order_status = 'completed'. Queries asking for
"total revenue" without a status qualifier should use this definition. Note: monthly
trends computed from customer_events (event_type = 'order_placed') measure placed
order volume, not recognized revenue.

## Active Customer
A customer is considered "active" if they have at least one subscription with
status = 'active' as of the reporting date.

## Monthly Recurring Revenue (MRR)
MRR is the sum of `monthly_fee` across all subscriptions whose status is
'active'. Paused and past_due subscriptions are excluded from MRR.

## Churn
A customer has churned when all of their subscriptions have status
'canceled' or 'past_due' and no new subscription has been created within
30 days.

## Refund Rate
Refund Rate = (sum of refunds.amount) / (sum of orders.amount where
order_status = 'completed') for a given period.

## Customer Lifetime Value (CLV)
CLV = (sum of completed orders.amount) + (sum of active subscription
monthly_fee * 12) per customer.

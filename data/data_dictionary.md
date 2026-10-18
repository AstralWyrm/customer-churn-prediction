# Churn Dataset Data Dictionary

Each row corresponds to one telecom customer account observed at the end of a billing period.

## Identifier

- `customer_id`: unique synthetic customer identifier

## Categorical Features

- `region`: one of `Metro`, `Regional`, `Coastal`, `Rural`
- `contract_type`: one of `Month-to-month`, `One year`, `Two year`
- `internet_service`: one of `Fiber`, `DSL`, `Wireless`, `None`
- `payment_method`: one of `Electronic check`, `Bank transfer`, `Credit card`, `Mailed check`
- `streaming_bundle`: `Yes` or `No`
- `device_protection`: `Yes` or `No`
- `paperless_billing`: `Yes` or `No`
- `senior_citizen`: `Yes` or `No`

## Numerical Features

- `tenure_months`: number of months the customer has been with the provider
- `monthly_charges`: current monthly bill in dollars
- `avg_monthly_gb`: average monthly data usage in gigabytes
- `support_tickets_90d`: support tickets in the last 90 days
- `late_payments_6m`: number of late payments in the last 6 months
- `outage_hours_90d`: service outage hours experienced in the last 90 days
- `num_services`: number of subscribed services

## Target

- `churned`: `1` if the customer cancelled service within the next 30 days, otherwise `0`

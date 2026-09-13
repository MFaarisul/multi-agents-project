-- Clean up existing schema
DROP TABLE IF EXISTS customer_events CASCADE;
DROP TABLE IF EXISTS refunds CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS subscriptions CASCADE;
DROP TABLE IF EXISTS marketing_campaigns CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- 1. CUSTOMERS TABLE
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    country VARCHAR(50) NOT NULL,
    segment VARCHAR(20) CHECK (segment IN ('Enterprise', 'SMB', 'Consumer')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. SUBSCRIPTIONS TABLE
CREATE TABLE subscriptions (
    subscription_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id) ON DELETE CASCADE,
    plan_tier VARCHAR(20) CHECK (plan_tier IN ('Basic', 'Pro', 'Enterprise')),
    monthly_fee NUMERIC(10, 2) NOT NULL,
    status VARCHAR(20) CHECK (status IN ('active', 'canceled', 'paused', 'past_due')),
    start_date DATE NOT NULL,
    end_date DATE
);

-- 3. ORDERS TABLE
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id) ON DELETE CASCADE,
    amount NUMERIC(10, 2) NOT NULL,
    order_status VARCHAR(20) CHECK (order_status IN ('completed', 'pending', 'failed', 'refunded')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. REFUNDS TABLE
CREATE TABLE refunds (
    refund_id SERIAL PRIMARY KEY,
    order_id INT REFERENCES orders(order_id) ON DELETE CASCADE,
    amount NUMERIC(10, 2) NOT NULL,
    reason VARCHAR(100),
    refunded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. MARKETING CAMPAIGNS TABLE
CREATE TABLE marketing_campaigns (
    campaign_id SERIAL PRIMARY KEY,
    campaign_name VARCHAR(100) NOT NULL,
    channel VARCHAR(50) CHECK (channel IN ('Google Ads', 'Meta', 'LinkedIn', 'Email', 'Organic')),
    budget NUMERIC(10, 2) NOT NULL,
    spent NUMERIC(10, 2) NOT NULL,
    quarter VARCHAR(5) CHECK (quarter IN ('Q1', 'Q2', 'Q3', 'Q4')),
    campaign_year INT NOT NULL
);

-- 6. CUSTOMER EVENTS TABLE (historical activity log, derived from the tables
--    above at seed time — see the population section at the bottom)
CREATE TABLE customer_events (
    event_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id) ON DELETE CASCADE,
    order_id INT REFERENCES orders(order_id) ON DELETE SET NULL,
    event_type VARCHAR(30) CHECK (event_type IN ('signup', 'subscription_started', 'order_placed', 'refund_issued')),
    event_date TIMESTAMP WITH TIME ZONE NOT NULL,
    amount NUMERIC(10, 2),
    details VARCHAR(200)
);

-- ============================================================================
-- DYNAMIC POPULATION (50 to 150 rows per table)
-- ============================================================================

-- Fix the PRNG seed so every fresh init produces the IDENTICAL dataset.
-- (Remove this line if you want a different random dataset per reset.)
SELECT setseed(0.42);

-- Populate Customers (600 rows)
INSERT INTO customers (full_name, email, country, segment, created_at)
SELECT
    'User ' || i AS full_name,
    'user_' || i || '@example' || (i % 5 + 1) || '.com' AS email,
    (ARRAY['United States', 'Canada', 'United Kingdom', 'Germany', 'Australia', 'Indonesia'])[floor(random() * 6 + 1)] AS country,
    (ARRAY['Enterprise', 'SMB', 'Consumer'])[floor(random() * 3 + 1)] AS segment,
    NOW() - (i || ' days')::INTERVAL AS created_at
FROM generate_series(1, 600) s(i);

-- Populate Subscriptions (1000 rows, random customer assignment)
-- NOTE: tier/fee are coupled via a single per-row index k. Do NOT use an
-- uncorrelated CROSS JOIN LATERAL here — PostgreSQL evaluates it once and
-- every row would get the same tier (observed: all rows 'Pro').
INSERT INTO subscriptions (customer_id, plan_tier, monthly_fee, status, start_date, end_date)
SELECT
    c.customer_id,
    (ARRAY['Basic', 'Pro', 'Enterprise'])[c.k] AS plan_tier,
    (ARRAY[29.00, 199.00, 1500.00])[c.k] AS monthly_fee,
    (ARRAY['active', 'active', 'active', 'canceled', 'paused', 'past_due'])[floor(random() * 6 + 1)] AS status,
    c.start_date,
    CASE
        WHEN random() < 0.2 THEN (c.start_date + (floor(random() * 150 + 30)::INT || ' days')::INTERVAL)::DATE
        ELSE NULL
    END AS end_date
FROM (
    SELECT
        floor(random() * 600 + 1)::INT AS customer_id,
        floor(random() * 3 + 1)::INT AS k,
        (NOW() - (floor(random() * 730)::INT || ' days')::INTERVAL)::DATE AS start_date
    FROM generate_series(1, 1000) s(i)
) c;

-- Populate Orders (1000 rows, spread over 2 years for trend analysis)
INSERT INTO orders (customer_id, amount, order_status, created_at)
SELECT
    floor(random() * 600 + 1)::INT AS customer_id,
    (ARRAY[29.00, 199.00, 500.00, 1500.00, 2500.00])[floor(random() * 5 + 1)] AS amount,
    (ARRAY['completed', 'completed', 'completed', 'pending', 'failed', 'refunded'])[floor(random() * 6 + 1)] AS order_status,
    NOW() - (floor(random() * 730)::INT || ' days')::INTERVAL AS created_at
FROM generate_series(1, 1000) s(i);

-- Populate Refunds (200 rows, sampled from completed orders only — keeps a
-- realistic ~20% refund rate and preserves completed orders for revenue demos)
INSERT INTO refunds (order_id, amount, reason, refunded_at)
SELECT
    o.order_id,
    o.amount AS amount,
    (ARRAY['Customer dissatisfaction', 'Accidental duplicate purchase', 'Billing error', 'Service downgrade'])[floor(random() * 4 + 1)] AS reason,
    o.created_at + (floor(random() * 10 + 1)::INT || ' days')::INTERVAL AS refunded_at
FROM orders o
WHERE o.order_status = 'completed'
LIMIT 200;

UPDATE orders SET order_status = 'refunded' WHERE order_id IN (SELECT order_id FROM refunds);

-- Populate Marketing Campaigns (1000 rows)
-- NOTE: budget is computed per row inside the inner subquery (never in an
-- uncorrelated lateral, which PostgreSQL evaluates once for all rows).
-- spent is derived in the outer query from that row's own budget.
INSERT INTO marketing_campaigns (campaign_name, channel, budget, spent, quarter, campaign_year)
SELECT
    'Campaign ' || x.i || ' - ' || x.suffix AS campaign_name,
    x.channel AS channel,
    x.budget AS budget,
    ROUND((x.budget * (0.85 + (random() * 0.30)))::NUMERIC, 2) AS spent,
    x.quarter AS quarter,
    x.campaign_year AS campaign_year
FROM (
    SELECT
        s.i,
        (ARRAY['Growth', 'Retargeting', 'Brand', 'Product Launch'])[floor(random() * 4 + 1)] AS suffix,
        (ARRAY['Google Ads', 'Meta', 'LinkedIn', 'Email', 'Organic'])[floor(random() * 5 + 1)] AS channel,
        ROUND((random() * 50000 + 5000)::NUMERIC, 2) AS budget,
        (ARRAY['Q1', 'Q2', 'Q3', 'Q4'])[floor(random() * 4 + 1)] AS quarter,
        (ARRAY[2024, 2025, 2026])[floor(random() * 3 + 1)] AS campaign_year
    FROM generate_series(1, 1000) s(i)
) x;

-- ============================================================================
-- CUSTOMER EVENTS (historical activity log, ~2800 rows)
-- Derived from the real tables above (no randomness) so timestamps, amounts
-- and references stay perfectly consistent.
-- ============================================================================

-- Signups (600)
INSERT INTO customer_events (customer_id, order_id, event_type, event_date, amount, details)
SELECT customer_id, NULL, 'signup', created_at, NULL, 'Account created'
FROM customers;

-- Subscription starts (1000)
INSERT INTO customer_events (customer_id, order_id, event_type, event_date, amount, details)
SELECT customer_id, NULL, 'subscription_started', start_date::TIMESTAMPTZ, monthly_fee, plan_tier || ' plan started'
FROM subscriptions;

-- Orders placed (1000)
INSERT INTO customer_events (customer_id, order_id, event_type, event_date, amount, details)
SELECT customer_id, order_id, 'order_placed', created_at, amount, 'Order placed (status: ' || order_status || ')'
FROM orders;

-- Refunds issued (200)
INSERT INTO customer_events (customer_id, order_id, event_type, event_date, amount, details)
SELECT o.customer_id, r.order_id, 'refund_issued', r.refunded_at, r.amount, r.reason
FROM refunds r
JOIN orders o ON o.order_id = r.order_id;

-- ============================================================================
-- ENFORCE READ-ONLY DATABASE SECURITY ROLE
-- ============================================================================

-- Create dedicated agent user if not exists
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'agent_readonly') THEN
      CREATE USER agent_readonly WITH PASSWORD 'readonly_secret_password';
   END IF;
END $$;

-- Revoke write/schema modification permissions
REVOKE ALL ON SCHEMA public FROM agent_readonly;
GRANT CONNECT ON DATABASE analytics_db TO agent_readonly;
GRANT USAGE ON SCHEMA public TO agent_readonly;

-- Grant SELECT-only privileges on current and future tables
GRANT SELECT ON ALL TABLES IN SCHEMA public TO agent_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO agent_readonly;

-- Explicitly revoke write capabilities
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON ALL TABLES IN SCHEMA public FROM agent_readonly;

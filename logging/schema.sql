-- AIAA real-data logging schema
-- Run once: psql -U <user> -d <db> -f schema.sql

CREATE TABLE IF NOT EXISTS task_logs (
    id              SERIAL PRIMARY KEY,
    task_id         VARCHAR(64) NOT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),

    -- what the agent saw
    task_type       VARCHAR(64),          -- e.g. 'calendar_event', 'bill_payment'
    context_json    JSONB NOT NULL,       -- full input context, raw

    -- what each decision path said
    rule_decision   BOOLEAN,              -- did the hardcoded rule say "intervene"?
    rule_threshold  FLOAT,                -- threshold value used at decision time
    model_prob      FLOAT,                -- LSTM's predicted probability (NULL until trained)
    final_decision  BOOLEAN,              -- what actually happened (rule or model, whichever gated)
    decision_source VARCHAR(16),          -- 'rule' | 'model' -- which one actually decided

    -- ground truth outcome
    outcome_success BOOLEAN,              -- did the downstream action actually succeed
    outcome_detail  JSONB,                -- confirmation ID, error msg, API response, etc.
    user_feedback   VARCHAR(16),          -- 'accepted' | 'overridden' | 'rejected' | NULL

    execution_time_ms INT
);

CREATE INDEX IF NOT EXISTS idx_task_logs_type ON task_logs(task_type);
CREATE INDEX IF NOT EXISTS idx_task_logs_created ON task_logs(created_at);

-- Query you'll actually use for your impact number later:
-- SELECT decision_source, COUNT(*), AVG(outcome_success::int)
-- FROM task_logs GROUP BY decision_source;

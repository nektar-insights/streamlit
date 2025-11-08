-- ============================================================================
-- Maturity Date Schema Migration
-- ============================================================================
-- This script extends the loan_summaries table with canonical maturity fields
-- and creates a DQ tracking table for maturity date quality issues.
--
-- IMPORTANT: Run this in your Supabase SQL editor
-- ============================================================================

-- Step 1: Extend loan_summaries table with maturity metadata
-- ============================================================================

-- Add maturity_basis column (tracks whether maturity is original/amended/renewed/extended)
ALTER TABLE loan_summaries
ADD COLUMN IF NOT EXISTS maturity_basis VARCHAR(20) DEFAULT 'original';

COMMENT ON COLUMN loan_summaries.maturity_basis IS
'Origin of maturity date: original | amended | renewed | extended';

-- Add maturity_version column (increments with each amendment)
ALTER TABLE loan_summaries
ADD COLUMN IF NOT EXISTS maturity_version INT DEFAULT 1;

COMMENT ON COLUMN loan_summaries.maturity_version IS
'Version number of maturity date (increments with amendments)';

-- Add maturity_last_updated_at column (audit timestamp)
ALTER TABLE loan_summaries
ADD COLUMN IF NOT EXISTS maturity_last_updated_at TIMESTAMPTZ DEFAULT NOW();

COMMENT ON COLUMN loan_summaries.maturity_last_updated_at IS
'Timestamp when maturity_date was last updated from HubSpot';

-- Add maturity_quality column (validation status)
ALTER TABLE loan_summaries
ADD COLUMN IF NOT EXISTS maturity_quality VARCHAR(20) DEFAULT 'ok';

COMMENT ON COLUMN loan_summaries.maturity_quality IS
'Data quality flag: ok | missing | invalid | backdated';

-- Add maturity_source column (HubSpot field name)
ALTER TABLE loan_summaries
ADD COLUMN IF NOT EXISTS maturity_source VARCHAR(100) DEFAULT 'maturity_date';

COMMENT ON COLUMN loan_summaries.maturity_source IS
'Original HubSpot field name that provided this maturity date';

-- Add amendment_id column (reference to amendment record)
ALTER TABLE loan_summaries
ADD COLUMN IF NOT EXISTS amendment_id VARCHAR(50);

COMMENT ON COLUMN loan_summaries.amendment_id IS
'Reference to amendment record if maturity was amended';

-- Create index on maturity_quality for DQ queries
CREATE INDEX IF NOT EXISTS idx_loan_summaries_maturity_quality
ON loan_summaries(maturity_quality)
WHERE maturity_quality != 'ok';

-- Create index on maturity_last_updated_at for incremental processing
CREATE INDEX IF NOT EXISTS idx_loan_summaries_maturity_updated
ON loan_summaries(maturity_last_updated_at DESC);


-- Step 2: Create maturity DQ log table
-- ============================================================================

CREATE TABLE IF NOT EXISTS maturity_dq_log (
    id SERIAL PRIMARY KEY,
    loan_id VARCHAR(50) NOT NULL,
    issue_type VARCHAR(50) NOT NULL,
    issue_description TEXT,
    old_value DATE,
    new_value DATE,
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT
);

COMMENT ON TABLE maturity_dq_log IS
'Data quality log for maturity date issues and changes';

COMMENT ON COLUMN maturity_dq_log.issue_type IS
'Type of DQ issue: missing | invalid | backdated | amended | corrected';

-- Create indexes for DQ log queries
CREATE INDEX IF NOT EXISTS idx_maturity_dq_log_loan_id
ON maturity_dq_log(loan_id);

CREATE INDEX IF NOT EXISTS idx_maturity_dq_log_detected_at
ON maturity_dq_log(detected_at DESC);

CREATE INDEX IF NOT EXISTS idx_maturity_dq_log_unresolved
ON maturity_dq_log(resolved)
WHERE resolved = FALSE;


-- Step 3: Create view for maturity DQ summary
-- ============================================================================

CREATE OR REPLACE VIEW maturity_dq_summary AS
SELECT
    issue_type,
    COUNT(*) as issue_count,
    COUNT(CASE WHEN resolved = FALSE THEN 1 END) as unresolved_count,
    MIN(detected_at) as first_seen,
    MAX(detected_at) as last_seen
FROM maturity_dq_log
GROUP BY issue_type
ORDER BY unresolved_count DESC, issue_count DESC;

COMMENT ON VIEW maturity_dq_summary IS
'Summary of maturity DQ issues by type';


-- Step 4: Backfill script (validate existing data)
-- ============================================================================

-- This query identifies existing data quality issues that should be logged
-- Run this AFTER the migration to populate the DQ log with existing issues

INSERT INTO maturity_dq_log (loan_id, issue_type, issue_description, new_value, detected_at)
SELECT
    loan_id,
    CASE
        WHEN maturity_date IS NULL THEN 'missing'
        WHEN funding_date IS NOT NULL AND maturity_date <= funding_date THEN 'invalid'
        ELSE 'unknown'
    END as issue_type,
    CASE
        WHEN maturity_date IS NULL THEN 'Maturity date is missing'
        WHEN funding_date IS NOT NULL AND maturity_date <= funding_date THEN
            'Maturity date (' || maturity_date || ') is on or before funding date (' || funding_date || ')'
        ELSE 'Unknown issue'
    END as issue_description,
    maturity_date as new_value,
    NOW() as detected_at
FROM loan_summaries
WHERE
    (maturity_date IS NULL) OR
    (funding_date IS NOT NULL AND maturity_date <= funding_date)
ON CONFLICT DO NOTHING;

-- Update maturity_quality field based on validation
UPDATE loan_summaries
SET maturity_quality = CASE
    WHEN maturity_date IS NULL THEN 'missing'
    WHEN funding_date IS NOT NULL AND maturity_date <= funding_date THEN 'invalid'
    ELSE 'ok'
END
WHERE maturity_quality = 'ok' OR maturity_quality IS NULL;


-- Step 5: Grant permissions (adjust based on your setup)
-- ============================================================================

-- Grant SELECT on DQ log to authenticated users (read-only)
GRANT SELECT ON maturity_dq_log TO authenticated;
GRANT SELECT ON maturity_dq_summary TO authenticated;

-- Grant INSERT on DQ log to service role (for Python app)
-- GRANT INSERT ON maturity_dq_log TO service_role;


-- ============================================================================
-- Verification Queries
-- ============================================================================

-- Check maturity quality distribution
SELECT
    maturity_quality,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM loan_summaries
GROUP BY maturity_quality
ORDER BY count DESC;

-- Check recent DQ issues
SELECT
    loan_id,
    issue_type,
    issue_description,
    old_value,
    new_value,
    detected_at,
    resolved
FROM maturity_dq_log
ORDER BY detected_at DESC
LIMIT 20;

-- View DQ summary
SELECT * FROM maturity_dq_summary;

-- ============================================================================
-- Rollback Script (in case of issues)
-- ============================================================================

/*
-- To rollback this migration, run:

DROP VIEW IF EXISTS maturity_dq_summary;
DROP TABLE IF EXISTS maturity_dq_log;
DROP INDEX IF EXISTS idx_loan_summaries_maturity_quality;
DROP INDEX IF EXISTS idx_loan_summaries_maturity_updated;

ALTER TABLE loan_summaries DROP COLUMN IF EXISTS maturity_basis;
ALTER TABLE loan_summaries DROP COLUMN IF EXISTS maturity_version;
ALTER TABLE loan_summaries DROP COLUMN IF EXISTS maturity_last_updated_at;
ALTER TABLE loan_summaries DROP COLUMN IF EXISTS maturity_quality;
ALTER TABLE loan_summaries DROP COLUMN IF EXISTS maturity_source;
ALTER TABLE loan_summaries DROP COLUMN IF EXISTS amendment_id;
*/

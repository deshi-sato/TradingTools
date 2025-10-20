-- ops_add_unique_indexes.sql
-- Purpose: fix ON CONFLICT(order_id) error by ensuring a UNIQUE target exists.

BEGIN;

-- paper_pairs.order_id must be UNIQUE (or PK) so INSERT ... ON CONFLICT(order_id) works.
CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_pairs_order_id ON paper_pairs(order_id);

COMMIT;

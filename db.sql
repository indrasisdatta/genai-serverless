-- ----------------------------------------------------------------------------
-- 1. Document registry
--
-- One row per uploaded PDF. Everything the system knows *about* a document lives
-- here; the chunk tables only carry a foreign key back to it. That is what makes
-- later additions cheap -- adding effective_to, states, or content_kind means
-- altering this five-row table, not rewriting hundreds of chunk rows.
-- ----------------------------------------------------------------------------
-- ============================================================================
-- documents - registry of policy PDFs available to the RAG pipeline
--
-- Rows are created by the existing admin upload section, which stores the file
-- and inserts with index_status = 'pending'. The RAG app picks up pending rows,
-- chunks and embeds them, and moves them to 'indexed'.
--
-- Requires PostgreSQL 13+ (gen_random_uuid built in).
-- ============================================================================

CREATE TABLE documents (
  -- ---- identity ------------------------------------------------------------
  doc_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Stable human-readable key, e.g. 'doc_vzw_ca_2025'. This is the string in
  -- gold_doc_ids in golden_v1.jsonl. It exists because doc_id is regenerated if
  -- a row is ever recreated; if the golden set referenced UUIDs, that would
  -- break every reference and recall@k would read 0 while looking like a
  -- retriever bug.
  slug              TEXT UNIQUE NOT NULL,

  title             TEXT NOT NULL,   -- used in the identity stamp and citations
  doc_type          TEXT NOT NULL,   -- customer_agreement | return_policy | ...

  -- Array because one document can cover several lines: the Fios agreement
  -- covers internet, TV and home phone. Retrieval filters with && (overlap).
  product_line      TEXT[] NOT NULL,

  -- ---- file location -------------------------------------------------------
  -- storage_path is the source of truth for indexing: the local copy written by
  -- the admin section. source_url is provenance only - where the document came
  -- from, so a human can check whether a newer version has been published.
  -- Never index from source_url: Verizon replaces the file at
  -- 'customer-agreement-policy-current-en.pdf' in place, so re-indexing from the
  -- URL would silently embed different text under the same doc_id.
  storage_path      TEXT NOT NULL,
  source_url        TEXT,
  file_hash         TEXT,            -- sha256 of the stored bytes; drift detection

  -- ---- versioning ----------------------------------------------------------
  version_label     TEXT,            -- 'CA-0625EN', 'v.2022-2'
  effective_from    DATE,            -- NULL where the document states no date
  effective_to      DATE,            -- NULL = still current
  supersedes_doc_id UUID REFERENCES documents(doc_id),  -- self-ref: version chain

  -- 1 = governing contract, 2 = policy/collateral, 3 = FAQ/marketing.
  -- Tie-break when two documents disagree; the FAQ never beats the contract.
  authority_tier    SMALLINT NOT NULL DEFAULT 2,

  status            TEXT NOT NULL DEFAULT 'active',  -- active | superseded | draft

  -- ---- indexing state ------------------------------------------------------
  -- A state column rather than a boolean, because "never tried" and "tried and
  -- failed" need to be distinguishable or a retry sweep cannot tell them apart.
  -- 'indexing' is a claim marker: workers take rows with
  -- UPDATE ... WHERE index_status = 'pending' RETURNING, so two concurrent runs
  -- cannot index the same document twice.
  index_status      TEXT NOT NULL DEFAULT 'pending',  -- pending | indexing | indexed | failed
  index_attempts    SMALLINT NOT NULL DEFAULT 0,      -- ceiling stops infinite retries
  index_error       TEXT,                             -- last failure, for triage
  indexed_at        TIMESTAMP,

  -- ---- bookkeeping ---------------------------------------------------------
  notes             TEXT,            -- known caveats, read by humans not code
  created_at        TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at        TIMESTAMP NOT NULL DEFAULT NOW(),

  -- ---- constraints ---------------------------------------------------------
  -- Typos in these become silent retrieval failures rather than errors, so they
  -- are enforced here. CHECK rather than ENUM: adding a value is an ALTER on the
  -- constraint, not a type migration.
  CONSTRAINT documents_status_check
    CHECK (status IN ('active', 'superseded', 'draft')),
  CONSTRAINT documents_index_status_check
    CHECK (index_status IN ('pending', 'indexing', 'indexed', 'failed')),
  CONSTRAINT documents_authority_tier_check
    CHECK (authority_tier BETWEEN 1 AND 3),
  CONSTRAINT documents_product_line_not_empty
    CHECK (cardinality(product_line) > 0),
  -- A document cannot stop being effective before it starts.
  CONSTRAINT documents_effective_range_check
    CHECK (effective_to IS NULL OR effective_from IS NULL OR effective_to > effective_from)
);

-- Worker queue lookup: WHERE index_status = 'pending'. Partial index because
-- only unfinished rows are ever queried this way, and the table is mostly
-- 'indexed' in steady state.
CREATE INDEX idx_documents_pending
  ON documents (index_status)
  WHERE index_status IN ('pending', 'failed');

-- No index on product_line. Five rows: a sequential scan beats an index lookup.
-- Revisit at a few thousand documents.


INSERT INTO documents
  (slug, title, doc_type, product_line,
   storage_path, source_url,
   version_label, effective_from, authority_tier, status,
   index_status, notes)
VALUES

-- 1 ---------------------------------------------------------------------------
('doc_vzw_ca_2025',
 'My Verizon Wireless Customer Agreement',
 'customer_agreement',
 ARRAY['mobile_postpaid','mobile_prepaid'],
 '/srv/documents/doc_vzw_ca_2025.pdf',
 'https://ss7.vzw.com/is/content/VerizonWireless/Footer/cap/customer-agreement-policy-current-en.pdf',
 'CA-0625EN', DATE '2025-07-09', 1, 'active',
 'pending',
 'Body ends "Updated July 9, 2025". Source for the $7 late fee and $30 returned payment fee.'),

-- 2 ---------------------------------------------------------------------------
-- Tier 2 despite reproducing the CA text near-verbatim, so doc_vzw_ca_2025 wins
-- on shared clauses. Still the ONLY source for ETF amounts, activation fees and
-- surcharge rates (golden items gd-020 to gd-029).
('doc_vzw_ipi_2025',
 'Customer Agreement & Important Information',
 'plan_information_brochure',
 ARRAY['mobile_postpaid'],
 '/srv/documents/doc_vzw_ipi_2025.pdf',
 'https://www.verizon.com/support/pdf/collateral/2025/ipi-ca-9-26-25-eng.pdf',
 'NRBROCH0925ENCAII', DATE '2025-09-26', 2, 'active',
 'pending',
 'MIXED AUTHORITY: duplicated agreement text plus unique plan info. Split into two rows if section-level authority becomes necessary.'),

-- 3 ---------------------------------------------------------------------------
-- Separate fee schedule from the CA: $5 late fee, "whichever is LESS", after
-- 15 days. Must never be merged with wireless fees.
('doc_vzw_dpa_2026',
 'Device Payment Agreement (Retail Installment Sale Agreement)',
 'device_financing_agreement',
 ARRAY['mobile_device_financing'],
 '/srv/documents/doc_vzw_dpa_2026.pdf',
 'https://ss7.vzw.com/is/content/VerizonWireless/Device-Payment-Agreement-Template',
 '06/08/2026', DATE '2026-06-08', 1, 'active',
 'pending',
 'Template with placeholder financial fields. State-specific provisions for DC, SD, IL.'),

-- 4 ---------------------------------------------------------------------------
('doc_fios_return_2022',
 'Fios Equipment Return Policy for Internet, TV and Home Phone',
 'return_policy',
 ARRAY['fios_internet','fios_tv','fios_home_phone'],
 '/srv/documents/doc_fios_return_2022.pdf',
 'https://www.verizon.com/content/dam/verizon/support/consumer/documents/fios-equipment-return-policy.pdf',
 'v.2022-2', DATE '2022-02-01', 2, 'active',
 'pending',
 'Oldest document in the corpus - staleness probe. Twelve-item exclusion list is Fios-specific and must never surface on a wireless ticket (gd-069).'),

-- 5 ---------------------------------------------------------------------------
-- effective_from is NULL on purpose: the document states no date in its body.
-- Guessing from the filename would make date filtering quietly wrong later.
('doc_fios_ca_home',
 'Verizon Customer Agreement (Fios Internet / TV / Home Phone)',
 'customer_agreement',
 ARRAY['fios_internet','fios_tv','fios_home_phone'],
 '/srv/documents/doc_fios_ca_home.pdf',
 'https://www.verizon.com/about/sites/default/files/Verizon-Customer-Agreement-Home-Aug31.pdf',
 NULL, NULL, 1, 'active',
 'pending',
 'NO EFFECTIVE DATE IN BODY - verify before relying on date filtering. Source for the $9 late fee and the 30-vs-60-day billing dispute contradiction (gd-051).')

ON CONFLICT (slug) DO UPDATE SET
  title          = EXCLUDED.title,
  doc_type       = EXCLUDED.doc_type,
  product_line   = EXCLUDED.product_line,
  storage_path   = EXCLUDED.storage_path,
  source_url     = EXCLUDED.source_url,
  version_label  = EXCLUDED.version_label,
  effective_from = EXCLUDED.effective_from,
  authority_tier = EXCLUDED.authority_tier,
  status         = EXCLUDED.status,
  notes          = EXCLUDED.notes,
  updated_at     = NOW();
  -- index_status deliberately NOT updated: re-running the seed must not reset a
  -- document that is already indexed back to pending.

-- ----------------------------------------------------------------------------
-- 2. Chunk tables
--
-- Two columns each, and that is deliberate:
--   doc_id       -> scored against gold_doc_ids, and the join key for filtering
--   section_path -> citation surface, and matches gold_sections for review
--
-- doc_id is on child_chunks as well as parent_documents even though it looks
-- redundant. Retrieval filters on children, because children are what carry the
-- embedding. Without doc_id here, every query would need child -> parent ->
-- documents, a two-hop join in the hot path. One denormalised FK avoids it.
-- ----------------------------------------------------------------------------
CREATE TABLE parent_documents (
  id UUID PRIMARY KEY,
  doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  content TEXT,
  section_path TEXT,
  metadata JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE child_chunks (
  id UUID PRIMARY KEY,
  parent_id UUID REFERENCES parent_documents(id) ON DELETE CASCADE,
  doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  product_line TEXT[] NOT NULL,
  section_path TEXT,
  content TEXT,
  embedding VECTOR(384),
  metadata JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

-- product_line is denormalised onto child_chunks deliberately: it is the only
-- predicate that runs on every single query, and keeping it on the table being
-- scanned avoids a join in the hot path.

CREATE INDEX idx_child_doc_id  ON child_chunks (doc_id);
CREATE INDEX idx_parent_doc_id ON parent_documents (doc_id);

-- Create a Fast similarity search on the embedding column using cosine distance
-- IVF Flat groups vectors into clusters, searches only relevant clusters
-- vector_cosine_ops ensures index works correctly for:
--    ORDER BY embedding <=> query_vector
-- WITH (lists = 100) -> split data into 100 clusters
-- Rule of thumb => List = sqdt(num of rows)

-- CREATE INDEX ON child_chunks
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

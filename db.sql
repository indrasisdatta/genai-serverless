CREATE TABLE parent_documents (
  id UUID PRIMARY KEY,
  content TEXT,
  metadata JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE child_chunks (
  id UUID PRIMARY KEY,
  parent_id UUID REFERENCES parent_documents(id) ON DELETE CASCADE,
  content TEXT,
  embedding VECTOR(1536),
  metadata JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Create a Fast similarity search on the embedding column using cosine distance 
-- IVF Flat groups vectors into clusters, searches only relevant clusters
-- vector_cosine_ops ensures index works correctly for: 
--    ORDER BY embedding <=> query_vector
-- WITH (lists = 100) -> split data into 100 clusters 
-- Rule of thumb => List = sqdt(num of rows)
CREATE INDEX ON child_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
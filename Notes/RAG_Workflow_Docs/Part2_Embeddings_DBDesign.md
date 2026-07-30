# RAG Workflow Documentation (Part 2)

## Database Design, pgvector, Embeddings & Retrieval

---

# Table of Contents

1. Database Design
2. Why PostgreSQL + pgvector?
3. Database Schema
4. Parent Documents Table
5. Child Chunks Table
6. Why Parent-Child Retrieval?
7. Embeddings
8. Query Embedding
9. SQL Queries Explained
10. Similarity Search
11. Complete Retrieval Flow
12. Why Not Search Parent Documents?
13. Performance Optimizations
14. Interview Explanations (1 min / 5 mins / 20 mins)

---

# 1. Database Design

Instead of storing the entire PDF in one table, the project stores information in **two separate tables**.

```
                    PDF
                     │
                     ▼
              Parent Documents
                     │
                     │ 1 : N
                     ▼
              Child Chunks
                     │
                     ▼
              Vector Embeddings
```

This design improves:

- Search accuracy
- LLM context quality
- Performance
- Scalability

---

# 2. Why PostgreSQL + pgvector?

Originally the project used **FAISS**.

```
PDF

↓

Embeddings

↓

FAISS Index
```

Advantages

- Very fast
- In-memory search

Disadvantages

- Hard to update
- Need to save/load index files
- Difficult for multiple users
- Doesn't store metadata efficiently

---

The project now uses PostgreSQL with pgvector.

```
PDF

↓

Embeddings

↓

PostgreSQL

↓

pgvector

↓

SQL Search
```

Advantages

- Persistent storage
- SQL queries
- Metadata filtering
- ACID transactions
- Easier backups
- Easier deployment
- Supports multiple sessions

---

# Why migrate from FAISS?

Old Flow

```
Upload

↓

Create FAISS

↓

Save Index

↓

Reload Index

↓

Search
```

New Flow

```
Upload

↓

Insert into PostgreSQL

↓

Search directly

↓

No file management
```

---

# 3. Database Schema

## parent_documents

```
+--------------------------------------+
| id (UUID)                            |
|--------------------------------------|
| content (TEXT)                       |
| metadata (JSONB)                     |
| created_at                           |
+--------------------------------------+
```

---

## child_chunks

```
+--------------------------------------+
| id (UUID)                            |
|--------------------------------------|
| parent_id (UUID FK)                  |
| content (TEXT)                       |
| embedding VECTOR(384)                |
| metadata (JSONB)                     |
| created_at                           |
+--------------------------------------+
```

Relationship

```
parent_documents

       1

       │

       │

       ▼

child_chunks

       N
```

---

# 4. parent_documents Table

Purpose

Store complete contextual information.

Example

```
Parent Chunk

----------------------------------------

Chapter 3 discusses vector databases.
Vector databases store embeddings.
Similarity search is performed...
```

No embeddings are stored here.

Why?

Because parent chunks are **too large** for efficient retrieval.

---

Columns

| Column | Purpose |
|---------|----------|
| id | Primary key |
| content | Parent text |
| metadata | Session information |
| created_at | Timestamp |

---

# 5. child_chunks Table

Purpose

Store searchable pieces.

Example

```
Parent

↓

Child 1

↓

Child 2

↓

Child 3
```

Each child has

- Parent ID
- Small text
- Embedding

---

Columns

| Column | Purpose |
|---------|----------|
| id | UUID |
| parent_id | Links to parent |
| content | Small text |
| embedding | VECTOR(384) |
| metadata | session_id |
| created_at | Timestamp |

---

# Why store embeddings only here?

Because

Small chunk

↓

Higher semantic accuracy

↓

Better search results

If we embedded the entire parent

```
1000 words

↓

1 embedding
```

Important details could be averaged out.

---

# 6. Parent-Child Retrieval

This is the most important concept.

Imagine this page.

```
Page

↓

Parent

↓

Child 1

Child 2

Child 3

Child 4
```

Searching

```
Child
```

gives

```
Parent
```

for the LLM.

---

Why?

Searching

```
Entire Page
```

↓

Low precision

Searching

```
Sentence
```

↓

Very high precision

Answering

```
Sentence
```

↓

No context

So the project combines both.

```
Search

↓

Child

↓

Answer

↓

Parent
```

Best of both worlds.

---

# Example

Question

```
What is vector search?
```

Top child

```
Vector search compares embeddings
using cosine similarity...
```

Parent

```
Chapter 5

Vector databases...

Embeddings...

Cosine similarity...

Applications...
```

LLM receives

Parent

instead of only one sentence.

---

# 7. Embeddings

What is an embedding?

A vector representation of text.

Example

```
"Dog"

↓

[0.13,
-0.82,
0.64,
...
384 values]
```

Another sentence

```
"Puppy"

↓

[0.11,
-0.79,
0.61,
...]
```

These vectors are close together.

---

# Why 384 Dimensions?

The project uses

```
all-MiniLM-L6-v2
```

which produces

```
384-dimensional vectors
```

Database

```
VECTOR(384)
```

must exactly match.

---

Earlier Error

```
expected 1536 dimensions

not 384
```

Reason

Database

```
VECTOR(1536)
```

Embedding Model

```
384
```

Mismatch.

---

Correct Design

```
Embedding Model

↓

384

↓

Database

VECTOR(384)
```

---

# Document Embedding

During upload

```
Child Chunk

↓

embed_documents()

↓

384 Vector
```

Batch processing

```
Chunk1

Chunk2

Chunk3

↓

One API Call
```

instead of

```
embed()

embed()

embed()
```

Much faster.

---

# Query Embedding

During chat

Question

```
Explain RAG
```

↓

embed_query()

↓

384 Vector

Same model

Same dimension

---

# Why use the same embedding model?

If

Documents

↓

Model A

Questions

↓

Model B

Vectors may not be comparable.

Always use the same model.

---

# 8. SQL Queries Explained

---

## Step 1

Generate query embedding.

```
Question

↓

Embedding

↓

384 values
```

---

## Step 2

Find similar child chunks.

SQL

```sql
SELECT parent_id
FROM child_chunks
WHERE metadata ->> 'session_id' = :session_id
ORDER BY embedding <=> :embedding::vector
LIMIT 5;
```

---

# WHERE metadata ->> 'session_id'

The metadata column is JSONB.

Example

```json
{
    "session_id": "abc123"
}
```

The operator

```
->>
```

extracts the value as text.

Equivalent to

```
metadata.session_id
```

---

Why?

Suppose

User A uploads

```
Book A
```

User B uploads

```
Book B
```

Without filtering

```
Search

↓

Book A

+

Book B
```

Wrong.

With

```
session_id
```

Each user searches only their own documents.

---

# ORDER BY embedding <=> :embedding

The operator

```
<=>
```

is provided by pgvector.

It calculates vector distance.

Smaller distance

↓

More similar.

Example

| Distance | Similarity |
|-----------|------------|
| 0.05 | Excellent |
| 0.12 | Very Good |
| 0.60 | Poor |

Sorting

```
ORDER BY
```

returns the closest vectors first.

---

# ::vector

SQLAlchemy sends

```
Python List
```

The database expects

```
VECTOR
```

So

```sql
:embedding::vector
```

casts the value.

---

# LIMIT 5

Suppose

```
1000 chunks
```

We don't want all.

We retrieve only

```
Top 5
```

Benefits

- Faster
- Less prompt size
- Lower token cost

---

# Duplicate Parent IDs

Imagine

```
Child 1

Parent A
```

```
Child 2

Parent A
```

```
Child 3

Parent B
```

Search returns

```
A

A

B
```

We convert to

```
A

B
```

using

```python
set(parent_ids)
```

This prevents duplicate context.

---

# Fetch Parent Documents

SQL

```sql
SELECT content
FROM parent_documents
WHERE id = ANY(:ids);
```

Meaning

```
Fetch

Parent A

Parent B

Parent C
```

using one SQL query.

---

Why ANY()?

Instead of

```sql
WHERE id = x

OR id = y

OR id = z
```

we pass an array.

Cleaner.

More efficient.

---

# 9. Complete Retrieval Flow

```
User Question

↓

embed_query()

↓

384 Vector

↓

child_chunks

↓

Similarity Search

↓

Top Parent IDs

↓

parent_documents

↓

Context

↓

LangChain

↓

LLM
```

---

# 10. Why Not Search Parent Documents?

Suppose

Parent

```
1000 words
```

Embedding

↓

Average meaning.

Small details disappear.

Instead

```
1000 words

↓

100 child chunks

↓

100 embeddings
```

Much more precise.

---

# 11. Performance Optimizations

## Batch Embeddings

Instead of

```
embed()

embed()

embed()
```

the project uses

```
embed_documents()
```

Advantages

- Faster
- Lower latency
- Better throughput

---

## Bulk Inserts

Instead of

```
INSERT

INSERT

INSERT
```

the project prepares arrays.

```
parent_rows

child_rows
```

and inserts together.

---

## Session Filtering

Only documents belonging to the current upload are searched.

No unnecessary scans.

---

## Parent Lookup

LLM receives fewer but richer documents.

Smaller prompts.

Better answers.

---

# 12. Retrieval Sequence Diagram

```
User

 │

 │ Question

 ▼

Generate Query Embedding

 │

 ▼

PostgreSQL (pgvector)

 │

 ▼

Search child_chunks

 │

 ▼

Top Parent IDs

 │

 ▼

Retrieve parent_documents

 │

 ▼

LangChain QA Chain

 │

 ▼

Groq LLM

 │

 ▼

Answer
```

---

# 13. Common Errors

## Wrong Vector Size

```
expected 1536

not 384
```

Reason

Database dimension doesn't match embedding model.

---

## Missing pgvector Extension

```
extension vector does not exist
```

Need

```sql
CREATE EXTENSION vector;
```

---

## Syntax Error

Wrong

```sql
\:embedding
```

Correct

```sql
:embedding
```

---

## Missing Session Filter

Without

```
WHERE session_id
```

the application searches documents from all users.

---

# 14. Interview Explanations

## 1-Minute Explanation

> The project stores large parent chunks and smaller child chunks separately. Only the child chunks receive embeddings because smaller chunks improve semantic search accuracy. During chat, the user's question is converted into a vector, PostgreSQL with pgvector retrieves the most similar child chunks, their parent documents are fetched, and those parent documents are passed to the LLM. This approach combines precise retrieval with rich context.

---

## 5-Minute Explanation

> Instead of storing one embedding for an entire document, the application first splits the document into parent chunks and then into child chunks. Child chunks are embedded using a 384-dimensional Sentence Transformer model and stored in the `child_chunks` table along with a reference to their parent. During retrieval, the user's question is embedded using the same model, and pgvector performs a nearest-neighbor search using the `<=>` operator. The application retrieves the corresponding parent documents, which provide enough surrounding context for the LLM to answer accurately. PostgreSQL also stores metadata such as `session_id`, allowing each upload to remain isolated.

---

## 20-Minute Explanation

> The application replaces the previous FAISS implementation with PostgreSQL and pgvector to simplify storage and improve scalability. During upload, parent chunks are created for contextual storage while child chunks are created for semantic indexing. The child chunks are embedded in batches using a 384-dimensional embedding model and inserted into the `child_chunks` table. The parent chunks are stored separately in `parent_documents`. When the user asks a question, `embed_query()` generates another 384-dimensional vector. A SQL query orders all child embeddings by vector distance using the `<=>` operator and returns the closest matches for the current `session_id`. Duplicate parent IDs are removed, and a second SQL query retrieves the corresponding parent documents. These parent documents are supplied to the LangChain QA chain, which then invokes the Groq LLM to generate the final answer. This architecture provides accurate retrieval, efficient database operations, and high-quality context for the language model while remaining easy to scale and maintain.

---

**End of Part 2**

**Next Part:** Chat API, Streaming, LangChain Pipeline, Prompt Engineering, Async Streaming, and End-to-End Response Generation.
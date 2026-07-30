# RAG Workflow Documentation (Part 1)
## Project Overview, Architecture, Startup Flow & Upload API

**Project:** Multi PDF Chat Application using LangChain + PostgreSQL (pgvector)

**Author:** Indrasis Datta

---

# Table of Contents

1. Introduction
2. What is RAG?
3. Project Architecture
4. Technologies Used
5. Project Structure
6. Application Startup Flow
7. Upload API Workflow
8. Parent-Child Retrieval Concept
9. Upload Sequence Diagram
10. Important Functions Explained
11. End-to-End Upload Flow
12. Interview Explanations (1 min / 5 mins / 20 mins)

---

# 1. Introduction

This application allows users to upload one or more PDF files and ask questions about their contents.

Instead of sending the entire PDF to the LLM every time, the application stores semantic embeddings inside PostgreSQL using the **pgvector** extension.

When a user asks a question:

- The question is converted into an embedding.
- The most relevant document chunks are retrieved.
- Only those chunks are sent to the LLM.
- The LLM generates an answer.
- The answer is streamed back to the client.

This architecture is called **Retrieval Augmented Generation (RAG).**

---

# 2. What is RAG?

Traditional LLM

```
User
   │
   ▼
Large Language Model
   │
   ▼
Answer
```

Problem:

The LLM only knows what it was trained on.

It cannot answer questions about:

- Internal company documents
- Uploaded PDFs
- Private knowledge
- Latest information

---

RAG

```
User Question
      │
      ▼
Generate Embedding
      │
      ▼
Vector Database
      │
      ▼
Relevant Documents
      │
      ▼
Large Language Model
      │
      ▼
Answer
```

Instead of relying only on memory, the LLM receives the relevant documents as additional context.

---

# 3. High-Level Architecture

```
                +----------------------+
                |      Client          |
                |  React / Postman     |
                +----------+-----------+
                           |
                           |
                 HTTP Request
                           |
                           ▼
               +----------------------+
               |      FastAPI         |
               +----------+-----------+
                          |
         +----------------+----------------+
         |                                 |
         ▼                                 ▼
  Upload API                       Chat API
         |                                 |
         ▼                                 ▼
 PDF Processing                 Generate Query Embedding
         |                                 |
         ▼                                 ▼
 Parent Splitter                 PostgreSQL (pgvector)
         |                                 |
         ▼                                 ▼
 Child Splitter                 Retrieve Parent Documents
         |                                 |
         ▼                                 ▼
 Embedding Model                   LangChain QA Chain
         |                                 |
         ▼                                 ▼
 PostgreSQL + pgvector           Groq Llama 3.1
                                           |
                                           ▼
                                 Streaming Response
```

---

# 4. Technologies Used

| Technology | Purpose |
|------------|---------|
| FastAPI | REST API |
| LangChain | Build Retrieval QA chain |
| HuggingFace Embeddings | Generate vector embeddings |
| PostgreSQL | Store parent documents |
| pgvector | Store embeddings & similarity search |
| SQLAlchemy | Database access |
| Groq | LLM inference |
| PyPDF2 | Read PDFs |
| StreamingResponse | Token streaming |
| asyncio | Async execution |

---

# 5. Project Structure

```
multipdf_chat/

│
├── main.py
│
├── helper.py
│
├── upload.py
│
├── query.py
│
├── models.py
│
├── prompts.py
│
└── requirements.txt
```

---

## main.py

Responsible for

- Creating FastAPI app
- Loading embedding model
- Creating database session
- Registering endpoints

Endpoints

```
POST /upload

POST /chat/stream
```

---

## helper.py

Contains most of the business logic.

Examples

- Reading PDF
- Splitting documents
- Generating embeddings
- Streaming responses
- LangChain chain creation

---

## upload.py

Receives uploaded files.

Calls helper methods.

Returns

```
session_id
```

which is later used during chat.

---

## query.py

Contains the non-streaming query implementation.

The streaming endpoint reuses most of this logic.

---

# 6. Application Startup Flow

When FastAPI starts,

the application creates shared objects.

```
Application Starts
        │
        ▼
Load Environment Variables
        │
        ▼
Create Embedding Model
        │
        ▼
Create Database Session Factory
        │
        ▼
Store in app.state
```

Example

```
app.state.embeddings

app.state.db
```

These are created **once** during startup.

Every request reuses them.

This is much faster than creating a new embedding model for every API call.

---

# Why use app.state?

Without app.state

```
Every Request

↓

Load Embedding Model

↓

Load Database

↓

Run Query
```

Very slow.

---

With app.state

```
Application Starts

↓

Load Once

↓

Every Request

↓

Reuse
```

Much better performance.

---

# 7. Upload API Workflow

Endpoint

```
POST /upload
```

Purpose

- Upload PDFs
- Generate embeddings
- Store documents
- Return session ID

---

## Overall Flow

```
Client
   │
   ▼
Upload PDF
   │
   ▼
Read PDF
   │
   ▼
Extract Text
   │
   ▼
Split into Parent Chunks
   │
   ▼
Split into Child Chunks
   │
   ▼
Generate Embeddings
   │
   ▼
Store Parent Documents
   │
   ▼
Store Child Chunks
   │
   ▼
Return Session ID
```

---

# Step 1

Receive Uploaded Files

```
POST /upload
```

FastAPI receives

```
List[UploadFile]
```

Example

```
book.pdf

notes.pdf

manual.pdf
```

---

# Step 2

Extract Text

Each PDF is read page by page.

```
PDF

↓

Page 1

↓

Page 2

↓

Page 3

↓

Merge

↓

Raw Text
```

Now we have one long string.

Example

```
"This is page one...

This is page two...

This is page three..."
```

---

# Step 3

Parent Splitter

Large documents cannot be embedded efficiently.

So we first split into larger chunks.

Example

```
Entire PDF

↓

Parent Chunk 1

↓

Parent Chunk 2

↓

Parent Chunk 3
```

Each parent chunk may contain several paragraphs.

Purpose

Keep enough context for the LLM.

---

# Step 4

Child Splitter

Each parent chunk is split again.

```
Parent Chunk

↓

Child 1

Child 2

Child 3

Child 4
```

These child chunks are much smaller.

These are the chunks that receive embeddings.

---

# Why Two Levels?

Instead of

```
Entire PDF

↓

Embedding
```

We use

```
PDF

↓

Parent

↓

Child

↓

Embedding
```

Reason

Small chunks improve search quality.

Large chunks improve answer quality.

The application combines both advantages.

---

# Step 5

Generate Embeddings

Every child chunk becomes a vector.

Example

```
Child Chunk

↓

Embedding Model

↓

[0.18,
-0.92,
0.47,
...
384 values]
```

The project uses a 384-dimensional embedding model.

---

# Batch Embedding

Instead of

```
for chunk

embed(chunk)
```

the application uses

```
embed_documents()

↓

Chunk1

Chunk2

Chunk3

Chunk4
```

All chunks are embedded together.

Benefits

- Faster
- Fewer API calls
- Better throughput

---

# Step 6

Prepare Database Records

Two collections are prepared.

```
Parent Rows
```

Contain

- UUID
- Content
- Metadata

---

```
Child Rows
```

Contain

- UUID
- Parent UUID
- Content
- Embedding
- Metadata

---

# Step 7

Bulk Insert

Instead of

```
INSERT

INSERT

INSERT

INSERT
```

the project prepares arrays and inserts them together.

Advantages

- Faster
- Fewer database round trips
- Better scalability

---

# Step 8

Commit

```
Commit Transaction
```

Everything is saved.

If anything fails,

```
Rollback
```

ensures partial data is not stored.

---

# 8. Parent-Child Retrieval Concept

Imagine a 200-page book.

Searching the whole book is inefficient.

Searching tiny chunks loses context.

Instead,

```
Book

↓

Parent

↓

Child
```

Search happens on

```
Child
```

Answer generation uses

```
Parent
```

This gives

- accurate retrieval
- complete context

---

# 9. Upload Sequence Diagram

```
User
 │
 │ Upload PDF
 ▼
FastAPI
 │
 ▼
Extract Text
 │
 ▼
Parent Splitter
 │
 ▼
Child Splitter
 │
 ▼
Embedding Model
 │
 ▼
Generate 384-D Vectors
 │
 ▼
PostgreSQL
 │
 ├──────────────► parent_documents
 │
 └──────────────► child_chunks
 │
 ▼
Return session_id
```

---

# 10. Important Functions (Upload Flow)

## upload_handler()

Responsibilities

- Validate uploaded files
- Read PDFs
- Generate session ID
- Call embedding pipeline
- Return success response

---

## get_pdf_text()

Responsibilities

- Read every page
- Extract text
- Merge pages
- Return a single string

---

## get_parent_child_splitters()

Creates

```
Parent Splitter

Child Splitter
```

Both use recursive splitting with different chunk sizes.

---

## generate_embedding()

This is the heart of the upload pipeline.

Responsibilities

- Create parent chunks
- Create child chunks
- Generate embeddings
- Build parent rows
- Build child rows
- Bulk insert into PostgreSQL
- Commit transaction

---

# 11. End-to-End Upload Flow

```
Upload PDF

↓

Extract Text

↓

Parent Splitter

↓

Child Splitter

↓

Generate Embeddings

↓

Create Parent Rows

↓

Create Child Rows

↓

Insert parent_documents

↓

Insert child_chunks

↓

Commit

↓

Return session_id
```

---

# 12. Interview Explanations

## 1-Minute Explanation

> The Upload API accepts one or more PDF files, extracts their text, splits the content into larger parent chunks and smaller child chunks, generates vector embeddings for the child chunks using a HuggingFace embedding model, and stores everything in PostgreSQL with pgvector. The parent chunks preserve context for the LLM, while the child chunk embeddings enable fast semantic search. Finally, the API returns a session ID that is used for future chat requests.

---

## 5-Minute Explanation

> When a user uploads a PDF, FastAPI receives the files and extracts text from every page. The text is first split into larger parent chunks and then into smaller child chunks. Embeddings are generated only for the child chunks because smaller chunks improve retrieval accuracy. Each child chunk maintains a reference to its parent chunk using `parent_id`. The application batches embedding generation for better performance and performs bulk inserts into PostgreSQL. The parent documents are stored in one table, while child chunks and their vectors are stored in another. A session ID is attached to every record so multiple uploaded document sets remain isolated. This prepares the knowledge base for efficient retrieval during chat.

---

## 20-Minute Explanation

> The upload process begins when the client sends a `POST /upload` request containing one or more PDF files. The `upload_handler()` function validates the files, extracts their text using `get_pdf_text()`, and creates a unique session ID. The text is then processed by `generate_embedding()`, which uses `get_parent_child_splitters()` to create two levels of chunks. Parent chunks retain larger sections of the document for context, while child chunks are optimized for semantic search. The embedding model converts each child chunk into a 384-dimensional vector using batch processing. Parent rows and child rows are prepared separately and inserted into the `parent_documents` and `child_chunks` tables using bulk SQL operations inside a transaction. If any step fails, the transaction is rolled back to maintain consistency. Once the transaction succeeds, the API returns the session ID, which is later used by the chat endpoint to retrieve only the relevant document set. This design improves retrieval quality, reduces database operations, and scales well for large document collections.

---

**End of Part 1**

**Next Part:** Database Design, pgvector, SQL Queries, Embeddings, and Parent-Child Retrieval in Detail.
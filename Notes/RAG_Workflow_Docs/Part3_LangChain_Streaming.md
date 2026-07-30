# RAG Workflow Documentation (Part 3)

# Chat API, LangChain Pipeline & Streaming Response

---

# Table of Contents

1. Chat API Overview
2. End-to-End Chat Flow
3. Streaming vs Normal Response
4. stream_user_input()
5. Query Embedding
6. Vector Search
7. Parent Document Retrieval
8. LangChain Pipeline
9. Prompt Template
10. ChatGroq
11. StreamingHandler
12. Async Queue
13. StreamingResponse
14. Complete Sequence Diagram
15. Error Handling
16. Interview Explanations (1 min / 5 mins / 20 mins)

---

# 1. Chat API Overview

Endpoint

```
POST /chat/stream
```

Request

```json
{
    "user_question": "Summarize in 10 bullet points",
    "session_id": "2d1690cd-2220-48ee-b4bf-cf7f387072c1"
}
```

Purpose

- Accept user's question
- Generate embedding
- Search relevant chunks
- Retrieve parent documents
- Pass documents to LLM
- Stream response back token by token

---

# 2. End-to-End Chat Flow

```
                    User

                      │

                      ▼

               POST /chat/stream

                      │

                      ▼

            Generate Query Embedding

                      │

                      ▼

         Search child_chunks (pgvector)

                      │

                      ▼

          Get Top Matching Parent IDs

                      │

                      ▼

          Fetch Parent Documents

                      │

                      ▼

             LangChain QA Chain

                      │

                      ▼

                  ChatGroq

                      │

          Tokens Generated

                      │

                      ▼

             StreamingHandler

                      │

                      ▼

             asyncio.Queue

                      │

                      ▼

           StreamingResponse

                      │

                      ▼

                   Client
```

---

# 3. Why Streaming?

Traditional API

```
User

↓

Wait...

↓

Wait...

↓

Wait...

↓

Complete Response
```

User waits until the entire answer is generated.

---

Streaming API

```
User

↓

H

↓

He

↓

Hel

↓

Hello

↓

Hello...
```

The user starts seeing the answer immediately.

Benefits

- Better user experience
- Faster perceived response
- Suitable for long answers

---

# 4. stream_user_input()

This is the heart of the chat workflow.

Responsibilities

- Generate query embedding
- Search PostgreSQL
- Retrieve parent documents
- Create LangChain chain
- Start background LLM execution
- Stream tokens to client

---

## Step 1

Generate Query Embedding

```
User Question

↓

embed_query()

↓

384-D Vector
```

Example

```
"What is RAG?"

↓

[0.23,
-0.11,
0.82,
...
384]
```

---

# Step 2

Search child_chunks

SQL

```sql
SELECT parent_id
FROM child_chunks
WHERE metadata ->> 'session_id' = :session_id
ORDER BY embedding <=> :embedding::vector
LIMIT 5;
```

Returns

```
Parent A

Parent B

Parent A

Parent C
```

Duplicates removed

↓

```
A

B

C
```

---

# Step 3

Retrieve Parent Documents

SQL

```sql
SELECT content
FROM parent_documents
WHERE id = ANY(:ids)
```

Returns

```
Document A

Document B

Document C
```

These become

```
input_documents
```

for LangChain.

---

# Why retrieve parents?

Searching child chunks gives

```
Small context
```

LLM needs

```
Larger context
```

Therefore

```
Search

↓

Child

↓

Answer

↓

Parent
```

---

# 5. LangChain Pipeline

After retrieving documents

the code creates

```
QA Chain
```

Pipeline

```
Parent Documents

↓

Prompt Template

↓

ChatGroq

↓

Output Parser

↓

Streaming Callback
```

---

# 6. Prompt Template

Current Prompt

```
Answer the question as detailed as possible
from the provided context.

If answer is not within the provided
content, just mention

"Content not available"

Context:
{context}

Question:
{question}

Answer:
```

---

LangChain fills

```
{context}
```

with retrieved parent documents.

Example

```
Context

Chapter 5...

Vector search...

Embeddings...
```

Question

```
Explain vector search
```

Prompt sent to LLM

```
Context

Chapter 5...

Vector search...

Embeddings...

Question

Explain vector search
```

---

# Why Prompt Templates?

Instead of manually building strings

```
Context +

Question +

Instructions
```

LangChain automatically constructs the prompt.

Advantages

- Reusable
- Cleaner
- Easy to modify

---

# 7. ChatGroq

The project uses

```
ChatGroq
```

Model

```
llama-3.1-8b-instant
```

Configuration

```
streaming=True

callbacks=[handler]
```

Streaming enabled means

instead of returning

```
Entire answer
```

it returns

```
Token

Token

Token

Token
```

---

# 8. load_qa_chain()

The project creates

```
load_qa_chain()
```

Chain Type

```
stuff
```

Meaning

All retrieved documents are concatenated.

```
Doc1

+

Doc2

+

Doc3

↓

One Prompt
```

Simple

Fast

Suitable for small context.

---

Other chain types

```
map_reduce

refine

map_rerank
```

are useful for larger document collections.

---

# 9. Background Execution

Instead of waiting

the project starts

```
chain.ainvoke()
```

using

```python
asyncio.create_task(...)
```

Flow

```
Background Thread

↓

LLM Generates Tokens

↓

Main Thread

↓

Streams Tokens
```

Advantages

- Non-blocking
- Efficient
- Responsive

---

# 10. StreamingHandler

StreamingHandler extends

```
AsyncCallbackHandler
```

Responsibilities

- Receive tokens
- Store tokens
- Signal completion

Methods

```
on_llm_new_token()

on_llm_end()

on_llm_error()
```

---

# on_llm_new_token()

Every generated token

```
Hello

,

world
```

arrives here.

The handler pushes each token into

```
asyncio.Queue
```

Example

```
Queue

↓

Hello

↓

,

↓

world
```

---

# on_llm_end()

When generation completes

```
Queue

↓

None
```

is inserted.

The streaming loop exits.

---

# on_llm_error()

If something fails

```
Error message

↓

Queue

↓

None
```

Client immediately receives the error.

---

# 11. asyncio.Queue

Think of it as a pipe.

```
LLM

↓

Queue

↓

Streaming Loop

↓

Browser
```

Producer

```
ChatGroq
```

Consumer

```
StreamingResponse
```

Both run independently.

---

# Queue Visualization

```
LLM

↓

Token

↓

Queue

↓

Token

↓

Queue

↓

Token

↓

Queue

↓

Client
```

This prevents blocking.

---

# 12. Buffering

The project doesn't send every token immediately.

Instead

```
Token

↓

Buffer

↓

20 Characters

↓

Yield
```

Example

Tokens

```
Hello

,

 it

's

 nice
```

Buffer

```
Hello, it's nice
```

Yield

```
Hello, it's nice
```

Advantages

- Fewer network packets
- Better rendering
- Reduced overhead

---

# Common Bug

Incorrect

```
yield

↓

buffer not cleared

↓

duplicate output
```

Correct

```
yield

↓

buffer=""

↓

continue
```

Final buffer should be yielded **after** the loop finishes.

---

# 13. StreamingResponse

FastAPI

```
StreamingResponse()

↓

Consumes Generator

↓

Writes HTTP Response
```

Unlike

```
return response
```

StreamingResponse sends

```
Chunk

Chunk

Chunk

Chunk
```

until

```
None
```

is received.

---

# 14. Complete Sequence Diagram

```
User

 │

 │ POST /chat/stream

 ▼

FastAPI

 │

 ▼

embed_query()

 │

 ▼

384-D Vector

 │

 ▼

PostgreSQL

 │

 ▼

Top Child Chunks

 │

 ▼

Parent IDs

 │

 ▼

Parent Documents

 │

 ▼

PromptTemplate

 │

 ▼

ChatGroq

 │

 ▼

StreamingHandler

 │

 ▼

asyncio.Queue

 │

 ▼

StreamingResponse

 │

 ▼

Browser/Postman
```

---

# 15. Error Handling

## No Matching Documents

```
No parent IDs

↓

Return

"No relevant data"
```

---

## Database Failure

```
SQL Error

↓

Exception

↓

Rollback

↓

Close Session
```

---

## LLM Error

```
Groq API Error

↓

on_llm_error()

↓

Queue

↓

Streaming Ends
```

---

## Client Disconnect

StreamingResponse automatically stops when the connection closes.

No unnecessary token generation.

---

# 16. Performance Optimizations

## Reuse Embedding Model

Loaded once

```
app.state.embeddings
```

---

## Reuse Database Factory

Loaded once

```
app.state.db
```

---

## Batch Retrieval

Retrieve

```
Top 5
```

instead of

```
Entire Database
```

---

## Background Task

```
LLM

↓

Background

↓

Main Thread Streams
```

No blocking.

---

## Buffering

Instead of

```
H

e

l

l

o
```

Send

```
Hello
```

Better performance.

---

# 17. Complete Chat Workflow

```
User Question

↓

Generate Query Embedding

↓

Search child_chunks

↓

Retrieve Parent IDs

↓

Retrieve Parent Documents

↓

PromptTemplate

↓

ChatGroq

↓

StreamingHandler

↓

Queue

↓

StreamingResponse

↓

Browser
```

---

# 18. Interview Explanations

## 1-Minute Explanation

> When a user sends a question to `/chat/stream`, the application first converts the question into an embedding using the same embedding model that was used during upload. PostgreSQL with pgvector searches the most similar child chunks and returns their parent document IDs. The parent documents are passed to a LangChain QA chain that uses a prompt template and the Groq Llama 3.1 model. Since streaming is enabled, the LLM generates tokens incrementally, which are captured by `StreamingHandler` and sent back to the client through FastAPI's `StreamingResponse`.

---

## 5-Minute Explanation

> The chat workflow begins by generating a 384-dimensional embedding for the user's question. A pgvector similarity search retrieves the top matching child chunks while filtering by `session_id`. Duplicate parent IDs are removed, and the corresponding parent documents are fetched from PostgreSQL. These documents become the context supplied to a LangChain `load_qa_chain()` with the `stuff` chain type. A `PromptTemplate` combines the context and the user's question into a single prompt that is sent to Groq's `llama-3.1-8b-instant` model. Because streaming is enabled, LangChain invokes the `StreamingHandler` callback for every generated token. The callback pushes tokens into an `asyncio.Queue`, while `StreamingResponse` asynchronously consumes the queue and streams buffered text chunks to the client. This architecture provides low latency, efficient retrieval, and a responsive user experience.

---

## 20-Minute Explanation

> The `/chat/stream` endpoint orchestrates the complete Retrieval-Augmented Generation pipeline. After receiving the user's question and session ID, the application generates a query embedding using the same Sentence Transformer model that created the document embeddings. This guarantees that both query and document vectors exist in the same semantic space. PostgreSQL with pgvector performs a nearest-neighbor search using the `<=>` operator and returns the most relevant child chunks. Because child chunks only exist for retrieval purposes, the application extracts their unique `parent_id` values and retrieves the corresponding parent documents, preserving broader context for the LLM. LangChain's `load_qa_chain()` with the `stuff` chain type concatenates the retrieved documents into a single prompt using a `PromptTemplate`. The Groq-hosted `llama-3.1-8b-instant` model processes this prompt with streaming enabled. Every generated token triggers `on_llm_new_token()` in the custom `StreamingHandler`, which immediately places the token into an `asyncio.Queue`. Meanwhile, FastAPI's `StreamingResponse` continuously reads from the queue, buffers tokens into readable chunks, and sends them to the client until `on_llm_end()` inserts a `None` sentinel value. This asynchronous producer-consumer architecture keeps the API responsive while providing real-time streaming output to the user.

---

## Summary

The chat pipeline can be summarized as:

```
User Question
      │
      ▼
Generate Query Embedding
      │
      ▼
Search Child Chunks (pgvector)
      │
      ▼
Retrieve Parent Documents
      │
      ▼
Build Prompt
      │
      ▼
Groq LLM
      │
      ▼
StreamingHandler
      │
      ▼
asyncio.Queue
      │
      ▼
StreamingResponse
      │
      ▼
User
```

---

### My understanding of streaming API /stream/query:

1) The API should return `StreamingResponse` (FastAPI provided)
2) Controller function is async, should have -> `yield buffer` (or yield "No data") 
3) `StreamingHandler` extends LangChain's AsyncCallbackHandler - `on_llm_new_token`, `on_llm_error`, `on_llm_end` 
   (pushes to asyncio.queue, streams data until None is received)
4) docs - retrieved from parent_documents DB after doing similarity search on child_chunks 
5) PromptTemplate configuration 

    ```template = """Answer the question based on context. Question: {question}\n, Context: {context} """
    prompt = PromptTemplate(template=template, input_variables=[context, question])

6) ChatGroq object 
      ```
      handler = StreamingHandler()
      model = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=groq_api_key,
            streaming=True,
            callbacks=[handler]
      )
7) `load_qa_chain` imported from langchain.chains 
    
    ```
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)     


8) Invoke chain asyncronously - When chain runs, output is pushed to asyncio.queue
    ```
    asyncio.create_task(
        chain.ainvoke({
            "input_documents": docs,
            "question": question
        )
    )

    Although we never explicitly pass context, `load_qa_chain(chain_type="stuff")` constructs it from input_documents and injects it into your PromptTemplate automatically. That's why your prompt works with {context} even though your application code only supplies input_documents.
    
9) Buffering logic - instead of returning token by token, store 20 tokens in buffer and return in bulk

    ```buffer = ""
    while True:
        token = await handler.queue.get()
        if token is None:
            break 
        // Not sure??
        if not token or token.isspace():
            continue;
        buffer += token 
        if len(buffer) >= 20:
            yield buffer 
            buffer = ""
    
    if buffer:
        yield buffer 

---

**End of Part 3**

**Next Part:** Deep Code Walkthrough, Design Decisions, Interview Questions, Troubleshooting, Improvements, and Future Enhancements.
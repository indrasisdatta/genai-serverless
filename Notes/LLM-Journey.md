## LLM Journey: ANN → Transformers

## References

* YouTube (CampusX): https://youtu.be/8fX3rOjTloc
* Blog: *From RNNs to LLMs: A Journey through Sequential Modeling in NLP*
  https://medium.com/@harsuminder/from-rnns-to-llms-a-journey-through-sequential-modeling-in-nlp-d42de5eb2cb9
* Seq2Seq Models:
  https://learnwith.campusx.in/blog/seq2seq-models-tracing-the-ai-revolution-from-rnn-to-gpt-3

---

## Evolution Flow

ANN → RNN → LSTM/GRU → Seq2Seq → Attention → Transformers

---

## 1️. Artificial Neural Networks (ANN)

### Idea

* Feedforward network (no sequence awareness)

### Limitations

* No memory → treats words independently
* No order understanding
* Weak for NLP context

### Example

"The apple was red"
→ "red" not linked to "apple"

---

## 2. Recurrent Neural Networks (RNN)

### Idea

* Processes sequence step-by-step
* Maintains hidden state (memory)

### Flow

x1 → h1 → h2 → h3 → ... → hn

### Pros

* Handles variable-length input
* Captures sequence order

### Limitations

* Vanishing gradient
* Poor long-term memory
* Slow (sequential processing)

---

## 3. LSTM & GRU

### Idea

* Introduce gates to control memory flow

### Gates (LSTM)

* Forget Gate → what to drop
* Input Gate → what to store
* Output Gate → what to expose

### Pros

* Better long-term memory
* Mitigates vanishing gradient

### Limitations

* Still sequential
* Computationally heavy

---

## 4. Seq2Seq (Encoder–Decoder)

### Idea

* Encoder compresses input
* Decoder generates output

### Flow

Input → Encoder → [Context Vector] → Decoder → Output

### Use Case

* Machine Translation

### Limitation

* Bottleneck → all information in single vector

---

## 5️. Attention Mechanism

### Idea

* Decoder attends to all encoder hidden states

### Formula (intuition)

Context = Σ (attention weights × hidden states)

### Pros

* Removes bottleneck
* Focus on relevant words

### Example

"The bank of the river"
→ Focus on "river" to interpret "bank"

### Limitation

* Still built on RNN → slow

---

## 6️. Transformers

### Paper

* Attention Is All You Need

### Core Idea

* Remove RNN → use Self-Attention

### Key Components

* Self-Attention
* Multi-Head Attention
* Positional Encoding
* Feedforward Layers

### Flow

Input Embedding + Positional Encoding
↓
Multi-Head Self-Attention
↓
Feed Forward Network
↓
(Stacked Layers)

### Pros

* Parallel processing (no RNN bottleneck)
* Self-attention - Captures global context
* Scales well

### Limitations

* High compute cost
* O(n²) attention complexity

---

## 7️⃣ Transfer Learning

Before LLMs, if you wanted a model to "Summarize Legal Documents," you had to train it from scratch on domain-specific data. This was expensive and often failed due to lack of large datasets and poor general language understanding.

Transfer Learning changed the game into a two-step approach:

1. **Pre-training (The Generalist)**  
   A Transformer is trained on a massive, general dataset (e.g., Wikipedia, Common Crawl). The model learns language patterns—grammar, facts, reasoning, and some coding ability.

2. **Fine-tuning (The Specialist)**  
   The pre-trained model is adapted using domain-specific data (e.g., medical or legal text). Since it already understands language, it requires much less data and training to specialize.

> Note: In modern LLMs, fine-tuning is often complemented or replaced by prompting, instruction tuning, and RLHF.
---

## 8️⃣ Transformers → LLMs Evolution

### Built on Transformers

---

### Step 1: Pretraining (Transfer Learning - Generalist)

- Train on massive corpus (Wikipedia, Common Crawl)
- Learns grammar, facts, reasoning

---

### Step 2: Fine-tuning (Task-Specific)

- Adapt model for specific tasks (classification, QA, etc.)
- Uses labeled datasets

---

### Step 3: Instruction Tuning

- Train model on instruction–response pairs
- Examples:
  - "Summarize this text"
  - "Translate to French"

**Goal**
- Make model follow human instructions (not just predict next word)

---

###  Step 4: RLHF (Reinforcement Learning from Human Feedback)

- Humans rank multiple model outputs
- Train a reward model
- Optimize model to generate better responses

**Goal**
- Improve:
  - Helpfulness
  - Safety
  - Alignment with human intent

---

### Step 5: Chat-based LLMs (e.g., ChatGPT)

- Combines:
  - Pretraining
  - Instruction tuning
  - RLHF

- Optimized for:
  - Conversations
  - Multi-turn context
  - Assistant-like behavior

---

### Model Types

- Autoregressive → GPT  
- Bidirectional → BERT  

---

### Capabilities

- Text generation  
- Q&A, summarization  
- Code generation  
- Reasoning  

---

## ⚡ Final Summary

ANN         → No context
RNN         → Short memory
LSTM/GRU    → Better memory
Seq2Seq     → Translation
Attention   → Focus mechanism
Transformer → Parallel + scalable
LLMs        → General intelligence layer

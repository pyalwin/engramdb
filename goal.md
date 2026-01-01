Here is the structured One-Pager for **EngramDB**, focusing on the Problems, Solution, and Benchmarking strategy.

---

# **Project: EngramDB**
### *Solving the "Amnesia" of Artificial Intelligence*

### **I. The Unsolved Problems of LLM Memory**

Current Large Language Models (LLMs) are brilliant "Processors" but terrible "Hard Drives." They suffer from three fundamental memory failures that current tools (Vector RAG) cannot solve.

**1. The "Context Window" Trap (Short-Term Amnesia)**
* **Problem:** LLMs have a fixed limit on how much text they can read at once. If a conversation exceeds this limit, the beginning is simply deleted.
* **Current Band-aid:** Summarization. (But summarizing a 50-page legal contract into 1 paragraph loses critical details).
* **Result:** The AI "forgets" specific instructions or facts from earlier in the project.

**2. The "Fuzzy Logic" Failure (Hallucination)**
* **Problem:** Vector databases retrieve data based on "vibe" (semantic similarity), not exact facts.
* **Example:** If you ask "Who is the CEO of Apple?", a vector DB might return a document about Steve Jobs because it is "highly similar" to the query, even though Tim Cook is the current CEO. The LLM then hallucinates an answer based on the wrong retrieved context.
* **Result:** High confidence, incorrect answers.

**3. The "Multi-Hop" Blindness (Reasoning Failure)**
* **Problem:** LLMs cannot connect two distant facts unless they appear in the *same* retrieved chunk.
* **Example:**
    * *Chunk A:* "Alice is Bob's sister."
    * *Chunk B:* "Bob lives in Paris."
    * *Query:* "Does Alice have any family in France?"
    * *Failure:* The Vector DB retrieves Chunk A or Chunk B, but rarely both together because they don't share keywords with the query. The AI fails to make the hop (Alice -> Bob -> Paris -> France).

---

### **II. Our Solution: EngramDB**

We are building **EngramDB**, an embedded "Associative Memory" engine. It moves beyond simple text storage to create a structured "Knowledge Web."

**The Core Innovation: The "Super Node"**
Instead of choosing between a Graph Database and a Vector Database, we fuse them. Every memory (Engram) is stored with three layers of data:

1.  **The Content Layer (Text):** Stores the raw fact. *(e.g., "Bob lives in Paris")*
2.  **The Vector Layer (Embedding):** Allows fuzzy search. *(e.g., Finds this note when you ask about "Bob's location")*
3.  **The Graph Layer (Synapse):** Stores the logic. *(e.g., Hard-links this note to "France" and "Bob")*



**The Workflow:**
When a query comes in, we use **"Hybrid Traversal"**:
1.  **Anchor:** Use the Vector to land on the most relevant node (The entry point).
2.  **Traverse:** Use the Graph connections to "walk" to related facts, gathering context that the Vector search missed.
3.  **Answer:** Feed this rich, interconnected context to the LLM.

---

### **III. Benchmarking: How We Measure Success**

We cannot just say "it feels better." We must prove it. We will benchmark EngramDB against a standard RAG system (using just Pinecone/Chroma) using these three metrics:

#### **1. The "Needle in a Haystack" Test (Precision)**
* **The Test:** We insert a specific, obscure fact (the "Needle") into a massive dataset of random text (the "Haystack").
* **The Query:** We ask a question that requires finding that exact fact.
* **Metric:** **Recall Rate %.** (Does EngramDB find it more often than standard Vector search?)

#### **2. The "Multi-Hop" Reasoning Test (The "Graph" Advantage)**
* **The Test:** We feed the system disjointed facts:
    * *Fact 1:* "The red key opens the blue door."
    * *Fact 2:* "The treasure is behind the blue door."
* **The Query:** "Which key do I need to get the treasure?"
* **Metric:** **Success Rate %.** Standard Vector RAG usually fails this (0-20% success). EngramDB should score >80% because of the graph link.

#### **3. The "Update" Test (Data Integrity)**
* **The Test:** We tell the system "The CEO is Bob." Later, we tell it "The CEO is now Alice."
* **The Query:** "Who is the CEO?"
* **Metric:** **Accuracy %.** Vector DBs often return *both* answers (Bob and Alice) causing confusion. EngramDB should use a timestamped edge to know Alice is the *current* relationship.

---

### **IV. Technical Stack (Alpha Version)**

* **Engine:** **DuckDB** (Single-file, embedded SQL & Vector & JSON).
* **Language:** Python (Middleware logic).
* **Hardware:** Runs on a standard laptop (No GPU required for the DB itself).

**Goal:** Prove that a structured "Graph+Vector" memory outperforms a raw "Vector-only" memory on complex reasoning tasks.

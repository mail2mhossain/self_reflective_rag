A. Data Ingest
01. Calculating parent and child chunk sizes with overlap value
02. Splitting a file into parent chunks with parent_id and file_id and stored in key-pair DB
03. Each parent chunk will be splitted into child chunks and child chunks will be stored in vector db
04. QA pairs will be generated from each parent chunk and will be stored in vector db

B. Retriever
01. User query will be searched in both child and qa vector db.
02. parent_id will be picked from metadata of searched results 
03. Re-ranked both vector db search results and picked top 5
04. parent chunk will be collected from  key-pair DB based on parent_id

C. Self Reflective RAG
01. RAG with Query transformation
02. RAG with Multi-Query
03. RAG with Query Decomposition


## Building a Self-Reflective RAG Pipeline: From Smart Chunking to Insightful Retrieval


| Part  | Working title                                    | Scope                                                                                                                                                             | Why it deserves its own part                                                |
| ----- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **1** | *Smart Chunking & Storage Fundamentals*          | • How to size parent vs child chunks<br>• Overlap heuristics (10 – 15 %)<br>• Writing chunks to a key-value store and a vector DB                                 | Gives readers the mental model for everything that follows.                 |
| **2** | *Dual-Store Retrieval: Child, QA and Re-ranking* | • Querying child and QA indexes<br>• Metadata-based parent lookup<br>• Hybrid / re-rank logic (top-k)                                                             | Shows how the ingestion choices pay off at retrieval time.                  |
| **3** | *Self-Reflective RAG Techniques*                 | • Query transformation<br>• Multi-query expansion<br>• Query decomposition                                                                                        | Isolates the “reflective” ideas so they don’t get lost in plumbing details. |
| **4** | *End-to-End Walk-through, Evaluation & Scaling*  | • Wiring the pieces together<br>• Example notebook / API flow<br>• Offline evaluation, latency tips<br>• Scaling options (Redis vs Elasticsearch, batching, GPUs) | Gives a concrete finish and answers “What’s next?”                          |

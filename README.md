Query
   ↓
Query embedding
   ↓
Cosine similarity with document embeddings
   ↓
Top K documents
   ↓
Split each document into sentences
   ↓
Rank sentences by keyword overlap
   ↓
Return best sentence


After retrieving the top documents using cosine similarity over precomputed embeddings, I implemented a secondary sentence-level ranking mechanism.
Each document is segmented into sentences, and each sentence is scored based on query-term overlap.
The sentence with the highest query-term match score is selected and displayed as the most relevant snippet from that document

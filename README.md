## Example application of RAG using LangChain framework

Code losely based on the following tutorials:
- https://python.langchain.com/docs/use_cases/question_answering/
- https://python.langchain.com/docs/use_cases/question_answering/vector_db_retrieval
- https://github.com/pixegami/langchain-rag-tutorial

### First, create a vector store from the documents

`python create_database.py`

This uses the OpenAI embedding model to create a vector store from the documents. The raw documents are sources from `data/` directory.
The created vector store is saved as `chroma` in the current directory.

### Then, query the vector store

`python query_data.py --question "Who is Sherlock Holmes' nemesis?"`

This uses the LangChain framework to query the vector store.

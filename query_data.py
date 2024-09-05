import argparse
import os
from dotenv import load_dotenv

import openai

# from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# from langchain_openai import OpenAIEmbeddings

CHROMA_PATH = "./chroma"

PROMPT_TEMPLATE = """
Answer the question based on the context below. Use up to 10 sentences, if needed, but not more..
If the context does not contain the answer, say "I don't know".

Context: {context}
"""

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def main():

    # create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True,
                        help="Question to ask the model")
    args = parser.parse_args()
    query = args.question

    embeddings = OpenAIEmbeddings()
    # create vector store
    vector_store = Chroma(persist_directory=CHROMA_PATH, 
                          embedding_function=embeddings)
    
    # search the DB
    results = vector_store.similarity_search_with_relevance_scores(query, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print("Unable to find matching results")
        return
    
    context_text = "\n\n--\n\n".join([doc.page_content for doc, _score in results])

    # create retriever
    retriever = vector_store.as_retriever()

    # create prompt
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=context_text, question=query)
    # print(prompt)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_TEMPLATE),
            ("user", "{input}"),
            # ("context", "{context}")
        ]
    )
    print("-"*40); print(" PROMPT "); print("-"*40)
    print(prompt)
    print("-"*80)

    # model
    llm = ChatOpenAI(model="gpt-4o-mini")

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    
    response = rag_chain.invoke({"input": args.question, "context": context_text})
    response_text = response["answer"]

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    print("-"*40); print(" REPLY "); print("-"*40)
    formatted_response = f"{response_text}\n\n---\n\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
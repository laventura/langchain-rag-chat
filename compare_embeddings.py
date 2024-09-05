import openai

from dotenv import load_dotenv
import os

from langchain_community.embeddings import OpenAIEmbeddings
from langchain.evaluation import load_evaluator

# load env variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    # get embeddings
    embeddings = OpenAIEmbeddings()
    print(embeddings)

    words = ["Sherlock Holmes", "Conan Doyle", "Lupin", "Maurice Leblanc"]

    v1 = embeddings.embed_query(words[0])
    v2 = embeddings.embed_query(words[1])
    v3 = embeddings.embed_query(words[2])
    v4 = embeddings.embed_query(words[3])

    evaluator = load_evaluator("pairwise_embedding_distance")
    # evaluator.evaluate_string_pairs(prediction=v1, prediction_b=v2)
    res = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"v1 {words[0]} v2 {words[1]} res {res}")

    res = evaluator.evaluate_string_pairs(prediction=words[2], prediction_b=words[3])
    print(f"v3 {words[2]} v4 {words[3]} res {res}")

    res = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[3])
    print(f"v1 {words[0]} v4 {words[3]} res {res}")

if __name__ == "__main__":
    main()
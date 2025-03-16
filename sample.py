import streamlit as st
import pickle
import torch
from sentence_transformers import SentenceTransformer, util

# Load precomputed embeddings and sentences
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit app
def main():
    st.title("Research Paper Recommendation System")
    
    # Input for paper title
    paper_title = st.text_input("Enter the title of a paper you like:")

    # Recommend button
    if st.button("Recommend"):
        if paper_title.strip() == "":
            st.warning("Please enter a valid paper title.")
        else:
            # Compute cosine similarity
            input_embedding = model.encode(paper_title)
            cosine_scores = util.cos_sim(embeddings, input_embedding)

            # Get the top 5 similar papers
            top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)

            st.write("### Recommended Papers:")
            for i in top_similar_papers.indices.flatten():
                st.write(f"- {sentences[i.item()]}")

if __name__ == "__main__":
    main()

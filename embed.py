# Sample code for embedding content using the Gemini API and gemma-300m local model

def test_embed_content(self):
    # [START embed_content]
    from google import genai
    from google.genai import types

    client = genai.Client()
    text = "Hello World!"
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=10),
    )
    print(result.embeddings)
    # [END embed_content]

def test_batch_embed_contents(self):
    # [START batch_embed_contents]
    from google import genai
    from google.genai import types

    client = genai.Client()
    texts = [
        "What is the meaning of life?",
        "How much wood would a woodchuck chuck?",
        "How does the brain work?",
    ]
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=10),
    )
    print(result.embeddings)
    # [END batch_embed_contents]

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300m")

query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]
query_embeddings = model.encode_query(query)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# (768,) (4, 768)

# Compute similarities to determine a ranking
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.3011, 0.6359, 0.4930, 0.4889]])


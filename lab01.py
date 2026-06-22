import gensim.downloader as dl
from scipy.spatial.distance import cosine

print("Loading Word2Vec model...")
model = dl.load("word2vec-google-news-300")
print("Model loaded successfully")

print(f"First 10 dimensions of 'king' vector:\n\n{model['king'][:10]}")

print("Top 10 words most similar to 'king':\n")
for word, similarity in model.most_similar("king"):
    print(f"{word}: {similarity:.4f}")

print("Analogy: 'king' + 'woman' - 'man' ≈ ?")
for word, sim in model.most_similar(positive=["king", "woman"], negative=["man"], topn=1):
    print(f"{word} (Similarity: {sim:.4f})")

print("Analogy: 'paris' + 'italy' - 'france' ≈ ?\n")
for word, similarity in model.most_similar(positive=["paris", "italy"], negative=["france"]):
    print(f"{word}: {similarity:.4f}")

print("Analogy: 'walking' + 'swimming' - 'walk' ≈ ?\n")
for word, similarity in model.most_similar(positive=["walking", "swimming"], negative=["walk"]):
    print(f"{word}: {similarity:.4f}")

similarity = 1 - cosine(model["king"], model["queen"])
print(f"Cosine similarity between 'king' and 'queen' is {similarity:.4f}")

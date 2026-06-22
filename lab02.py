import gensim.downloader as dl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("Loading Word2Vec model...")
model = dl.load("word2vec-google-news-300")
print("Model loaded successfully")

words = ["computer", "internet", "software", "hardware", "keyboard", "mouse", "server", "network", "programming", "database"]
vectors = [model[word] for word in words]
reduced = PCA(n_components=2).fit_transform(vectors)

print("Top 5 words similar to 'computer':\n")
for word, score in model.most_similar("computer", topn=5):
    print(f"{word}: {score:.4f}")

plt.figure(figsize=(8, 6))
for word, coord in zip(words, reduced):
    plt.scatter(coord[0], coord[1])
    plt.annotate(word, coord)

plt.title("PCA Visualization of Technology Word Embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
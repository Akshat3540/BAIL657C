import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

medical_corpus = [
    "The patient was diagnosed with diabetes and hypertension.",
    "MRI scans reveal abnormalities in the brain tissue.",
    "The treatment involves antibiotics and regular monitoring.",
    "Symptoms include fever, fatigue, and muscle pain.",
    "The vaccine is effective against several viral infections.",
    "Doctors recommend physical therapy for recovery.",
    "The clinical trial results were published in the journal.",
    "The surgeon performed a minimally invasive procedure.",
    "The prescription includes pain relievers and anti-inflammatory drugs.",
    "The diagnosis confirmed a rare genetic disorder."
]

processed_corpus = [sentences.lower().split() for sentences in medical_corpus]
model = Word2Vec(processed_corpus, min_count=1, epochs=50)

words = model.wv.index_to_key
reduced = TSNE(n_components=2).fit_transform(model.wv.vectors)

plt.figure(figsize=(10, 8))
for word, coord in zip(words, reduced):
    plt.scatter(coord[0], coord[1])
    plt.text(coord[0], coord[1], word)

plt.title("Word Embeddings Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()

for input_word in ["treatment", "vaccine"]:
    print(f"Words similar to '{input_word}':\n")
    for word, similarity in model.wv.most_similar(input_word, topn=5):
        print(f"{word}\t({similarity:.2f})")
import gensim.downloader as dl

model = dl.load("glove-wiki-gigaword-100")
seed = "freedom"

sel = []
for w, score in model.most_similar(seed, topn=5):
    sel.append(w)

print(
    f"In a world defined by {seed}, "
    f"people found themselves surrounded by concepts like {', '.join(sel[:-1])}, and {sel[-1]}. "
    f"These ideas shaped the way they thought, acted, and dreamed. Every step forward in their journey reflected the essence of '{seed}', "
    f"bringing them closer to understanding the true meaning of {sel[0]}."
)
import gensim.downloader as dl
from transformers import pipeline

model = dl.load("glove-wiki-gigaword-100")
orig_prompt = "Explain benefits of exercise"

enriched = []
for word in orig_prompt.split():
    if word.lower() in model:
        for w, score in model.most_similar(word.lower(), topn=2):
            enriched.append(w)
enr_prompt = orig_prompt + " " + " ".join(enriched)

generator = pipeline("text2text-generation", model="google/flan-t5-base")
orig_out = generator(orig_prompt, max_length=100)[0]["generated_text"]
enr_out = generator(enr_prompt, max_length=100)[0]["generated_text"]

print("Original:", orig_out)
print("\nEnriched:", enr_out)

print("Original Detail:", len(orig_out.split()))
print("Enriched Detail:", len(enr_out.split()))

w1 = [w for w in orig_prompt.lower().split() if w in model]
w2 = [w for w in orig_out.lower().split() if w in model]
w3 = [w for w in enr_out.lower().split() if w in model]

print("Original Relevance:", model.n_similarity(w1, w2))
print("Enriched Relevance:", model.n_similarity(w1, w3))
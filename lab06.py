from transformers import pipeline

classifier = pipeline("sentiment-analysis")

print("Sentiment Analysis Results:\n")
for text in [
    "The new phone I bought is absolutely amazing!",
    "Worst customer service ever. I'm never coming back.",
    "The experience was average, nothing special.",
    "Fast delivery and the packaging was perfect.",
    "The product broke within two days. Very disappointed."
]:
    result, = classifier(text)
    print(f"Input Sentence: {text}\nPredicted Sentiment: {result['label']}, Confidence Score: {result['score']:.2f}\n")

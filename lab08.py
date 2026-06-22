import os
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere

os.environ["COHERE_API_KEY"] = input("Enter API Key: ")
text = TextLoader("Teaching.txt").load()[0].page_content

chain = PromptTemplate.from_template("Summarize this:\n{text}") | ChatCohere()
print("Generated Output:\n", chain.invoke({"text": text}).content)
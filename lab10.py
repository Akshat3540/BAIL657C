file_name = "IPC.pdf"
print("Uploaded file:", file_name)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader(file_name)
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(texts, embeddings)

def ipc_chatbot(query):
    docs = vectorstore.similarity_search(query, k=3)
    
    print("\nRelevant IPC Sections:\n")
    for i, doc in enumerate(docs):
        print(f"{i+1}. {doc.page_content[:500]}")
        print("-" * 50)

print("\nIPC Chatbot Ready (type 'exit' to stop)\n")

while True:
    query = input("You: ")
    
    if query.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
        
    ipc_chatbot(query)

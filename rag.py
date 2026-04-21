import json 
from langchain_ollama import OllamaEmbeddings 
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def load_knowledge(path = "knowledge.json") -> list[Document]:
    with open (path ,"r",) as f :
        data = json.load(f)
        
    docs = []
     
    
    for plan in data["plans"]:
        text = (f"Plan: {plan['name']} — ${plan['price_monthly']}/month. "
                f"{plan['videos_per_month']} videos/month. "
                f"{plan['resolution']} resolution. "
                f"AI captions: {plan['ai_captions']}. "
                f"Support: {plan['support']}.")
        docs.append(Document(page_content=text))

    for policy in data["policies"]:
        docs.append(Document(page_content=policy))

    return docs

def build_retriever():
    docs = load_knowledge()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

def retrieve_context(query: str, retriever) -> str:
    results = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in results])
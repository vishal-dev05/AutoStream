from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def classify_intent(user_message: str) -> str:
    llm =   ChatOllama(model="llama3")
    Systtem_prompt = """
    You are an intent classifier.
    Classify the user message into EXACTLY one of these:
    - greeting  
    - inquiry
    - high_intent

    Reply with only the label. No explanation."""
    messages =[
        SystemMessage(content=Systtem_prompt),
        HumanMessage(content=user_message)
    ]
    response = llm.invoke(messages)
    lable = response.content.strip().lower()
    
    if lable not in ["greeting", "inquiry", "high_intent"]:
        return "inquiry"
    return lable
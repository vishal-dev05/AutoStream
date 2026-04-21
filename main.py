from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from state import AgentState
from intent import classify_intent
from rag import build_retriever, retrieve_context
from tools import mock_lead_capture

llm = ChatOllama(model="llama3")
retriever = build_retriever()



def classify_intent_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1].content
    intent = classify_intent(last_message)
    return {**state, "intent": intent}


def rag_response_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    context = retrieve_context(query, retriever)

    prompt = f"""You are a helpful assistant for AutoStream.
Use the context below to answer the user's question.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    new_messages = state["messages"] + [AIMessage(content=response.content)]
    return {**state, "messages": new_messages}



def parse_lead_info_node(state: AgentState) -> AgentState:
    user_reply = state["messages"][-1].content

    if not state["lead_name"]:
        return {**state, "lead_name": user_reply}
    elif not state["lead_email"]:
        return {**state, "lead_email": user_reply}
    elif not state["lead_platform"]:
        return {**state, "lead_platform": user_reply}

    return state

def lead_collection_node(state: AgentState) -> AgentState:
    if state["lead_captured"]:
        return state

    # Check what's missing and ask for it
    if not state["lead_name"]:
        reply = "Great! I'd love to get you started. What's your name?"
    elif not state["lead_email"]:
        reply = f"Thanks {state['lead_name']}! What's your email address?"
    elif not state["lead_platform"]:
        reply = "Which platform do you create for? (YouTube, Instagram, TikTok, etc.)"
    else:
        # All 3 collected — fire the tool
        result = mock_lead_capture.invoke({
            "name": state["lead_name"],
            "email": state["lead_email"],
            "platform": state["lead_platform"]
        })
        reply = f"You're all set! {result} We'll be in touch soon."
        state = {**state, "lead_captured": True}

    new_messages = state["messages"] + [AIMessage(content=reply)]
    return {**state, "messages": new_messages}



def route_after_intent(state: AgentState) -> str:
    intent = state["intent"]
    if intent == "greeting":
        return "greet"
    elif intent == "inquiry":
        return "rag"
    else:  # high_intent
        return "parse_lead"

def greet_node(state: AgentState) -> AgentState:
    reply = "Hey! Welcome to AutoStream. How can I help you today?"
    return {**state, "messages": state["messages"] + [AIMessage(content=reply)]}

graph = StateGraph(AgentState)

graph.add_node("classify", classify_intent_node)
graph.add_node("greet", greet_node)
graph.add_node("rag", rag_response_node)
graph.add_node("parse_lead", parse_lead_info_node)
graph.add_node("collect_lead", lead_collection_node)

graph.set_entry_point("classify")

graph.add_conditional_edges("classify", route_after_intent, {
    "greet": "greet",
    "rag": "rag",
    "parse_lead": "parse_lead"
})

graph.add_edge("greet", END)
graph.add_edge("rag", END)
graph.add_edge("parse_lead", "collect_lead")
graph.add_edge("collect_lead", END)

agent = graph.compile()

def run():
    state: AgentState = {
        "messages": [],
        "intent": "",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False
    }

    print("AutoStream Agent ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        state["messages"].append(HumanMessage(content=user_input))
        state = agent.invoke(state)

        last_ai = [m for m in state["messages"] if isinstance(m, AIMessage)]
        if last_ai:
            print(f"Agent: {last_ai[-1].content}\n")
            
            
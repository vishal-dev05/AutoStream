from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from state import AgentState
from intent import classify_intent
from rag import build_retriever, retrieve_context
from tools import mock_lead_capture
import re

llm = ChatOllama(model="llama3")
retriever = build_retriever()


# ── Passive extraction ────────────────────────────────────────────────────────
# Runs on EVERY message, regardless of intent.
# Uses the LLM to pull name/email/platform from natural language.

def extract_lead_info_from_message(text: str, current_state: AgentState) -> dict:
    """
    Ask the LLM to extract any lead fields present in the user's message.
    Returns a dict with keys: name, email, platform (each is a string or None).
    Only extracts fields that are not already collected.
    """
    already_have = []
    if current_state.get("lead_name"):
        already_have.append(f"name={current_state['lead_name']}")
    if current_state.get("lead_email"):
        already_have.append(f"email={current_state['lead_email']}")
    if current_state.get("lead_platform"):
        already_have.append(f"platform={current_state['lead_platform']}")

    already_str = ", ".join(already_have) if already_have else "nothing yet"

    prompt = f"""Extract lead information from the user message below.
Already collected: {already_str}

Rules:
- Only extract fields NOT already collected.
- name: a person's first name or full name (not a company or product name)
- email: anything matching the pattern user@domain.ext
- platform: one of YouTube, Instagram, TikTok, Twitter, LinkedIn (or similar content platform)
- If a field is not clearly present in the message, return null for it.
- Return ONLY a raw JSON object, no markdown, no explanation.

User message: "{text}"

Return format:
{{"name": "string or null", "email": "string or null", "platform": "string or null"}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Strip markdown fences if the model adds them anyway
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        import json
        extracted = json.loads(raw)
        return {
            "name": extracted.get("name") or None,
            "email": extracted.get("email") or None,
            "platform": extracted.get("platform") or None,
        }
    except Exception:
        # Fallback: plain regex for email at minimum
        email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
        return {
            "name": None,
            "email": email_match.group(0) if email_match else None,
            "platform": None,
        }


def apply_extracted(state: AgentState, extracted: dict) -> AgentState:
    """Merge extracted fields into state, never overwriting existing values."""
    updated = {**state}
    changed = []

    if extracted.get("name") and not state.get("lead_name"):
        updated["lead_name"] = extracted["name"]
        changed.append(f"name={extracted['name']}")

    if extracted.get("email") and not state.get("lead_email"):
        updated["lead_email"] = extracted["email"]
        changed.append(f"email={extracted['email']}")

    if extracted.get("platform") and not state.get("lead_platform"):
        updated["lead_platform"] = extracted["platform"]
        changed.append(f"platform={extracted['platform']}")

    if changed:
        print(f"\n  [STATE] Passively extracted → {', '.join(changed)}")

    return updated


# ── Debug printer ─────────────────────────────────────────────────────────────

def print_debug_state(state: AgentState, label: str = ""):
    """Print current state fields to terminal in a readable format."""
    print(f"\n  ┌── DEBUG STATE {f'({label}) ' if label else ''}─────────────────")
    print(f"  │  intent:        {state.get('intent', '—')}")
    print(f"  │  lead_name:     {state.get('lead_name') or 'None'}")
    print(f"  │  lead_email:    {state.get('lead_email') or 'None'}")
    print(f"  │  lead_platform: {state.get('lead_platform') or 'None'}")
    print(f"  │  lead_captured: {state.get('lead_captured', False)}")
    print(f"  └────────────────────────────────────────")


# ── Graph nodes ───────────────────────────────────────────────────────────────

def classify_intent_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1].content

    # 1. Classify intent
    intent = classify_intent(last_message)
    print(f"\n  [INTENT] Classified as → {intent}")

    # 2. Passively extract lead info from this message (runs every turn)
    extracted = extract_lead_info_from_message(last_message, state)
    state = apply_extracted(state, extracted)

    return {**state, "intent": intent}


def rag_response_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    context = retrieve_context(query, retriever)

    prompt = f"""You are a helpful assistant for AutoStream, a video editing SaaS for content creators.
Use the context below to answer the user's question.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {query}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    new_messages = state["messages"] + [AIMessage(content=response.content)]
    return {**state, "messages": new_messages}


def lead_collection_node(state: AgentState) -> AgentState:
    """
    Asks for whichever lead fields are still missing, one at a time.
    Fires mock_lead_capture only once all three are collected.
    """
    if state.get("lead_captured"):
        reply = "You're already signed up! Our team will reach out soon."
        return {**state, "messages": state["messages"] + [AIMessage(content=reply)]}

    if not state.get("lead_name"):
        reply = "Awesome, let's get you set up! What's your name?"

    elif not state.get("lead_email"):
        reply = f"Nice to meet you, {state['lead_name']}! What's your email address?"

    elif not state.get("lead_platform"):
        reply = "Which platform do you mainly create for? (YouTube, Instagram, TikTok, etc.)"

    else:
        # All 3 collected — fire the CRM tool
        result = mock_lead_capture.invoke({
            "name": state["lead_name"],
            "email": state["lead_email"],
            "platform": state["lead_platform"],
        })
        reply = f"🎉 You're all set, {state['lead_name']}! {result} Our team will reach out soon."
        state = {**state, "lead_captured": True}
        print(f"\n  [CRM] mock_lead_capture fired → {state['lead_name']} | {state['lead_email']} | {state['lead_platform']}")

    new_messages = state["messages"] + [AIMessage(content=reply)]
    return {**state, "messages": new_messages}


def greet_node(state: AgentState) -> AgentState:
    name_part = f", {state['lead_name']}" if state.get("lead_name") else ""
    reply = f"Hey{name_part}! 👋 Welcome to AutoStream — automated video editing for content creators. How can I help you today?"
    return {**state, "messages": state["messages"] + [AIMessage(content=reply)]}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_intent(state: AgentState) -> str:
    intent = state["intent"]
    if intent == "greeting":
        return "greet"
    elif intent == "inquiry":
        return "rag"
    else:  # high_intent
        return "collect_lead"


# ── Graph assembly ────────────────────────────────────────────────────────────

graph = StateGraph(AgentState)

graph.add_node("classify", classify_intent_node)
graph.add_node("greet", greet_node)
graph.add_node("rag", rag_response_node)
graph.add_node("collect_lead", lead_collection_node)   # parse + collect merged

graph.set_entry_point("classify")

graph.add_conditional_edges("classify", route_after_intent, {
    "greet": "greet",
    "rag": "rag",
    "collect_lead": "collect_lead",
})

graph.add_edge("greet", END)
graph.add_edge("rag", END)
graph.add_edge("collect_lead", END)

agent = graph.compile()


# ── CLI runner ────────────────────────────────────────────────────────────────

def run():
    state: AgentState = {
        "messages": [],
        "intent": "",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
    }

    print("AutoStream Agent ready. Type 'quit' to exit.\n")
    print("  Tip: debug state prints automatically after each turn.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
        state = agent.invoke(state)

        last_ai = [m for m in state["messages"] if isinstance(m, AIMessage)]
        if last_ai:
            print(f"\nAgent: {last_ai[-1].content}")

        print_debug_state(state)
        print()


if __name__ == "__main__":
    run()
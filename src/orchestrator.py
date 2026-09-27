import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. SETUP & PATH RESOLUTION
# ==========================================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[0]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.llm_providers import PROVIDERS, build_llm, message_text, resolve_model

load_dotenv(project_root / ".env")

PROMPT_PATH = project_root / "src" / "prompts" / "system_prompt.txt"

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: Could not find {PROMPT_PATH}")
        return "You are a helpful AI assistant."

def default_tools():
    # Imported lazily: agent_tools connects to Pinecone and loads embeddings on import
    from src.agent_tools import query_telemetry_db, fetch_corridor_conditions, search_compliance_sop
    return [query_telemetry_db, fetch_corridor_conditions, search_compliance_sop]

# ==========================================
# 2. GRAPH ARCHITECTURE ASSEMBLY
# ==========================================
def build_agent(llm, tools=None, system_prompt=None, checkpointer=None):
    """Compile the reasoner/tools loop for any LangChain chat model."""
    tools = default_tools() if tools is None else tools
    system_message = SystemMessage(content=system_prompt if system_prompt is not None else load_system_prompt())
    llm_with_tools = llm.bind_tools(tools)

    def reasoning_node(state: AgentState):
        # System prompt is prepended per call (not stored in history) so every provider gets it
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("reasoner", reasoning_node)
    graph_builder.add_node("tools", ToolNode(tools))

    graph_builder.add_edge(START, "reasoner")
    graph_builder.add_conditional_edges("reasoner", tools_condition)
    graph_builder.add_edge("tools", "reasoner")

    return graph_builder.compile(checkpointer=checkpointer or MemorySaver())

# ==========================================
# 4. CHAT LOOP TESTING PANEL
# ==========================================
if __name__ == "__main__":
    provider_id = os.getenv("AGENT_LLM", "gemini").strip().lower()
    if provider_id not in PROVIDERS:
        provider_id = "ollama"
    model_name = resolve_model(provider_id)

    print("⚙️ Compiling LangGraph FDE Orchestrator...")
    fde_agent = build_agent(build_llm(provider_id, model_name))

    print("\n" + "="*55)
    print("🚀 FDE Supply Chain Orchestrator State Machine Online")
    print(f"   Configured Execution: [LLM: {PROVIDERS[provider_id].label} / {model_name}] -> [Embeddings: {os.getenv('EMBEDDING_MODEL', 'LOCAL')}]")
    print("="*55 + "\n")

    thread_config = {"configurable": {"thread_id": "production_test_1"}}

    while True:
        user_input = input("\nDispatcher > ")
        if user_input.lower() in ['exit', 'quit']:
            break

        events = fde_agent.stream({"messages": [("user", user_input)]}, config=thread_config, stream_mode="updates")
        for event in events:
            for node_name, node_state in event.items():
                if node_name == "tools":
                    print("   [System] 🔄 Retrieving external data elements via ToolNode...")
                elif node_name == "reasoner":
                    latest_text = message_text(node_state["messages"][-1])
                    if latest_text:
                        print(f"\n🤖 FDE Agent:\n{latest_text}")

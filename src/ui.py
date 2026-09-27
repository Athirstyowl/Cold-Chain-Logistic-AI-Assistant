import os
import sys
import hmac
import uuid
import json
import urllib
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_core.messages import HumanMessage, ToolMessage

# ==========================================
# 1. IMMEDIATE PATH & ENVIRONMENT RESOLUTION
# ==========================================
script_dir = Path(__file__).resolve().parent  # points to src/
project_root = script_dir.parent              # climbs to project root

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

from src.orchestrator import build_agent
from src.llm_providers import PROVIDERS, available_providers, build_llm, final_answer, friendly_error, message_text, resolve_model

CUSTOM_MODEL_OPTION = "Other model ID…"

@st.cache_resource(show_spinner="Loading the model and data tools…")
def get_agent(provider_id: str, model_name: str):
    """One compiled agent per provider/model, shared across reruns."""
    return build_agent(build_llm(provider_id, model_name))

# ==========================================
# 2. SQL CREDENTIALS MAPPING FROM .ENV
# ==========================================
db_host = os.getenv("SQL_SERVER_HOST", "localhost")
db_port = os.getenv("SQL_SERVER_PORT", "1433")
db_user = os.getenv("SQL_AGENT_USER", "USR_FDE_RO")
db_password = os.getenv("SQL_AGENT_PASSWORD")

# Engine for the Agent to write logs using its standard credentials
connection_string = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={db_host},{db_port};"
    f"DATABASE=master;"
    f"UID={db_user};"
    f"PWD={db_password};"
    f"Encrypt=no;"
    f"TrustServerCertificate=yes;"
)

log_params = urllib.parse.quote_plus(connection_string)

log_engine = create_engine(f"mssql+pyodbc:///?odbc_connect={log_params}")

def write_audit_log(session_id, node_name, tool_name, content):
    """Silently writes agent execution traces to the SQL audit table using agent permissions."""
    try:
        with log_engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO FDE_VIEWS.AgentAuditLog (SessionID, NodeExecuted, ToolName, Content)
                VALUES (:session_id, :node_name, :tool_name, :content)
            """), {
                "session_id": session_id,
                "node_name": node_name,
                "tool_name": tool_name,
                "content": content
            })
            conn.commit()
    except Exception as e:
        print(f"Audit Log Failed (Silent): {e}")

# ==========================================
# 3. PAGE CONFIGURATION & THEME
# ==========================================
# Palette, fonts and radii live in .streamlit/config.toml; CSS here only covers what the theme cannot
st.set_page_config(
    page_title="Cold-chain dispatch",
    page_icon=":material/ac_unit:",
    layout="wide",
    initial_sidebar_state="auto"
)

st.html("""
<style>
    /* Tokens: canvas, ink, accent. Everything else is ink at an opacity. */
    :root {
        --canvas: #0B0F19;
        --ink: #E8E6E1;
        --accent: #FF4F00;
        --ink-5: rgb(232 230 225 / 0.05);
        --ink-10: rgb(232 230 225 / 0.10);
        --ink-25: rgb(232 230 225 / 0.25);
        --ink-40: rgb(232 230 225 / 0.40);
        --ink-55: rgb(232 230 225 / 0.55);
        --serif: "Instrument Serif", Georgia, serif;
        --mono: "Geist Mono", ui-monospace, monospace;
        --swift: 150ms ease;
        --spring: 260ms cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    .stMainBlockContainer, [data-testid="stBottomBlockContainer"] { max-width: 68rem; }
    .stMainBlockContainer { padding-top: 3rem; }

    h1 { font-size: clamp(2.75rem, 6vw, 4.25rem); line-height: 0.95; letter-spacing: -0.02em; }
    [data-testid="stSidebar"] h3 { font-size: 1.6rem; line-height: 1.05; }

    /* Depth comes from 1px rules, never shadows */
    [data-baseweb="popover"] > div, [data-baseweb="menu"], [role="listbox"], [data-testid="stExpander"] details,
    [data-testid="stForm"], [data-testid="stChatInput"] {
        box-shadow: none !important;
    }
    [data-baseweb="popover"] > div, [role="listbox"] { border: 1px solid var(--ink-10); }

    /* Thermal scale: SOP section 1, monochrome, breach hatched */
    .thermal-scale { margin: 0; }
    .thermal-scale .track { display: grid; grid-template-columns: 2fr 3fr 3fr; height: 0.75rem; border: 1px solid var(--ink-25); border-radius: 2px; overflow: hidden; }
    .thermal-scale .frozen { background: var(--ink-5); }
    .thermal-scale .fresh { background: var(--ink-40); border-inline: 1px solid var(--ink); }
    .thermal-scale .breach { background: repeating-linear-gradient(-45deg, var(--ink-55) 0 1px, transparent 1px 6px); }
    .thermal-scale .ticks { display: grid; grid-template-columns: 2fr 3fr 3fr; font-family: var(--mono); font-size: 0.78rem; color: var(--ink-55); padding-top: 0.45rem; }
    .thermal-scale .ticks span + span { padding-left: 0.4rem; border-left: 1px solid var(--ink-25); }
    .thermal-scale .ticks b { display: block; font-weight: 500; color: var(--ink); }
    .thermal-scale figcaption { font-size: 0.85rem; color: var(--ink-55); padding-top: 0.75rem; max-width: 30rem; }

    /* SOP triggers: a definition list with hairline rules */
    [data-testid="stHtml"] dl.sop-rules { margin: 0; padding: 0; border-top: 1px solid var(--ink-10); }
    .sop-rules > div { padding: 0.9rem 0; border-bottom: 1px solid var(--ink-10); }
    .sop-rules dt { font-family: var(--mono); font-size: 0.85rem; font-weight: 500; }
    .sop-rules dd { margin: 0.3rem 0 0; color: var(--ink-55); font-size: 0.95rem; line-height: 1.5; }
    .sop-source { font-size: 0.8rem; color: var(--ink-55); padding-top: 0.6rem; }

    /* Buttons: spring scale, 150ms colour swaps */
    button[data-testid^="stBaseButton"] {
        box-shadow: none;
        transition: transform var(--spring), border-color var(--swift), color var(--swift), background-color var(--swift);
    }
    button[data-testid^="stBaseButton"]:hover { transform: scale(1.02); border-color: var(--ink-40); color: var(--ink); }
    button[data-testid^="stBaseButton"]:active { transform: scale(0.98); }

    /* Primary actions own the accent; hover inverts fill to outline */
    button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
        background: var(--accent); border: 1px solid var(--accent); color: var(--canvas); font-weight: 500;
    }
    button[data-testid="stBaseButton-primary"] p, button[data-testid="stBaseButton-primaryFormSubmit"] p { font-weight: 500; }
    button[data-testid="stBaseButton-primary"]:hover, button[data-testid="stBaseButton-primaryFormSubmit"]:hover {
        background: transparent; border-color: var(--accent); color: var(--accent);
    }

    /* Starter questions: ruled rows that slide instead of scale */
    [class*="st-key-starter"] button {
        justify-content: flex-start; text-align: left; background: transparent; border: 0;
        border-bottom: 1px solid var(--ink-10); border-radius: 0; padding: 0.9rem 0; color: var(--ink-55);
        transition: transform var(--spring), border-color var(--swift), color var(--swift);
    }
    [class*="st-key-starter"] button > div { justify-content: flex-start; }
    [class*="st-key-starter"] button p { text-align: left; font-size: 1rem; }
    [class*="st-key-starter"] button:hover { transform: translateX(6px); color: var(--ink); border-color: var(--ink-40); background: transparent; }
    [class*="st-key-starter"] button:active { transform: translateX(3px); }
    [class*="st-key-starter_0"] button { border-top: 1px solid var(--ink-10); }

    /* Fields and selects: the visible border lives on these wrappers in Streamlit 1.62 */
    [data-testid="stSelectbox"] [role="group"], [data-testid="stTextInputRootElement"], [data-testid="stChatInput"] > div {
        transition: border-color var(--swift), background-color var(--swift);
    }
    [data-testid="stSelectbox"] [role="group"]:hover, [data-testid="stTextInputRootElement"]:hover, [data-testid="stChatInput"] > div:hover {
        border-color: var(--ink-40);
    }
    [data-baseweb="menu"] li, [role="option"] { transition: color var(--swift), background-color var(--swift); }

    /* Send is the primary action of the page */
    [data-testid="stChatInputSubmitButton"] { background: var(--accent); color: var(--canvas); transition: transform var(--spring), opacity var(--swift); }
    [data-testid="stChatInputSubmitButton"]:hover:not(:disabled) { transform: scale(1.08); }
    [data-testid="stChatInputSubmitButton"]:disabled { background: var(--ink-10); color: var(--ink-40); }

    /* Expanders and segmented control */
    [data-testid="stExpander"] details { transition: border-color var(--swift); }
    [data-testid="stExpander"] details:hover { border-color: var(--ink-25); }
    [data-testid="stExpander"] summary { color: var(--ink-55); transition: color var(--swift); }
    [data-testid="stExpander"] summary:hover { color: var(--ink); }
    [data-testid="stButtonGroup"] button { color: var(--ink-55); transition: color var(--swift), border-color var(--swift), background-color var(--swift); }
    [data-testid="stButtonGroup"] button:hover { color: var(--ink); border-color: var(--ink-40); }
    /* Selection is state, not an action: ink, not accent */
    [data-testid="stButtonGroup"] button[aria-checked="true"] { background: var(--ink-10) !important; border-color: var(--ink-40) !important; }
    [data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[aria-checked="true"] * { color: var(--ink) !important; }

    /* Focus on fields stays in ink; the accent ring is reserved for keyboard focus */
    [data-testid="stSelectbox"] [role="group"]:focus-within, [data-testid="stTextInputRootElement"]:focus-within,
    [data-testid="stChatInput"] > div:focus-within { border-color: var(--ink-55) !important; }

    /* Heading permalink icons add noise to an operations tool */
    [data-testid="stHeaderActionElements"] { display: none; }

    /* Conversation: the question is the headline, the answer sits beneath it */
    [data-testid^="stChatMessageAvatar"] { display: none; }
    [data-testid="stChatMessage"] { background: transparent; padding: 0; gap: 0; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        border-top: 1px solid var(--ink-10); margin-top: 2.5rem; padding-top: 2rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
        font-family: var(--serif); font-size: clamp(1.6rem, 3vw, 2.1rem); line-height: 1.15; font-weight: 400; max-width: 44rem;
    }
    /* Answers hang 3rem under their question */
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] { margin-left: 0; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] { margin-left: 3rem; max-width: 52rem; }
    @media (max-width: 640px) {
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] { margin-left: 0; }
    }
    [data-testid="stChatMessage"] :is(h1, h2, h3) { font-size: 1.5rem; padding: 1.25rem 0 0.4rem; }
    [data-testid="stChatMessage"] :is(h4, h5, h6) { font-size: 1.1rem; }
    [data-testid="stChatMessage"] table { font-family: var(--mono); font-size: 0.82rem; border-collapse: collapse; }
    [data-testid="stChatMessage"] :is(th, td) { border: 0; border-bottom: 1px solid var(--ink-10); padding: 0.55rem 0.9rem 0.55rem 0; }
    [data-testid="stChatMessage"] th { background: transparent; color: var(--ink-55); font-weight: 500; }

    /* Code stays in the three-colour system */
    [data-testid="stCode"] code, [data-testid="stCode"] code span { color: var(--ink) !important; }
    [data-testid="stForm"] { max-width: 40rem; }

    :focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { transition: none !important; }
        button[data-testid^="stBaseButton"]:hover, [class*="st-key-starter"] button:hover,
        [data-testid="stChatInputSubmitButton"]:hover { transform: none; }
    }
</style>
""")

TOOL_LABELS = {
    "query_telemetry_db": "Fleet telemetry",
    "fetch_corridor_conditions": "Route weather",
    "search_compliance_sop": "SOP lookup",
}

STARTER_QUESTIONS = [
    "Find active shipments near Los Angeles (33.8, -118.1), check the weather there and tell me if any cargo temperature breaks the fresh perishables SOP.",
    "Which shipments are High Risk with a delay probability above 0.65, and who do I escalate them to?",
    "Port congestion at Long Beach is above 7.0. What does the SOP tell me to do with active freight?",
]

def tool_label(name: str) -> str:
    return TOOL_LABELS.get(name, name)

def render_sources(traces):
    """One collapsed section per answer: what each tool was asked and what it returned."""
    inputs = [t for t in traces if t["type"] == "tool_input"]
    if not inputs:
        return
    outputs = {}
    for t in traces:
        if t["type"] == "tool_output":
            outputs.setdefault(t["name"], []).append(t["content"])

    checked = ", ".join(dict.fromkeys(tool_label(t["name"]) for t in inputs))
    with st.expander(f"Sources checked: {checked}", icon=":material/fact_check:"):
        for t in inputs:
            st.markdown(f"**{tool_label(t['name'])}**")
            if t["name"] == "query_telemetry_db" and "sql_query" in t["args"]:
                st.code(t["args"]["sql_query"], language="sql", wrap_lines=True)
            else:
                st.caption("; ".join(f"{key}: {value}" for key, value in t["args"].items()))
            returned = outputs.get(t["name"], [])
            if returned:
                st.code(returned.pop(0), language="text", wrap_lines=True)

def start_new_conversation(notice=None):
    st.session_state.ui_messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    if notice:
        st.session_state.llm_notice = notice

# ==========================================
# 4. MULTI-USER STATE & THREAD MANAGEMENT
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

# ==========================================
# 5. SIDEBAR: VIEW SWITCH & MODEL CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### SoCal Logistics Ops")
    st.caption("Cold-chain dispatch assistant")

    app_mode = st.segmented_control(
        "View", ["Dispatch", "Audit log"], default="Dispatch", key="app_mode", label_visibility="collapsed"
    ) or "Dispatch"

    st.markdown("#### Model")

    providers = available_providers()
    provider_ids = [p.id for p in providers]
    preferred_id = os.getenv("AGENT_LLM", "gemini").strip().lower()
    provider_id = st.selectbox(
        "Provider",
        options=provider_ids,
        index=provider_ids.index(preferred_id) if preferred_id in provider_ids else 0,
        format_func=lambda pid: PROVIDERS[pid].label,
    )
    spec = PROVIDERS[provider_id]

    default_model = resolve_model(provider_id)
    model_options = list(dict.fromkeys([default_model, *spec.suggested_models])) + [CUSTOM_MODEL_OPTION]
    model_choice = st.selectbox("Model", options=model_options, key=f"model_{provider_id}")
    if model_choice == CUSTOM_MODEL_OPTION:
        model_name = st.text_input("Model ID", key=f"custom_model_{provider_id}", placeholder=default_model).strip() or default_model
    else:
        model_name = model_choice

    if spec.key_env:
        st.caption(f"Using `{spec.key_env}` from the server .env")
    else:
        st.caption("Runs on the local Ollama server, no API key needed")

    if st.button("Test connection", icon=":material/network_check:", use_container_width=True):
        with st.spinner(f"Contacting {spec.label}…"):
            try:
                build_llm(provider_id, model_name).invoke("Reply with the single word OK.")
                st.success(f"Connected to {spec.label} ({model_name}).")
            except Exception as e:
                st.error(friendly_error(e, spec.label))

    unconfigured = [p for p in PROVIDERS.values() if p.id not in provider_ids]
    if unconfigured:
        with st.expander("Other providers"):
            st.caption("Add a key to the server .env to enable these:")
            for p in unconfigured:
                st.caption(f"{p.label}: `{p.key_env}`")

    # Switching engines mid-conversation starts a fresh thread: tool-call history is provider-specific
    active_llm = (provider_id, model_name)
    previous_llm = st.session_state.get("active_llm")
    if previous_llm and previous_llm != active_llm and st.session_state.ui_messages:
        start_new_conversation(f"Switched to {spec.label} ({model_name}). Started a new conversation.")
    st.session_state.active_llm = active_llm

    st.divider()
    if st.button("New conversation", icon=":material/add_comment:", use_container_width=True):
        start_new_conversation()
        st.rerun()
    st.caption(f"Conversation ID {st.session_state.thread_id[:8]}")

thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ==========================================
# 6. VIEW ROUTING (DISPATCH VS AUDIT)
# ==========================================

if app_mode == "Dispatch":
    # ------------------------------------------
    # DISPATCH: CHAT UI & AGENT EXECUTION
    # ------------------------------------------
    title_col, scale_col = st.columns([5, 3], gap="large", vertical_alignment="bottom")
    with title_col:
        st.title("Cold-chain dispatch")
        st.markdown("Ask about shipment temperatures, route conditions and what the SOP requires. Each answer lists the data it checked.")
    with scale_col:
        st.html("""
        <figure class="thermal-scale" role="img" aria-label="Fresh perishables must stay between 0.0 and 4.0 °C. Above 4.0 °C is a cold-chain breach.">
            <div class="track"><div class="frozen"></div><div class="fresh"></div><div class="breach"></div></div>
            <div class="ticks">
                <span><b>&lt; 0.0 °C</b>Frozen</span>
                <span><b>0.0–4.0 °C</b>Fresh</span>
                <span><b>&gt; 4.0 °C</b>Breach</span>
            </div>
            <figcaption>Above 4.0 °C: restart the auxiliary cooling unit, and divert to cold storage if the ETA slips past 1 hour.</figcaption>
        </figure>
        """)

    if notice := st.session_state.pop("llm_notice", None):
        st.info(notice, icon=":material/swap_horiz:")

    pending_prompt = None
    intro = st.empty()
    if not st.session_state.ui_messages:
        with intro.container():
            questions_col, rules_col = st.columns([7, 5], gap="large")
            with questions_col:
                st.markdown("### Try a question")
                for i, question in enumerate(STARTER_QUESTIONS):
                    if st.button(question, key=f"starter_{i}", use_container_width=True):
                        pending_prompt = question
            with rules_col:
                st.markdown("### When the SOP says act")
                st.html("""
                <dl class="sop-rules">
                    <div><dt>Cargo &gt; 4.0 °C</dt><dd>Call the driver to restart the auxiliary cooling unit. Over 1 hour late: divert to the nearest emergency cold storage.</dd></div>
                    <div><dt>Port congestion &gt; 7.0</dt><dd>Don't hold freight at Long Beach or LA. Send active shipments to the Inland Empire Overflow Depot, San Bernardino.</dd></div>
                    <div><dt>High Risk, delay &gt; 0.65</dt><dd>Escalate to the Tier 2 Logistics Manager.</dd></div>
                </dl>
                <p class="sop-source">Source: Cold_Chain_Incident_SOP_v2.md</p>
                """)

    for entry in st.session_state.ui_messages:
        with st.chat_message(entry["role"]):
            if entry.get("is_error"):
                st.error(entry["content"], icon=":material/error:")
            else:
                st.markdown(entry["content"])
            if entry.get("truncated"):
                st.warning("This answer hit the model's output limit and may be cut off. Ask again or pick another model.", icon=":material/content_cut:")
            render_sources(entry.get("traces", []))

    user_input = st.chat_input("Ask about a shipment, a route or an SOP rule") or pending_prompt

    if user_input:
        intro.empty()
        st.session_state.ui_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            final_response = ""
            truncated = False
            current_traces = []
            error_message = None

            with st.status("Reading the question…", expanded=True) as status:
                try:
                    events = get_agent(provider_id, model_name).stream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=thread_config,
                        stream_mode="updates"
                    )

                    for event in events:
                        for node_name, node_state in event.items():

                            if node_name == "reasoner":
                                latest_msg = node_state["messages"][-1]

                                # A. Intercept Tool Call Requests (Inputs)
                                if getattr(latest_msg, "tool_calls", None):
                                    labels = ", ".join(dict.fromkeys(tool_label(c["name"]) for c in latest_msg.tool_calls))
                                    status.update(label=f"Checking {labels}…")
                                    for tool_call in latest_msg.tool_calls:
                                        st.write(f"Asked {tool_label(tool_call['name'])}")
                                        current_traces.append({
                                            "type": "tool_input",
                                            "name": tool_call['name'],
                                            "args": tool_call['args']
                                        })

                                        write_audit_log(
                                            session_id=st.session_state.thread_id,
                                            node_name="reasoner",
                                            tool_name=tool_call['name'],
                                            content=json.dumps(tool_call['args'])
                                        )

                                # B. Intercept Final Generation (text sent alongside tool calls is only commentary)
                                answer = final_answer(latest_msg)
                                if answer is None:
                                    if commentary := message_text(latest_msg):
                                        st.caption(commentary)
                                else:
                                    final_response, truncated = answer

                                if answer and final_response:
                                    write_audit_log(
                                        session_id=st.session_state.thread_id,
                                        node_name="reasoner_final",
                                        tool_name=f"LLM Text Synthesis ({provider_id}/{model_name})"[:100],
                                        content=final_response
                                    )

                            elif node_name == "tools":
                                status.update(label="Writing the answer…")
                                for msg in node_state.get("messages", []):
                                    if isinstance(msg, ToolMessage):
                                        st.write(f"Got a response from {tool_label(msg.name)}")
                                        current_traces.append({
                                            "type": "tool_output",
                                            "name": msg.name,
                                            "content": msg.content
                                        })

                                        write_audit_log(
                                            session_id=st.session_state.thread_id,
                                            node_name="tools",
                                            tool_name=msg.name,
                                            content=msg.content
                                        )

                    status.update(label="Done", state="complete", expanded=False)
                except Exception as e:
                    error_message = friendly_error(e, spec.label)
                    status.update(label="The model call failed", state="error", expanded=False)

        if error_message:
            reply = {"role": "assistant", "content": error_message, "traces": current_traces, "is_error": True}
        elif final_response:
            reply = {"role": "assistant", "content": final_response, "traces": current_traces, "truncated": truncated}
        else:
            reason = " It used its whole output limit before writing one." if truncated else ""
            reply = {
                "role": "assistant",
                "content": f"The model returned an empty answer.{reason} Ask again or pick another model.",
                "traces": current_traces,
                "is_error": True,
            }
        st.session_state.ui_messages.append(reply)
        # Rerun so the answer renders through the same path as history (sources section included)
        st.rerun()


elif app_mode == "Audit log":
    # ------------------------------------------
    # AUDIT LOG VIEWER (ADMIN CREDENTIALS FROM .ENV)
    # ------------------------------------------
    st.title("Audit log")
    st.markdown("Every question, tool call and answer from the dispatch assistant, newest first. Admin access only.")

    expected_admin_user = os.getenv("SQL_ADMIN_USER")
    expected_admin_pass = os.getenv("SQL_ADMIN_PASSWORD")

    if not st.session_state.get("admin_ok"):
        with st.form("admin_auth_form"):
            col1, col2 = st.columns(2)
            with col1:
                input_user = st.text_input("Admin username")
            with col2:
                input_pass = st.text_input("Admin password", type="password")
            submit_admin = st.form_submit_button("Sign in", type="primary")

        if submit_admin:
            user_ok = hmac.compare_digest(input_user.encode(), (expected_admin_user or "").encode())
            pass_ok = hmac.compare_digest(input_pass.encode(), (expected_admin_pass or "").encode())

            if expected_admin_user and expected_admin_pass and user_ok and pass_ok:
                st.session_state.admin_ok = True
                st.rerun()
            else:
                st.error("That username and password don't match the admin account in the server .env.", icon=":material/lock:")
    else:
        if st.button("Sign out", icon=":material/logout:"):
            st.session_state.admin_ok = False
            st.rerun()

        try:
            # Connect with the admin credentials from .env, already verified at sign-in
            admin_params = urllib.parse.quote_plus(
                "DRIVER={ODBC Driver 18 for SQL Server};"
                f"SERVER={db_host},{db_port};"
                "DATABASE=master;"
                f"UID={expected_admin_user};"
                f"PWD={expected_admin_pass};"
                "Encrypt=no;"
                "TrustServerCertificate=yes;"
            )
            admin_engine = create_engine(f"mssql+pyodbc:///?odbc_connect={admin_params}")

            with admin_engine.connect() as conn:
                query = """
                    SELECT LogID, Timestamp, SessionID, NodeExecuted, ToolName, Content
                    FROM FDE_VIEWS.AgentAuditLog
                    ORDER BY Timestamp DESC
                """
                df = pd.read_sql(query, conn)

            if df.empty:
                st.info("No entries yet. Ask a question in Dispatch and it will appear here.", icon=":material/inbox:")
            else:
                df["NodeExecuted"] = df["NodeExecuted"].map(
                    {"reasoner": "Tool request", "tools": "Tool result", "reasoner_final": "Answer"}
                ).fillna(df["NodeExecuted"])
                df["ToolName"] = df["ToolName"].map(lambda name: TOOL_LABELS.get(name, name))

                sessions = ["All conversations"] + list(dict.fromkeys(df["SessionID"].dropna()))
                chosen = st.selectbox(
                    "Conversation",
                    sessions,
                    format_func=lambda s: s if s == "All conversations" else s[:8],
                )
                if chosen != "All conversations":
                    df = df[df["SessionID"] == chosen]
                st.caption(f"{len(df)} entries")

                st.dataframe(
                    df,
                    column_config={
                        "LogID": st.column_config.NumberColumn("ID", format="%d"),
                        "Timestamp": st.column_config.DatetimeColumn("Time", format="DD/MM/YYYY h:mm a"),
                        "SessionID": "Conversation",
                        "NodeExecuted": "Step",
                        "ToolName": "Tool",
                        "Content": st.column_config.TextColumn("Content", width="large"),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=600
                )

        except Exception as e:
            st.error(f"Couldn't load the audit log: {e}", icon=":material/error:")

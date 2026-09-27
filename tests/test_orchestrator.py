from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import Field

from src.orchestrator import build_agent


class RecordingChatModel(BaseChatModel):
    """Fake reasoner that records the messages it receives."""

    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


@tool
def noop_tool(query: str) -> str:
    """Does nothing."""
    return query


def _run(agent, text, thread="t1"):
    config = {"configurable": {"thread_id": thread}}
    agent.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    return config


def test_reasoner_always_receives_the_system_prompt_first():
    llm = RecordingChatModel()
    agent = build_agent(llm, tools=[noop_tool], system_prompt="You are the dispatch analyst.")
    config = _run(agent, "status?")
    _run(agent, "and now?")

    for call in llm.seen:
        assert isinstance(call[0], SystemMessage)
        assert call[0].content == "You are the dispatch analyst."
    assert len(llm.seen) == 2


def test_system_prompt_is_not_stored_in_conversation_history():
    llm = RecordingChatModel()
    agent = build_agent(llm, tools=[noop_tool], system_prompt="SYS")
    config = _run(agent, "status?")

    history = agent.get_state(config).values["messages"]
    assert not any(isinstance(m, SystemMessage) for m in history)
    assert [type(m) for m in history] == [HumanMessage, AIMessage]


def test_default_system_prompt_is_loaded_from_prompt_file():
    llm = RecordingChatModel()
    agent = build_agent(llm, tools=[noop_tool])
    _run(agent, "hi")
    assert "CRITICAL OPERATIONAL FLOW" in llm.seen[0][0].content

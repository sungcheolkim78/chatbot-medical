"""
Chat engine module for handling conversations and response generation.
"""

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, END
from typing import List, Dict, Any, Tuple
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
import time

from .document_loader import DocumentLoader
from .utils import setup_logging

setup_logging()


class ChatEngine:
    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-latest",
        model_provider: str = "anthropic",
        temperature: float = 0.5,
        file_path: str = "knowledge/slamon1987_claude.md",
        text_splitter_type: str = "sentence_transformers_token",
        top_k: int = 2,
    ):
        self.messages: List[Dict[str, str]] = []
        self.conversation = None
        self.llm = init_chat_model(
            model_name, model_provider=model_provider, temperature=temperature
        )
        self.document_loader = DocumentLoader(
            file_path=file_path,
            pdf_loader_type=None,
            text_splitter_type=text_splitter_type,
            need_vectorstore=True,
        )
        self.document_loader.load_paper()
        self.top_k = top_k
        self.tool = StructuredTool.from_function(
            func=self.retrieve,
            name="retrieve",
            description="Retrieve information from the knowledge base",
            return_direct=False,
        )
        self.memory = InMemorySaver()
        self.config = {"configurable": {"thread_id": "1"}}
        self.build_graph()

    # Functions for building the LangGraph graph
    def retrieve(self, query: str):
        """Retrieve information related to a query from the knowledge base."""

        retrieved_docs = self.document_loader.vectorstore.similarity_search(
            query, k=self.top_k
        )
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def query_or_respond(self, state: MessagesState):
        """Generate tool call for retrieval or respond."""

        llm_with_tools = self.llm.bind_tools([self.tool])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def generate(self, state: MessagesState):
        """Generate answer."""

        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant with a medical background for question-answering tasks. "
            "provide accurate, helpful, and engaing responses to a wide range of user queries, "
            "enhancing user experience and knowledge across various topics. "
            "\n"
            "Use the following peices of retrieved context to answer "
            "the question. If the question is not related to the context, say that "
            "Sorry, I am allowed to answer questions related to the context only. "
            "\n"
            "Use five sentences maximum and ensure the answer is concise and to the point."
            "if the answer has a complex medical term, explain it in a simple way, although "
            "you can not find the details in the context. Use your knowledge to explain it."
            "\n\n"
            f"Here is the context: {docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        response = self.llm.invoke(prompt)
        return {"messages": [response]}

    def build_graph(self):
        """Build the graph for the chat engine."""

        graph_builder = StateGraph(MessagesState)

        graph_builder.add_node("query_or_respond", self.query_or_respond)
        graph_builder.add_node("tools", ToolNode([self.tool]))
        graph_builder.add_node("generate", self.generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        self.graph = graph_builder.compile(checkpointer=self.memory)

    def add_message(self, role: str, content: str):
        """Add a message to the chat history."""

        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        """Get the chat history."""

        return self.messages

    def generate_response(self, prompt: str) -> Tuple[str, float]:
        """Generate a response for the given prompt and return the response and time taken."""

        if not self.graph:
            raise ValueError(
                "Conversation not initialized. Call initialize_conversation first."
            )

        start_time = time.time()
        msg = self.graph.invoke(
            {"messages": [{"role": "user", "content": prompt}]}, 
            config=self.config
        )
        end_time = time.time()
        response_time = end_time - start_time

        return msg["messages"][-1].content, response_time

    def clear_history(self):
        """Clear the chat history."""

        self.messages = []
        if self.conversation:
            self.conversation.memory.clear()

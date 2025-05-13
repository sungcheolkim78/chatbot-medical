"""
GUI handler module for managing the Streamlit interface.
"""

import os
import streamlit as st

from .chat_engine import ChatEngine


class GUIHandler:
    def __init__(self, left_column, chat_column, right_column):
        """Initialize GUI handler with the provided columns."""
        self.left_column = left_column
        self.chat_column = chat_column
        self.right_column = right_column

        if "messages" not in st.session_state:
            st.session_state.messages = []

    def render_header(self):
        """Render the header in the chat column."""
        with self.chat_column:
            st.title("Medical Chatbot")
            st.markdown(
                "A chatbot powered by LangChain and Streamlit. You can ask questions about the knowledge base."
            )

    def render_knowledge_base(self, selected_file: str):
        """Render the knowledge base panel."""
        with self.right_column:
            st.markdown("### Knowledge Base")

            # Display file content in a scrollable markdown container
            if selected_file:
                file_path = os.path.join("knowledge", selected_file)
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        # Create a container with fixed height and scrolling
                        with st.container():
                            st.markdown(
                                f"""
                                <div style="height: 700px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                                    {content}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

    def render_chat_messages(self, chat_engine: ChatEngine):
        """Render the chat messages."""
        with self.chat_column:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    def render_chat_input(self, chat_engine: ChatEngine):
        """Render the chat input and handle user interaction."""
        with self.chat_column:
            if prompt := st.chat_input("Ask something..."):
                # Add user message to chat history
                chat_engine.add_message("user", prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response, response_time = chat_engine.generate_response(prompt)

                        # Check if response contains thinking process
                        if "<think>" in response and "</think>" in response:
                            # Split response into thinking and final answer
                            thinking_start = response.find("<think>") + len("<think>")
                            thinking_end = response.find("</think>")
                            thinking = response[thinking_start:thinking_end].strip()
                            final_answer = response[
                                thinking_end + len("</think>") :
                            ].strip()

                            # Display thinking process in a collapsible section
                            with st.expander("Thinking Process", expanded=False):
                                st.write(thinking)

                            # Display final answer
                            st.write(final_answer)
                            st.session_state.messages.append({"role": "assistant", "content": final_answer})
                        else:
                            st.write(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})

                        st.caption(f"Response time: {response_time:.2f} seconds")

                # Add assistant response to chat history
                chat_engine.add_message("assistant", response)

    def render_instructions(self):
        """Render the instructions expander."""
        with self.chat_column:
            with st.expander("How to use this chatbot"):
                st.markdown("""
                1. Select your preferred model provider and model from the left panel
                2. Adjust temperature as needed
                3. Type your message in the chat input and press Enter
                4. The chatbot will respond using the selected model
                5. Your conversation history is maintained during the session
                6. View knowledge base documents in the right panel

                Example questions:
                - What was observed regarding the HER-2/neu loci in the study?
                - What impact does HER-2/neu gene amplification have on patient outcomes according to the study?
                - What statistical method was used to evaluate the predictive power of prognostic factors in the study?
                """)

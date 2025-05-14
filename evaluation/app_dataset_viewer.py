"""
Streamlit application for viewing the evaluation dataset
"""

import streamlit as st
import json
import os
from datetime import datetime


def load_dataset(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def get_dataset_files():
    dataset_dir = "evaluation/datasets"
    return sorted([f for f in os.listdir(dataset_dir) if f.endswith(".json")])


def format_timestamp(timestamp):
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        st.error(f"Error formatting timestamp: {e}")
        return timestamp


def main():
    st.set_page_config(page_title="Dataset Viewer", layout="wide")
    st.title("Evaluation Dataset Viewer")

    # Get list of dataset files
    dataset_files = get_dataset_files()

    # Dropdown for dataset selection
    selected_file = st.selectbox(
        "Select Dataset",
        dataset_files,
        index=len(dataset_files) - 1 if dataset_files else 0,
    )

    if selected_file:
        # Load selected dataset
        dataset_path = os.path.join("evaluation/datasets", selected_file)
        data = load_dataset(dataset_path)

        # Get unique LLM types
        llm_types = sorted(list(set(entry["llm_type"] for entry in data)))

        # Initialize session state for current index if not exists
        if "current_index" not in st.session_state:
            st.session_state.current_index = 0

        # Create two columns for better layout
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Question")
            if data:
                entry = data[st.session_state.current_index]
                st.write(entry["question"])

                st.subheader("Answer")
                st.write(entry["answer"])

                st.subheader("Source")
                st.write(entry["source"])

                st.subheader("Excerpt Context")
                st.write(entry["full_context"])

        with col2:
            st.subheader("Filters")
            # Add LLM Type filter
            selected_llm = st.selectbox("Filter by LLM Type", ["All"] + llm_types)

            # Filter data based on selected LLM type
            filtered_data = (
                data
                if selected_llm == "All"
                else [entry for entry in data if entry["llm_type"] == selected_llm]
            )

            # Reset index if it's out of bounds for filtered data
            if st.session_state.current_index >= len(filtered_data):
                st.session_state.current_index = 0

            st.subheader("Metadata")
            if filtered_data:
                entry = filtered_data[st.session_state.current_index]
                st.write(f"**Difficulty:** {entry['difficulty']}")
                st.write(f"**Confidence:** {entry['confidence']}")
                st.write(f"**LLM Type:** {entry['llm_type']}")
                st.write(f"**Timestamp:** {format_timestamp(entry['timestamp'])}")

                # Navigation buttons
                st.write("---")
                col_prev, col_next = st.columns(2)

                with col_prev:
                    if st.button(
                        "Previous", disabled=st.session_state.current_index == 0
                    ):
                        st.session_state.current_index = max(
                            0, st.session_state.current_index - 1
                        )
                        st.rerun()

                with col_next:
                    if st.button(
                        "Next",
                        disabled=st.session_state.current_index
                        == len(filtered_data) - 1,
                    ):
                        st.session_state.current_index = min(
                            len(filtered_data) - 1, st.session_state.current_index + 1
                        )
                        st.rerun()

                # Progress indicator
                st.write(
                    f"Entry {st.session_state.current_index + 1} of {len(filtered_data)}"
                )
            else:
                st.warning("No entries found for the selected LLM type.")


if __name__ == "__main__":
    main()

import streamlit as st
import json
import pandas as pd
import glob
import os

# Set page config
st.set_page_config(
    page_title="Chatbot Response Viewer",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Chatbot Response Viewer")

# Load the JSON data
@st.cache_data
def load_all_data():
    # Get all JSON files in the results directory
    result_files = glob.glob('evaluation/chatbot_results/results_*.json')
    all_data = {}
    
    for file_path in result_files:
        model_name = os.path.basename(file_path).replace('results_', '').replace('.json', '')
        with open(file_path, 'r') as f:
            all_data[model_name] = json.load(f)
    
    return all_data

# Load scores from CSV
@st.cache_data
def load_scores(model_name):
    score_file = f'evaluation/chatbot_results/llm_score_{model_name}.csv'
    if os.path.exists(score_file):
        scores_df = pd.read_csv(score_file)
        return scores_df.set_index('id').to_dict('index')
    return {}

# Load all data
all_data = load_all_data()

# Initialize session state for current index if it doesn't exist
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

# Initialize session state for selected model if it doesn't exist
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = list(all_data.keys())[0]

# Get data for selected model
data = all_data[st.session_state.selected_model]
scores = load_scores(st.session_state.selected_model)

# Function to navigate to next item
def next_item():
    if st.session_state.current_index < len(data) - 1:
        st.session_state.current_index += 1

# Function to navigate to previous item
def prev_item():
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1

# Get current item
current_item = data[st.session_state.current_index]

# Sidebar content
with st.sidebar:
    st.subheader("Model Selection")
    st.session_state.selected_model = st.selectbox(
        "Select Model",
        options=list(all_data.keys()),
        index=list(all_data.keys()).index(st.session_state.selected_model)
    )
    
    st.markdown("---")
    
    st.subheader("Metadata")
    st.write(f"**Item:** {st.session_state.current_index + 1} of {len(data)}")
    st.write(f"**Difficulty:** {current_item['difficulty']}")
    st.write(f"**Confidence:** {current_item['confidence']}")
    st.write(f"**Model:** {current_item['model_name']}")
    st.write(f"**Temperature:** {current_item['temperature']}")
    
    # Add scores from CSV
    if current_item['id'] in scores:
        score_data = scores[current_item['id']]
        st.write(f"**Correctness Score:** {score_data['correctness']:.2f}")
        st.write(f"**Style Score:** {score_data['style']:.2f}")
        st.write(f"**Response Time:** {score_data['response_time']:.2f}s")
    
    st.markdown("---")
    
    # Add navigation buttons
    st.subheader("Navigation")
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("⬅️ Previous", disabled=st.session_state.current_index == 0):
            prev_item()
            st.rerun()
    with nav_col2:
        if st.button("Next ➡️", disabled=st.session_state.current_index == len(data) - 1):
            next_item()
            st.rerun()

# Question section spanning both columns
st.subheader("Question")
st.write(current_item['question'])
st.markdown("---")

# Create two columns for main content
col_answer, col_truth = st.columns([1, 1])

# Left column: Chatbot Answer
with col_answer:
    st.subheader("Chatbot Answer")
    st.write(current_item['answer'])

# Right column: Ground Truth and Source
with col_truth:
    st.subheader("Ground Truth")
    st.write(current_item['ground_truth'])
    st.write("**Source:**")
    st.write(current_item['source']) 
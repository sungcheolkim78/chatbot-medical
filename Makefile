lint:
	ruff check src

format:
	ruff format src

app:
	streamlit run src/chatbot_humana/main.py
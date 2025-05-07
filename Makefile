lint:
	ruff check src evaluation

format:
	ruff format src evaluation

app_crewai:
	streamlit run src/chatbot_crewai/main.py

eval:
	streamlit run evaluation/eval.py

dataset:
	python evaluation/dataset_generator.py
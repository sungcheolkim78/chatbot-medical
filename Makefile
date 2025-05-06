lint:
	ruff check src evaluation

format:
	ruff format src evaluation

app:
	streamlit run src/chatbot_humana/main.py

eval:
	streamlit run evaluation/eval.py

dataset:
	python evaluation/dataset_generator.py
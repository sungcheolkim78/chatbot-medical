lint:
	ruff check src evaluation

format:
	ruff format src evaluation

app_crewai:
	streamlit run src/chatbot_crewai/main.py

app_langchain:
	streamlit run src/chatbot_langchain/app.py

eval:
	streamlit run evaluation/eval.py

dataset:
	python evaluation/dataset_generator.py

batch:
	python src/chatbot_langchain/batch.py --config evaluation/configs/batch_config.yaml
	python evaluation/llm_scorer.py --config evaluation/configs/batch_config.yaml
	python evaluation/score_plot.py
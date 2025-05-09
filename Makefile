lint:
	ruff check src evaluation

format:
	ruff format src evaluation

chatbot_crewai:
	streamlit run src/chatbot_crewai/main.py

chatbot_langchain:
	streamlit run src/chatbot_langchain/app.py

eval_app:
	streamlit run evaluation/app_eval.py

eval_dataset:
	python evaluation/dataset_generator.py

eval_batch:
	python src/chatbot_langchain/batch.py --config evaluation/configs/batch_config.yaml
	python evaluation/llm_scorer.py --config evaluation/configs/batch_config.yaml
	python evaluation/score_plot.py

eval_plot:
	python evaluation/score_plot.py
lint:
	ruff check src evaluation

format:
	ruff format src evaluation

chatbot_crewai:
	streamlit run src/chatbot_crewai/main.py

chatbot_langchain:
	streamlit run src/chatbot_langchain/app.py

eval_dataset_app:
	streamlit run evaluation/app_dataset_viewer.py

eval_score_app:
	streamlit run evaluation/app_response_viewer.py

eval_dataset:
	python evaluation/dataset_generator.py --num-questions 2

eval_batch:
	#python src/chatbot_langchain/batch.py --config evaluation/configs/batch_config_granite1.yaml
	#python src/chatbot_langchain/batch.py --config evaluation/configs/batch_config_granite2.yaml
	#python src/chatbot_langchain/batch.py --config evaluation/configs/batch_config_llama1.yaml
	#python src/chatbot_langchain/batch.py --config evaluation/configs/batch_config_llama2.yaml
	#python src/chatbot_langchain/batch.py --config evaluation/configs/batch_config_qwen1.yaml
	#python src/chatbot_langchain/batch.py --config evaluation/configs/batch_config_qwen2.yaml
	python evaluation/llm_scorer.py --config evaluation/configs/batch_config_granite1.yaml
	python evaluation/llm_scorer.py --config evaluation/configs/batch_config_granite2.yaml
	python evaluation/llm_scorer.py --config evaluation/configs/batch_config_llama1.yaml
	python evaluation/llm_scorer.py --config evaluation/configs/batch_config_llama2.yaml
	python evaluation/llm_scorer.py --config evaluation/configs/batch_config_qwen1.yaml
	python evaluation/llm_scorer.py --config evaluation/configs/batch_config_qwen2.yaml
	python evaluation/score_plot.py

eval_batch_single:
	python src/chatbot_langchain/batch.py --config evaluation/configs/batch_config_granite1.yaml

eval_score_single:
	python evaluation/llm_scorer.py --config evaluation/configs/batch_config_granite1.yaml

eval_plot:
	python evaluation/score_plot.py
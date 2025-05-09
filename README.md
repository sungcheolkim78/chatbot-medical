# Medical Chatbot 

A sophisticated chatbot implementation offering two versions: one using CrewAI for advanced agent-based interactions and another using LangChain for flexible chain-based conversations.

## Overview

This repository contains two distinct chatbot implementations:

1. **CrewAI Version**: Leverages CrewAI for advanced conversational AI capabilities with agent-based architecture, enabling complex multi-agent interactions and task delegation.
2. **LangChain Version**: Utilizes LangChain framework for flexible chain-based conversations, offering robust language model integration and customizable conversation flows.

Both systems are built with modern AI technologies and provide robust frameworks for natural language interactions.

## Features

### CrewAI Version
- Advanced conversational AI using CrewAI
- Multi-agent architecture
- Task delegation and coordination
- Complex conversation handling
- Streamlit-based web interface

### LangChain Version
- Flexible chain-based conversations
- Multiple language model integrations
- Customizable conversation flows
- Knowledge base integration
- Document processing capabilities

### Common Features
- Comprehensive evaluation framework
- Web interface support
- Knowledge base integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chatbot_humana.git
cd chatbot_humana
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The project provides several command-line interfaces for both versions via Makefile:

### LangChain Version
- `chatbot_langchain`: Main LangChain chatbot interface
- `eval_dataset`: Generate the question/answer dataset from the knowledge base
- `eval_app`: Web application to view the evaluation dataset
- `eval_batch`: Generate chatbot conversation based on the evaluation dataset and create the score using judge LLM

### CrewAI Version
- `chatbot_crewai`: Main CrewAI chatbot interface

To start either version:
```bash
# For LangChain version
make chatbot_langchain # or python src/chatbot_langchain/app.py

# For CrewAI version
make chatbot_crewai # or python src/chatbot_crewai/main.py
```

## Key Evaluation Metrics

The evaluation framework includes several key metrics for both versions:

1. Response Quality
   - [x] Relevance
   - [x] Coherence
   - [x] Accuracy

2. Performance Metrics
   - [x] Response Time
   - [ ] Resource Utilization
   - [ ] Error Rates

3. User Experience
   - [ ] User Feedback
   - [ ] Friendliness and engagenss
   - [ ] Easiness toward different levels of user's knowlege

## Chat Engine Evaluation

We have tested several open-source LLM models and measured the performance in three categories. Here are the key results as a plot.

![](docs/figs/metrics_boxplot_by_model.png)

You can find the details [here](docs/opensource_model_performance.md)

## Notes

- The system requires Python 3.10 or higher
- GPU support is recommended for optimal performance
- Environment variables should be properly configured for API access
- Regular updates to the knowledge base are recommended
- Choose the appropriate version based on your specific needs:
  - CrewAI version is ideal for complex, multi-agent interactions
  - LangChain version is better for straightforward, chain-based conversations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Contact

[Add your contact information here]

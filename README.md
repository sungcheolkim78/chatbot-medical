# Chatbot Humana

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
- Training and testing capabilities
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

The project provides several command-line interfaces for both versions:

### CrewAI Version
- `chatbot_humana`: Main CrewAI chatbot interface
- `run_crew`: Run the CrewAI-based chat system

### LangChain Version
- `langchain_chat`: Main LangChain chatbot interface
- `train_langchain`: Train the LangChain model
- `test_langchain`: Run LangChain tests

To start either version:
```bash
# For CrewAI version
python -m chatbot_humana.main

# For LangChain version
python -m chatbot_humana.langchain_main
```

## Key Evaluation Metrics

The evaluation framework includes several key metrics for both versions:

1. Response Quality
   - Relevance
   - Coherence
   - Accuracy
   - Completeness

2. Performance Metrics
   - Response Time
   - Resource Utilization
   - Error Rates

3. User Experience
   - User Satisfaction
   - Task Completion Rate
   - Conversation Flow

## Chat Engine Evaluation

### CrewAI Version
1. Model Performance
   - Agent Coordination
   - Task Delegation
   - Multi-agent Communication
   - Context Management

2. System Performance
   - Latency
   - Throughput
   - Resource Efficiency

3. Integration Testing
   - API Compatibility
   - Error Handling
   - Recovery Mechanisms

### LangChain Version
1. Model Performance
   - Chain Execution
   - Language Understanding
   - Response Generation
   - Context Management

2. System Performance
   - Chain Processing Speed
   - Memory Usage
   - Resource Efficiency

3. Integration Testing
   - Chain Compatibility
   - Error Handling
   - Recovery Mechanisms

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

# Open-source LLM models

Here, I used the models only with ollama models and tools capability
For the test of the small LLM, different quantized models were also
tested. For the local inference, I restricted the model parameter size up to 8B
so that the video card with less than 10MB can be used.

- llama3.1:latest
- llama3.2:latest
- granite3.2:2b
- granite3.2:8b
- qwen3:1.7b
- qwen3:8b

## Accuracy vs. Inference time based on model parameter size

We have 3 major KPI; factuality score, performance score, and user experience score
Here, we measured the accuracy and the average response time to measure the first two
metrics. 

|   | Model | Parameter Size | Accuracy | Average Response Time | Coments |
| 1 | llama3.1 | 8B | | 
| 2 | llama3.2 | 3B | |
| 3 | granite3.2 | 2B | | | 
| 4 | granite3.2 | 8B | | | 
| 5 | qwen3 | 1.7B | | | Reasoning model |
| 6 | qwen3 | 8B | | | Reasoning model |

## Accuracy

We prepared an annotated dataset of question and answer pair with 3 different knowlege levels
(easy, medium, hard). And our chatbot generate the answer for given 

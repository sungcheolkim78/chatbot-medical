# Open-Source LLM Model Evaluation Framework

## Model Selection Criteria

This evaluation framework focuses on open-source LLM models deployed through Ollama with tool capabilities. The selection criteria prioritized models that could run efficiently on consumer-grade hardware with less than 10GB VRAM, limiting the parameter size to 8B or smaller. In the same logic, we selected `Q4_K_M` quantized models that saves the GPU memory and allow high speed.

### Evaluated Models
- Llama 3.1 (8B parameters)
- Llama 3.2 (3B parameters)
- Granite 3.2 (2B and 8B parameters)
- Qwen 3 (1.7B and 8B parameters)

Note: Llama model is the baseline Open-source LLM model. Granite is chosen due to its high performance on the RAG system. Qwen model is selected due to the reasoning capability.

## Evaluation Metrics

The framework employs three primary Key Performance Indicators (KPIs):

1. **Factuality Score**: Measures the accuracy of model responses against ground truth
2. **Performance Score**: Evaluates response quality and relevance
3. **User Experience Score**: Assesses the overall interaction quality

### Accuracy Assessment
- Dataset: Annotated question-answer pairs categorized by knowledge complexity (easy, medium, hard)
- Methodology: Automated evaluation of model responses against ground truth annotations

### User Experience Evaluation
- Methodology: External evaluation using a state-of-the-art (SOTA) LLM
- Focus: Assessment of conversation quality and interaction patterns

### Response Time Analysis
- Measurement: End-to-end latency from query input to response generation
- Normalization: 
  - Optimal response time (0.1s) mapped to score 1.0
  - Maximum acceptable response time (10s) mapped to score 0.0
  - Linear interpolation for intermediate values

## Performance Comparison

The following table presents the quantitative comparison of model performance:

| Model | Parameter Size | Correctness | Style | Response Time | Notes |
|-------|---------------|----------|-----------|-----------|-------|
| Llama 3.1 | 8B | 0.72 | 0.67 | 0.78 | |
| Llama 3.2 | 3B | 0.61 | 0.83 | 0.87 | |
| Granite 3.2 | 2B | 0.13 | 0.98 | 0.64 | |
| Granite 3.2 | 8B | 0.13 | 0.98 | 0.64 | |
| Qwen 3 | 1.7B | 0.33 | 0.97 | 0.65 | Reasoning-focused model |
| Qwen 3 | 8B | 0.46 | 0.92 | 0.25 | Reasoning-focused model |

1. Llama 3.1 (8B) shows good correctness (0.72) and decent style (0.67), with moderate response time (0.78)
1. Llama 3.2 (3B) has slightly lower correctness (0.61) but better style (0.83), with slightly faster response time (0.87)
1. Granite 3.2 (both 2B and 8B) shows lower correctness (0.13) but excellent style (0.98), with good response times (0.64)
1. Qwen 3 (1.7B) shows moderate correctness (0.33) with excellent style (0.97) and moderate response time (0.65)
1. Qwen 3 (8B) shows better correctness (0.46) than its smaller version, with excellent style (0.92) and the slowest response time (0.25)

## Visual Analysis

The comparative performance metrics across models are visualized in the following boxplot:

![](figs/metrics_boxplot_by_model_v1.png)

After updating the LLM scoring prompt, the evaluation score were changed.

![](figs/metrics_boxplot_by_model_v2.png)
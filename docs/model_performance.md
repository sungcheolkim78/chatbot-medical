# Open-Source LLM Model Evaluation Framework

## Model Selection Criteria

This evaluation framework focuses on open-source LLM models deployed through Ollama with tool capabilities. The selection criteria prioritized models that could run efficiently on consumer-grade hardware with less than 10GB VRAM, limiting the parameter size to 8B or smaller.

### Evaluated Models
- Llama 3.1 (8B parameters)
- Llama 3.2 (3B parameters)
- Granite 3.2 (2B and 8B parameters)
- Qwen 3 (1.7B and 8B parameters)

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

| Model | Parameter Size | Accuracy | Average Response Time | Notes |
|-------|---------------|----------|----------------------|-------|
| Llama 3.1 | 8B | | | |
| Llama 3.2 | 3B | | | |
| Granite 3.2 | 2B | | | |
| Granite 3.2 | 8B | | | |
| Qwen 3 | 1.7B | | | Reasoning-focused model |
| Qwen 3 | 8B | | | Reasoning-focused model |

## Visual Analysis

The comparative performance metrics across models are visualized in the following boxplot:

![](figs/metrics_boxplot_by_model.png)

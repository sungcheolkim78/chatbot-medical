# Ollama Installation Guide

## Prerequisites
- Linux operating system
- Internet connection
- Terminal access

## Installation
1. Execute the installation script:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Model Installation
Install required models using the following commands:
```bash
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull granite3.2:2b
ollama pull granite3.3:2b
ollama pull qwen3:1.7b
ollama pull qwen3:8b
```

## Verification
To verify installed models:
```bash
ollama ls
```

## Additional Resources
For installation instructions on other operating systems, refer to the [ollama official documentation](https://ollama.com/download/linux).
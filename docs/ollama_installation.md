# Setting Ollama

The overall process is simple! Depending on the OS, you can find the installation instruction [here](https://ollama.com/download/linux)
In this document, we assume the linux system. 

Install using one command:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Download necessary models:
```bash
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull granite3.2:2b
ollama pull granite3.3:2b
ollama pull qwen3:1.7b
ollama pull qwen3:8b
```

To see the downloaded models:
```bash
ollama ls
```
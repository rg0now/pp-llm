# Pipeline Parallel Llama 3.1 1B - CPU Only

A simple implementation of pipeline parallelism for Llama 3.1 1B using CPU-only inference. Splits the model across two nodes connected via HTTP REST API.
- **Node 1**: Token embedding, position embedding, first 16 transformer layers
- **Node 2**: Remaining 16 layers, layer normalization, language modeling head, text generation

## Quick Start

### 1. Setup Environment
```bash
# Create Python virtual environment
python3 -m venv llama_pipeline_env
source llama_pipeline_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Accept Llama Model License
You need to request access to Llama models on HuggingFace:
1. Go to https://huggingface.co/meta-llama/Llama-3.1-1B
2. Click "Request access" and accept the license
3. Get your HuggingFace token: https://huggingface.co/settings/tokens

```bash
# Login to HuggingFace (one-time setup)
huggingface-cli login
# Enter your token when prompted
```

### 3. Run Pipeline Parallel Inference

**Terminal 1 - Start Node 2 (Final layers + Generation):**
```bash
source llama_pipeline_env/bin/activate
python node2.py --model meta-llama/Llama-3.1-1B --split-layer 16
```

**Terminal 2 - Start Node 1 (First layers + Embeddings):**
```bash
source llama_pipeline_env/bin/activate
python node1.py --model meta-llama/Llama-3.1-1B --split-layer 16
```

### 4. Test
```bash
# Single completion
curl -X POST http://localhost:5001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms:",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Multi-Machine Setup

### Machine 1 (Node 2):
```bash
# Start Node 2 on machine with IP 192.168.1.100
python node2.py --model meta-llama/Llama-3.1-1B --split-layer 16 --host 0.0.0.0 --port 5002
```

### Machine 2 (Node 1):
```bash
# Start Node 1, pointing to Node 2
python node1.py --model meta-llama/Llama-3.1-1B --split-layer 16 --node2-url http://192.168.1.100:5002 --host 0.0.0.0
```

## License

This code is provided for educational and research purposes. The Llama models are subject to Meta's license terms.


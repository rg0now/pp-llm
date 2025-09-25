# Pipeline Parallel Qwen 2.5 1.5B - CPU Only

A simple implementation of pipeline parallelism for Qwen 2.5 1.5B using CPU-only inference. Splits the model across two nodes connected via HTTP REST API.

```
[Client Request] 
       ↓
[Node 1: Embedding + Layers 0-15] 
       ↓ HTTP JSON
[Node 2: Layers 16-31 + Generation]
       ↓
[Generated Response]
```

Architecture overview:
- **Node 1**: Token embedding, position embedding, first 16 transformer layers
- **Node 2**: Remaining 16 layers, layer normalization, language modeling head, text generation

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv meta_pipeline_env
source meta_pipeline_env/bin/activate  # On Windows: meta_pipeline_env\Scripts\activate

# Install PyTorch CPU version first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt

# Log in to HF
huggingface-cli login
# Enter your token when prompted
```

### 2. Run Pipeline Parallel Inference

**Terminal 1 - Start Node 2 (Final layers + Generation):**
```bash
source meta_pipeline_env/bin/activate  # IMPORTANT: Always activate first!
python node2.py --model Qwen/Qwen2.5-1.5B-Instruct --split-layer 14
```

**Terminal 2 - Start Node 1 (First layers + Embeddings):**
```bash
source meta_pipeline_env/bin/activate  # IMPORTANT: Always activate first!
python node1.py --model Qwen/Qwen2.5-1.5B-Instruct --split-layer 14
```

## Usage

OpenAI-Compatible API:

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

## License

This code is provided for educational and research purposes. 


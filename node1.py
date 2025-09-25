#!/usr/bin/env python3
"""
Pipeline Parallel Node 1
Handles embedding + first N transformer layers
Provides OpenAI-compatible API
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import requests
import json
import time
import logging
import threading
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineNode1:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", split_layer=14, node2_url="http://localhost:5002"):
        """
        Initialize Node 1 with first part of Llama model
        
        Args:
            model_name: HuggingFace model name for Llama
            split_layer: Layer index to split at (Node 1 handles layers 0 to split_layer-1)
            node2_url: URL of Node 2's API endpoint
        """
        self.model_name = model_name
        self.split_layer = split_layer
        self.node2_url = node2_url
        
        logger.info(f"Initializing Node 1 with model: {model_name}")
        logger.info(f"Split at layer: {split_layer}")
        
        # Load tokenizer and config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Load full model to extract components
        logger.info("Loading full model (this may take a moment)...")
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # Extract Node 1 components
        self.embed_tokens = full_model.model.embed_tokens
        
        # Extract first N transformer layers  
        self.layers = nn.ModuleList(
            full_model.model.layers[:split_layer]
        )
        
        # Store model info for Node 2
        self.model_info = {
            "model_name": model_name,
            "vocab_size": self.config.vocab_size,
            "hidden_size": self.config.hidden_size,
            "num_attention_heads": self.config.num_attention_heads,
            "num_key_value_heads": getattr(self.config, 'num_key_value_heads', self.config.num_attention_heads),
            "intermediate_size": self.config.intermediate_size,
            "max_position_embeddings": self.config.max_position_embeddings,
            "rms_norm_eps": self.config.rms_norm_eps,
            "rope_theta": getattr(self.config, 'rope_theta', 10000.0),
            "split_layer": split_layer,
            "total_layers": self.config.num_hidden_layers
        }
        
        logger.info(f"Node 1 initialized with layers 0-{split_layer-1} of {self.config.num_hidden_layers} total")
        logger.info(f"Model parameters in Node 1: {sum(p.numel() for p in self.parameters()):,}")
    
    def parameters(self):
        """Get all parameters in Node 1"""
        for param in self.embed_tokens.parameters():
            yield param
        for layer in self.layers:
            for param in layer.parameters():
                yield param
    
    def forward_partial(self, input_ids, attention_mask=None):
        """
        Forward pass through Node 1's components
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            dict: Intermediate data for Node 2
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # Create position IDs
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            # Handle different return formats
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs
        
        return {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask, 
            "position_ids": position_ids,
            "input_ids": input_ids,
        }
    
    def send_to_node2(self, intermediate_data, generation_params=None):
        """Send intermediate data to Node 2 for completion"""
        if generation_params is None:
            generation_params = {"max_new_tokens": 50, "temperature": 0.7, "do_sample": True}
        
        payload = {
            "hidden_states": intermediate_data["hidden_states"].tolist(),
            "attention_mask": intermediate_data["attention_mask"].tolist(),
            "position_ids": intermediate_data["position_ids"].tolist(), 
            "input_ids": intermediate_data["input_ids"].tolist(),
            "model_info": self.model_info,
            "generation_params": generation_params
        }
        
        try:
            logger.info(f"Sending intermediate data to Node 2: {self.node2_url}/generate")
            response = requests.post(
                f"{self.node2_url}/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with Node 2: {e}")
            return {"error": str(e)}
    
    def generate(self, prompt, **generation_params):
        """End-to-end generation through the pipeline"""
        logger.info(f"Starting generation for prompt: '{prompt[:50]}...'")
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(1024, self.config.max_position_embeddings // 2)
        )
        
        # Forward through Node 1
        with torch.no_grad():
            intermediate_data = self.forward_partial(
                inputs["input_ids"],
                inputs.get("attention_mask", None)
            )
        
        # Send to Node 2
        result = self.send_to_node2(intermediate_data, generation_params)
        
        if "error" in result:
            return f"Generation failed: {result['error']}"
        
        return result.get("generated_text", "No text generated")


# Flask API
app = Flask(__name__)
node1_instance = None

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "node": "1", 
        "model": node1_instance.model_name if node1_instance else "not_loaded",
        "ready": node1_instance is not None
    })

@app.route("/v1/models", methods=["GET"])
def list_models():
    """OpenAI-compatible models endpoint"""
    if not node1_instance:
        return jsonify({"error": {"message": "Service unavailable", "type": "service_error"}}), 500
    
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": node1_instance.model_name,
                "object": "model", 
                "created": int(time.time()),
                "owned_by": "pipeline-parallel",
                "permission": [],
                "root": node1_instance.model_name,
                "parent": None
            }
        ]
    })

@app.route("/v1/completions", methods=["POST"])
def create_completion():
    """OpenAI-compatible completions endpoint"""
    try:
        data = request.get_json()
        
        # Extract OpenAI parameters
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 50)
        temperature = data.get("temperature", 1.0)
        top_p = data.get("top_p", 1.0)
        n = data.get("n", 1)
        stream = data.get("stream", False)
        
        if not prompt:
            return jsonify({"error": {"message": "No prompt provided", "type": "invalid_request_error"}}), 400
        
        if not node1_instance:
            return jsonify({"error": {"message": "Service unavailable", "type": "service_error"}}), 500
        
        if stream or n > 1:
            return jsonify({"error": {"message": "Streaming and n>1 not supported", "type": "invalid_request_error"}}), 400
        
        # Convert to generation params
        generation_params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "top_p": top_p
        }
        
        # Generate
        start_time = time.time()
        result = node1_instance.generate(prompt, **generation_params)
        end_time = time.time()
        
        if isinstance(result, str) and result.startswith("Generation failed"):
            return jsonify({"error": {"message": result, "type": "server_error"}}), 500
        
        # Calculate token usage
        prompt_tokens = len(node1_instance.tokenizer.encode(prompt))
        completion_tokens = len(node1_instance.tokenizer.encode(result))
        total_tokens = prompt_tokens + completion_tokens
        
        return jsonify({
            "id": f"cmpl-{int(time.time() * 1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": node1_instance.model_name,
            "choices": [
                {
                    "text": result,
                    "index": 0,
                    "finish_reason": "length",
                    "logprobs": None
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, 
                "total_tokens": total_tokens
            }
        })
        
    except Exception as e:
        logger.error(f"Completion error: {e}")
        return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500

@app.route("/config", methods=["GET"])
def get_config():
    """Get Node 1 configuration"""
    if node1_instance:
        return jsonify({
            **node1_instance.model_info,
            "status": "ready",
            "node": "1"
        })
    return jsonify({"error": "Node not initialized"}), 500

def initialize_node1(model_name, split_layer, node2_url):
    """Initialize Node 1 in separate thread"""
    global node1_instance
    try:
        node1_instance = PipelineNode1(model_name, split_layer, node2_url)
        logger.info("Node 1 initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize Node 1: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Parallel Node 1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Model name")
    parser.add_argument("--split-layer", type=int, default=14, help="Layer to split at") 
    parser.add_argument("--node2-url", default="http://localhost:5002", help="Node 2 URL")
    parser.add_argument("--port", type=int, default=5001, help="Port for Node 1 API")
    parser.add_argument("--host", default="0.0.0.0", help="Host for Node 1 API")
    
    args = parser.parse_args()
    
    # Initialize in background thread
    init_thread = threading.Thread(
        target=initialize_node1,
        args=(args.model, args.split_layer, args.node2_url)
    )
    init_thread.daemon = True
    init_thread.start()
    
    logger.info(f"Starting Node 1 API server on {args.host}:{args.port}")
    logger.info("OpenAI-compatible endpoints:")
    logger.info(f"  POST http://{args.host}:{args.port}/v1/completions")
    logger.info(f"  GET  http://{args.host}:{args.port}/v1/models")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)

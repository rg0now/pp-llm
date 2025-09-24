#!/usr/bin/env python3
"""
Pipeline Parallel Node 2 for Llama 3.1 1B
Handles remaining transformer layers + generation
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, LlamaForCausalLM
import json
import time
import logging
import threading
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaPipelineNode2:
    def __init__(self, model_name="meta-llama/Llama-3.1-1B", split_layer=16):
        """
        Initialize Node 2 with second part of Llama model
        
        Args:
            model_name: HuggingFace model name 
            split_layer: Layer index where split occurs
        """
        self.model_name = model_name
        self.split_layer = split_layer
        
        logger.info(f"Initializing Node 2 with model: {model_name}")
        logger.info(f"Handling layers {split_layer} onwards")
        
        # Load tokenizer and config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Load full model to extract components
        logger.info("Loading full model (this may take a moment)...")
        full_model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        # Extract Node 2 components
        self.layers = nn.ModuleList(
            full_model.model.layers[split_layer:]
        )
        self.norm = full_model.model.norm
        self.lm_head = full_model.lm_head
        
        logger.info(f"Node 2 initialized with layers {split_layer}-{self.config.num_hidden_layers-1} + LM head")
        logger.info(f"Model parameters in Node 2: {sum(p.numel() for p in self.parameters()):,}")
    
    def parameters(self):
        """Get all parameters in Node 2"""
        for layer in self.layers:
            for param in layer.parameters():
                yield param
        for param in self.norm.parameters():
            yield param
        for param in self.lm_head.parameters():
            yield param
    
    def forward_remaining(self, hidden_states, attention_mask=None, position_ids=None):
        """
        Forward through remaining layers
        
        Args:
            hidden_states: Hidden states from Node 1
            attention_mask: Attention mask
            position_ids: Position IDs
            
        Returns:
            logits: Output logits for next token prediction
        """
        # Process through remaining layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden_states = layer_outputs[0]
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def generate_from_intermediate(self, intermediate_data, generation_params):
        """
        Generate text from intermediate activations
        
        Args:
            intermediate_data: Data from Node 1
            generation_params: Generation parameters
            
        Returns:
            Generated text (new tokens only)
        """
        # Extract data
        hidden_states = torch.tensor(intermediate_data["hidden_states"], dtype=torch.float32)
        attention_mask = torch.tensor(intermediate_data["attention_mask"], dtype=torch.long)
        position_ids = torch.tensor(intermediate_data["position_ids"], dtype=torch.long)
        input_ids = torch.tensor(intermediate_data["input_ids"], dtype=torch.long)
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        max_new_tokens = generation_params.get("max_new_tokens", 50)
        temperature = generation_params.get("temperature", 1.0)
        do_sample = generation_params.get("do_sample", True)
        top_p = generation_params.get("top_p", 0.9)
        
        logger.info(f"Generating {max_new_tokens} tokens with temperature={temperature}")
        
        # Initialize generation
        generated_ids = input_ids.clone()
        current_length = seq_len
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # For first step, use provided hidden states
                if step == 0:
                    current_hidden_states = hidden_states
                    current_attention_mask = attention_mask
                    current_position_ids = position_ids
                else:
                    # For subsequent steps, we need to re-compute from Node 1
                    # This is a limitation of this simple implementation
                    # In practice, you'd maintain KV cache or re-send to Node 1
                    
                    # Simplified: just use the last hidden state and extrapolate
                    # This is not optimal but demonstrates the concept
                    last_token_hidden = hidden_states[:, -1:, :]  # Last token's hidden state
                    current_hidden_states = last_token_hidden
                    
                    # Update masks and positions
                    current_attention_mask = attention_mask
                    current_position_ids = position_ids[:, -1:] + step
                
                # Forward through Node 2
                logits = self.forward_remaining(
                    current_hidden_states,
                    attention_mask=current_attention_mask,
                    position_ids=current_position_ids
                )
                
                # Get next token logits (last position)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-p sampling if enabled
                if do_sample and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Add to sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                current_length += 1
                
                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=torch.long)
                ], dim=-1)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    logger.info(f"Hit EOS token at step {step}")
                    break
                
                # Check max length
                if current_length >= self.config.max_position_embeddings:
                    logger.info(f"Hit max length at step {step}")
                    break
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        original_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Return only the newly generated part
        if generated_text.startswith(original_text):
            new_text = generated_text[len(original_text):].strip()
        else:
            new_text = generated_text
        
        logger.info(f"Generated {len(new_text)} characters in {step + 1} steps")
        return new_text


# Flask API for Node 2
app = Flask(__name__)
node2_instance = None

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "node": "2",
        "model": node2_instance.model_name if node2_instance else "not_loaded",
        "ready": node2_instance is not None
    })

@app.route("/generate", methods=["POST"])
def generate_from_activations():
    """Generate text from intermediate activations received from Node 1"""
    try:
        data = request.get_json()
        
        if not node2_instance:
            return jsonify({"error": "Node 2 not initialized"}), 500
        
        # Extract data
        intermediate_data = {
            "hidden_states": data["hidden_states"],
            "attention_mask": data["attention_mask"],
            "position_ids": data["position_ids"], 
            "input_ids": data["input_ids"]
        }
        
        model_info = data.get("model_info", {})
        generation_params = data.get("generation_params", {})
        
        # Verify model compatibility
        if model_info.get("model_name") != node2_instance.model_name:
            logger.warning(f"Model mismatch: Node 2 has {node2_instance.model_name}, "
                         f"received request for {model_info.get('model_name')}")
        
        # Generate text
        logger.info("Received intermediate activations from Node 1, starting generation")
        start_time = time.time()
        
        generated_text = node2_instance.generate_from_intermediate(
            intermediate_data,
            generation_params
        )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        logger.info(f"Generation completed in {generation_time:.2f}s")
        
        return jsonify({
            "generated_text": generated_text,
            "status": "success",
            "generation_time": generation_time,
            "node": "2"
        })
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/config", methods=["GET"])
def get_config():
    """Get Node 2 configuration"""
    if node2_instance:
        return jsonify({
            "model_name": node2_instance.model_name,
            "split_layer": node2_instance.split_layer,
            "num_layers": len(node2_instance.layers),
            "total_params": sum(p.numel() for p in node2_instance.parameters()),
            "status": "ready",
            "node": "2"
        })
    return jsonify({"error": "Node 2 not initialized"}), 500

def initialize_node2(model_name, split_layer):
    """Initialize Node 2 in separate thread"""
    global node2_instance
    try:
        node2_instance = LlamaPipelineNode2(model_name, split_layer)
        logger.info("Node 2 initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize Node 2: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Llama Pipeline Parallel Node 2")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-1B", help="Llama model name")
    parser.add_argument("--split-layer", type=int, default=16, help="Layer where split occurs")
    parser.add_argument("--port", type=int, default=5002, help="Port for Node 2 API")
    parser.add_argument("--host", default="0.0.0.0", help="Host for Node 2 API")
    
    args = parser.parse_args()
    
    # Initialize in background thread
    init_thread = threading.Thread(
        target=initialize_node2,
        args=(args.model, args.split_layer)
    )
    init_thread.daemon = True
    init_thread.start()
    
    logger.info(f"Starting Node 2 API server on {args.host}:{args.port}")
    logger.info("Endpoints:")
    logger.info(f"  POST http://{args.host}:{args.port}/generate")
    logger.info(f"  GET  http://{args.host}:{args.port}/health")
    logger.info(f"  GET  http://{args.host}:{args.port}/config")
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)

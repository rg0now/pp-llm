#!/usr/bin/env python3
"""
Test client for Pipeline Parallel setup - Universal Model Support
Supports OpenAI-compatible API testing and concurrency testing
"""

import requests
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_node_status(node_url, node_name):
    """Check if a node is ready"""
    try:
        response = requests.get(f"{node_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            ready = data.get("ready", False)
            model = data.get("model", "unknown")
            
            print(f"{node_name}: {status} - Model: {model} - Ready: {ready}")
            return ready
        else:
            print(f"{node_name}: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"{node_name}: Connection failed - {e}")
        return False

def wait_for_nodes(node1_url="http://localhost:5001", node2_url="http://localhost:5002", timeout=60):
    """Wait for both nodes to be ready"""
    print("Waiting for nodes to initialize...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        node1_ready = check_node_status(node1_url, "Node 1")
        node2_ready = check_node_status(node2_url, "Node 2")
        
        if node1_ready and node2_ready:
            print("✓ Both nodes are ready!")
            return True
        
        print("Waiting for nodes to finish loading...")
        time.sleep(5)
    
    print("✗ Timeout waiting for nodes to be ready")
    return False

def test_openai_completion(node1_url, prompt, **params):
    """Test OpenAI-compatible completion"""
    payload = {
        "prompt": prompt,
        "max_tokens": params.get("max_tokens", 50),
        "temperature": params.get("temperature", 0.7),
        "top_p": params.get("top_p", 0.9),
    }
    
    try:
        print(f"Testing completion: '{prompt[:50]}...'")
        start_time = time.time()
        
        response = requests.post(
            f"{node1_url}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        end_time = time.time()
        response.raise_for_status()
        
        result = response.json()
        
        print(f"\n{'='*60}")
        print(f"COMPLETION RESULT")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        
        if 'choices' in result and len(result['choices']) > 0:
            generated = result['choices'][0]['text']
            print(f"Generated: {generated}")
            print(f"Finish reason: {result['choices'][0]['finish_reason']}")
        
        if 'usage' in result:
            usage = result['usage']
            print(f"Tokens: {usage['prompt_tokens']} + {usage['completion_tokens']} = {usage['total_tokens']}")
        
        print(f"Time: {end_time - start_time:.2f}s")
        print(f"{'='*60}\n")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Response text: {e.response.text}")
        return None

def test_concurrent_requests(node1_url, num_threads=3, requests_per_thread=2):
    """Test concurrent request handling"""
    print(f"\nTesting {num_threads} concurrent threads, {requests_per_thread} requests each")
    print("=" * 70)
    
    prompts = [
        "The future of artificial intelligence is",
        "Explain quantum computing:",
        "Write a short story about a robot:",
        "The benefits of renewable energy include",
        "In the year 2050, technology will",
        "Climate change can be addressed by",
        "The history of space exploration shows",
        "Machine learning algorithms work by"
    ]
    
    def worker_thread(thread_id):
        """Worker function for concurrent testing"""
        results = []
        thread_start = time.time()
        
        for req_num in range(requests_per_thread):
            prompt = prompts[(thread_id * requests_per_thread + req_num) % len(prompts)]
            
            try:
                result = test_openai_completion(
                    node1_url,
                    prompt,
                    max_tokens=30,
                    temperature=0.7
                )
                
                if result and 'choices' in result:
                    generated = result['choices'][0]['text'][:50] + "..." if len(result['choices'][0]['text']) > 50 else result['choices'][0]['text']
                    results.append({
                        'thread_id': thread_id,
                        'request_num': req_num,
                        'success': True,
                        'prompt': prompt,
                        'generated': generated,
                        'tokens': result.get('usage', {}).get('total_tokens', 0)
                    })
                    print(f"Thread {thread_id}, Request {req_num + 1}: ✓ Success")
                else:
                    results.append({
                        'thread_id': thread_id,
                        'request_num': req_num,
                        'success': False,
                        'prompt': prompt,
                        'error': 'No valid response'
                    })
                    print(f"Thread {thread_id}, Request {req_num + 1}: ✗ Failed")
                    
            except Exception as e:
                results.append({
                    'thread_id': thread_id,
                    'request_num': req_num,
                    'success': False,
                    'prompt': prompt,
                    'error': str(e)
                })
                print(f"Thread {thread_id}, Request {req_num + 1}: ✗ Error: {e}")
        
        thread_time = time.time() - thread_start
        print(f"Thread {thread_id} completed in {thread_time:.2f}s")
        return results, thread_time
    
    # Run concurrent threads
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(worker_thread, i): i for i in range(num_threads)}
        
        all_results = []
        thread_times = []
        
        for future in as_completed(futures):
            try:
                results, thread_time = future.result()
                all_results.extend(results)
                thread_times.append(thread_time)
            except Exception as e:
                print(f"Thread error: {e}")
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = sum(1 for r in all_results if r['success'])
    total = len(all_results)
    total_tokens = sum(r.get('tokens', 0) for r in all_results if r['success'])
    
    print(f"\nCONCURRENCY TEST RESULTS:")
    print(f"{'='*70}")
    print(f"Total requests: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success rate: {successful/total*100:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests/second: {total/total_time:.2f}")
    if total_tokens > 0:
        print(f"Total tokens: {total_tokens}")
        print(f"Tokens/second: {total_tokens/total_time:.2f}")
    print(f"{'='*70}")
    
    # Show examples
    if successful > 0:
        print(f"\nSample generations:")
        for i, result in enumerate([r for r in all_results if r['success']][:3]):
            print(f"{i+1}. '{result['prompt'][:40]}...'")
            print(f"   → '{result['generated']}'")

def main():
    parser = argparse.ArgumentParser(description="Llama Pipeline Test Client")
    parser.add_argument("--node1-url", default="http://localhost:5001", help="Node 1 URL")
    parser.add_argument("--node2-url", default="http://localhost:5002", help="Node 2 URL")
    parser.add_argument("--prompt", default="The future of artificial intelligence is", help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--concurrency-test", action="store_true", help="Run concurrency test")
    parser.add_argument("--threads", type=int, default=3, help="Number of concurrent threads")
    parser.add_argument("--requests-per-thread", type=int, default=2, help="Requests per thread")
    parser.add_argument("--skip-wait", action="store_true", help="Skip waiting for nodes to be ready")
    
    args = parser.parse_args()
    
    print("Pipeline Parallel Test Client - Universal Model Support")
    print("=" * 50)
    
    # Wait for nodes to be ready (unless skipped)
    if not args.skip_wait:
        if not wait_for_nodes(args.node1_url, args.node2_url):
            print("Nodes are not ready. Use --skip-wait to test anyway.")
            return
    
    # Check models endpoint
    try:
        response = requests.get(f"{args.node1_url}/v1/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"Available models: {[m['id'] for m in models['data']]}")
        else:
            print("Could not fetch models list")
    except Exception as e:
        print(f"Error fetching models: {e}")
    
    if args.concurrency_test:
        # Run concurrency test
        test_concurrent_requests(args.node1_url, args.threads, args.requests_per_thread)
    else:
        # Single test
        test_openai_completion(
            args.node1_url,
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )

if __name__ == "__main__":
    main()

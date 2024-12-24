import json
import unittest
import requests
import psutil
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

def get_gpu_memory():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Adjust if using multiple GPUs
        torch.cuda.empty_cache()  # Optional, clear unused cached memory
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # Convert to MB
        return memory_allocated, memory_reserved
    else:
        return 0, 0

class TestInputEmbeds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = '/home/jupyter/LLMASR/model'
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model)
        cls.ref_model = AutoModelForCausalLM.from_pretrained(cls.model)
        cls.embed_file_path = "/home/jupyter/LLMASR/test_embeds.txt"
        cls.constant_args = ["--disable-radix"]  # Always used
        cls.configs = [
            # Baseline Configuration
            [],
        ]
        cls.results = []  # To store results for summary
        cls.log_file = "results_log.json"  # Log file name

    def log_device_status(self):
        """Log memory and CUDA usage."""
        system_memory = psutil.virtual_memory()
        gpu_memory_allocated, gpu_memory_reserved = get_gpu_memory()
        return {
            "system_memory_used": system_memory.used / 1024**2,  # In MB
            "system_memory_total": system_memory.total / 1024**2,  # In MB
            "gpu_memory_allocated": gpu_memory_allocated / 1024**2,
            "gpu_memory_reserved": gpu_memory_reserved / 1024**2,
        }

    def load_input_embeddings(self):
        """Load embeddings from the file."""
        embeddings_list = []
        with open(self.embed_file_path, "r") as f:
            for line in f:
                try:
                    embeddings = json.loads(line.strip())  # Parse each line as JSON
                    embeddings_list.append(embeddings)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line}. Error: {e}")
        return embeddings_list

    def send_request(self, payload):
        """Send a POST request to the API and return the response."""
        start_time = time.time()
        response = requests.post(
            self.base_url + "/generate",
            json=payload,
            timeout=30,  # Set a reasonable timeout for the API request
        )
        elapsed_time = time.time() - start_time
        if response.status_code == 200:
            return response.json(), elapsed_time
        return {"error": f"Request failed with status {response.status_code}: {response.text}"}, elapsed_time

    def run_with_config(self, config):
        """Run test with a specific configuration."""
        print(f"Testing with config: {config}")
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=self.constant_args + config,
        )

        embeddings_list = self.load_input_embeddings()
        inference_times = []
        for idx, embeddings in enumerate(embeddings_list):
            payload = {
                "model": self.model,
                "input_embeds": embeddings,
                "sampling_params": {"temperature": 0, "max_new_tokens": 100},
            }
            response, elapsed_time = self.send_request(payload)
            inference_times.append(elapsed_time)
            print(
                f"Embeddings Input (line {idx + 1}):\nResponse: {json.dumps(response, indent=2)}\n{'-' * 80}\nInference time: {elapsed_time:.4f} seconds"
            )

        # Log device status and summary
        device_status = self.log_device_status()
        average_inference_time = sum(inference_times) / len(inference_times)
        config_result = {
            "config": config,
            "average_inference_time": average_inference_time,
            **device_status,
        }
        self.results.append(config_result)

        # Save results to file
        self.save_results()

        kill_process_tree(process.pid)  # Clean up

    def save_results(self):
        """Save the results to a JSON file."""
        with open(self.log_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {self.log_file}")

    def test_embedding_based_responses(self):
        """Run through all configurations and summarize results."""
        for config in self.configs:
            self.run_with_config(config)

        # Summarize results
        print("\n\nSummary of Results:")
        for result in self.results:
            print(json.dumps(result, indent=2))

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()

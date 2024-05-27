
import argparse
import time
import torch
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

np.random.seed(777)

def test(args):
    # init model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.float16,
                                                 device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(args.model_path)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()

    def run_to_completion():
        # NOTE: batchsize = 1
        dummy_prompt_token_ids = np.random.randint(10000,
                                                   size=(1, args.input_len))
        dummy_prompt_token_ids = torch.tensor(dummy_prompt_token_ids)
        inputs = {'input_ids': dummy_prompt_token_ids,
                  'attention_mask': torch.ones(1, args.input_len)}
        with torch.no_grad():
            start_time = time.perf_counter()
            outputs = model.generate(dummy_prompt_token_ids.to(model.device),
                                     max_new_tokens=args.output_len)
            end_time = time.perf_counter()
            latency = end_time - start_time
        return latency, outputs[0].shape[0] - dummy_prompt_token_ids.shape[1]

    # warm up
    for _ in tqdm(range(args.num_iters_warmup)):
        run_to_completion()

    latencies = []
    gen_token_len = []
    for _ in tqdm(range(args.num_iters)):
        latency, gen_len = run_to_completion()
        latencies.append(latency)
        gen_token_len.append(gen_len)
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90]
    percentiles = np.percentile(latencies, percentages)
    print(f'Avg latency: {np.mean(latencies)} seconds')
    print(f'Sum latency: {np.sum(latencies)} seconds')
    print(f'generate tokens: {np.sum(gen_token_len)}')
    for percentage, percentile in zip(percentages, percentiles):
        print(f'{percentage}% percentile latency: {percentile} seconds')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model_path', type=str, default='facebook/opt-125m')
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=10,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=10,
                        help='Number of iterations to run.')
    args = parser.parse_args()
    test(args)

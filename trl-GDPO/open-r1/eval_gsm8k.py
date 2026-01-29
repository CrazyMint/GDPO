#!/usr/bin/env python3
"""Evaluate trained model on GSM8K test set using vLLM for fast batched inference."""

import argparse
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = """
You are a helpful AI assistant.

For every request, you should carefully think through the math problem step by step, then provide the fianl answer in integer format.

Steps for Each Request:
1. Think: Provide detailed, step-by-step reasoning, calculations, or derivations.
2. Produce Final Answer: After step-by-step reasoning, output the final answer in integer format.

Output Format:
<think>Your thoughts and reasoning</think>
<answer>Final answer in integer format</answer>

Important Notes:
1. You must include your reasoning steps inside <think>.
2. You must always output the Final Answer within <answer> after the reasoning steps is done.
3. You should consistently work through the solution step by step before giving the final answer.
4. The final answer can only be an integer.
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (default: all)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max tokens to generate")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for vLLM inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization for vLLM")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} with vLLM...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,  # Greedy decoding
    )

    print("Loading GSM8K test set...")
    test_data = load_dataset("openai/gsm8k", "main")["test"]

    if args.max_samples:
        test_data = test_data.select(range(min(args.max_samples, len(test_data))))

    # Prepare all prompts
    print(f"Preparing {len(test_data)} prompts...")
    prompts = []
    gold_answers = []
    questions = []

    for item in test_data:
        question = item["question"]
        gold_answer = extract_hash_answer(item["answer"])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        gold_answers.append(gold_answer)
        questions.append(question)

    # Batched inference with vLLM
    print(f"Running batched inference with vLLM (batch_size={args.batch_size})...")
    correct = 0
    total = 0

    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i:i + args.batch_size]
        batch_gold = gold_answers[i:i + args.batch_size]
        batch_questions = questions[i:i + args.batch_size]

        outputs = llm.generate(batch_prompts, sampling_params)

        for j, output in enumerate(outputs):
            response = output.outputs[0].text
            predicted = extract_xml_answer(response)
            gold = batch_gold[j]

            if predicted == gold:
                correct += 1
            total += 1

        # Print progress every batch
        print(f"\nBatch {i//args.batch_size + 1}: Running Accuracy: {correct/total:.2%} ({correct}/{total})")

    print("\n" + "="*50)
    print(f"Final Results on GSM8K Test Set:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {correct/total:.2%}")
    print("="*50)


if __name__ == "__main__":
    main()

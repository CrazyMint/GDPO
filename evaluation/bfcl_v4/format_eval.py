"""
Standalone "Correct Format" evaluator on BFCL test data.

Loads a model checkpoint via vLLM offline batch inference, generates raw outputs
on BFCL single-turn test prompts (using the RLLA training-time prompt format from
rlla_qwen.py), and checks if outputs match the expected <think>/<tool_call> format.

Usage:
    python bfcl_v4/format_eval.py \
        --model_path /path/to/checkpoint \
        --model_id toolrl_gdpo_step_100 \
        --output_dir /path/to/output
"""

import argparse
import copy
import json
import os
import re
from pathlib import Path

BFCL_DATA_DIR = Path(__file__).parent.parent / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"

# All single-turn BFCL v4 categories (skip multi_turn which needs tool execution between turns)
SINGLE_TURN_CATEGORIES = [
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "irrelevance",
    "live_irrelevance",
    "live_relevance",
]

VERSION_PREFIX = "BFCL_v4"

# JSON example for tool call format (from rlla_qwen.py)
JSON_STRING = """{"name": "Tool name", "parameters": {"Parameter name": "Parameter content", "... ...": "... ..."}}
{"name": "... ...", "parameters": {"... ...": "... ...", "... ...": "... ..."}}"""

# RLLA training-time system prompt (from rlla_qwen.py) - this is what the model
# was trained with, instructing <think>/<tool_call>/<response> format.
# The paper's BFCL eval uses this prompt via the RLLAHandler.
SYSTEM_PROMPT_TEMPLATE = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

**Available Tools**
In your response, you can use the following tools:
{tools}

**Steps for Each Turn**
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.
3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

**Output Format**
```plaintext
<think> Your thoughts and reasoning </think>
<tool_call>
{json_string}
...
</tool_call>
<response> AI's final response </response>
```

**Important Notes**
1. You must always include the `<think>` field to outline your reasoning. Provide at least one of `<tool_call>` or `<response>`. Decide whether to use `<tool_call>` (possibly multiple times), `<response>`, or both.
2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields. Each tool call should be a JSON object with a "name" field and an "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.
3. Refer to the previous dialogue records in the history, including the user's queries, previous `<tool_call>`, `<response>`, and any tool feedback noted as `<obs>` (if exists).
"""


def convert_to_format_tool(tools, count=1):
    """Convert BFCL function schema to RLLA numbered tool format (from rlla_qwen.py)."""
    if isinstance(tools, dict):
        format_tools = {
            "name": tools["name"],
            "description": tools["description"],
            "parameters": tools["parameters"].get("properties", {}),
        }
        tool_string = f"{count}. Name: {format_tools['name']}\nDescription: {format_tools['description']}\nParameters: {json.dumps(format_tools['parameters'])}"
        return tool_string
    elif isinstance(tools, list):
        tool_list = [convert_to_format_tool(tool, idx + 1) for idx, tool in enumerate(tools)]
        return "\n".join(tool_list)
    else:
        return tools


def load_test_data(categories):
    """Load BFCL test entries for specified categories."""
    all_entries = []
    for cat in categories:
        fpath = BFCL_DATA_DIR / f"{VERSION_PREFIX}_{cat}.json"
        if not fpath.exists():
            print(f"WARNING: {fpath} not found, skipping")
            continue
        with open(fpath) as f:
            entries = [json.loads(line) for line in f]
        for e in entries:
            e["_category"] = cat
        all_entries.extend(entries)
        print(f"  Loaded {len(entries)} entries from {cat}")
    return all_entries


def build_prompt(entry):
    """Build the Qwen2.5 chat prompt from a BFCL test entry, using the RLLA prompt format.

    Uses the RLLA training-time system prompt (from rlla_qwen.py _format_prompt())
    so the model outputs <think>/<tool_call> tags for format checking.
    """
    messages = copy.deepcopy(entry["question"][0])  # first turn messages
    functions = entry["function"]

    # Convert function schemas to RLLA numbered format
    tools_str = convert_to_format_tool(functions)
    system_content = SYSTEM_PROMPT_TEMPLATE.format(tools=tools_str, json_string=JSON_STRING)

    # Build user prompt (matching rlla_qwen.py single-turn format)
    user_prompt = "**Dialogue Records History**\n"
    for msg in messages:
        if msg["role"] == "system":
            continue
        elif msg["role"] == "user":
            user_prompt += (
                f"<user> {msg['content'].strip()}\n"
                "If there's no appropriate tools to apply or required parameters are missing, "
                "please directly inform me in your response without any tool call, or call the "
                "tool with the name as 'None'. Otherwise, you should use one or more necessary "
                "tool calls to complete the given task in this turn. </user>\n"
            )
        elif msg["role"] == "assistant":
            user_prompt += f"\n{msg['content'].strip()}\n"
    user_prompt = user_prompt.strip()

    # Format as Qwen2.5 chat template
    formatted = f"<|im_start|>system\n{system_content}<|im_end|>\n"
    formatted += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted


def check_format(response):
    """
    Check if response matches any valid RLLA output format:
      1. tool_call only: <think>...</think>\n<tool_call>\n...\n</tool_call>
      2. response only:  <think>...</think>\n<response>...</response>
      3. both:           <think>...</think>\n<tool_call>\n...\n</tool_call>\n<response>...</response>
      4. think only:     <think>...</think>
    Returns True if format is correct.
    """
    # Case 3: both tool_call and response
    if "<tool_call>" in response and "<response>" in response:
        pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>$"
        return (
            bool(re.search(pattern, response, re.DOTALL))
            and response.count("<tool_call>") == 1
            and response.count("</tool_call>") == 1
            and response.count("<response>") == 1
            and response.count("</response>") == 1
        )
    # Case 1: tool_call only
    elif "<tool_call>" in response:
        pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$"
        return (
            bool(re.search(pattern, response, re.DOTALL))
            and response.count("<tool_call>") == 1
            and response.count("</tool_call>") == 1
        )
    # Case 2: response only
    elif "<response>" in response:
        pattern = r"^<think>.*?</think>\n<response>.*?</response>$"
        return bool(re.search(pattern, response, re.DOTALL))
    # Case 4: think only
    else:
        pattern = r"^<think>.*?</think>$"
        return bool(re.search(pattern, response, re.DOTALL))


def main():
    parser = argparse.ArgumentParser(description="Evaluate format correctness on BFCL data")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--model_id", required=True, help="Model identifier for output")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--categories", nargs="+", default=SINGLE_TURN_CATEGORIES,
                        help="BFCL categories to evaluate")
    args = parser.parse_args()

    # Load test data
    print(f"Loading BFCL test data for {len(args.categories)} categories...")
    entries = load_test_data(args.categories)
    print(f"Total entries: {len(entries)}")

    # Build prompts (using RLLA prompt format)
    print("Building prompts (RLLA format)...")
    prompts = [build_prompt(e) for e in entries]
    print(f"Built {len(prompts)} prompts")

    # Load model with vLLM offline
    print(f"Loading model from {args.model_path}...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Generate
    print("Generating outputs...")
    outputs = llm.generate(prompts, sampling_params)
    print(f"Generated {len(outputs)} outputs")

    # Check format and collect results
    per_category = {}
    raw_results = []

    for entry, output in zip(entries, outputs):
        response = output.outputs[0].text
        cat = entry["_category"]
        fmt_ok = check_format(response)

        if cat not in per_category:
            per_category[cat] = {"total": 0, "correct": 0}
        per_category[cat]["total"] += 1
        if fmt_ok:
            per_category[cat]["correct"] += 1

        raw_results.append({
            "id": entry["id"],
            "category": cat,
            "format_correct": fmt_ok,
            "response": response,
        })

    # Compute summary
    total = sum(v["total"] for v in per_category.values())
    correct = sum(v["correct"] for v in per_category.values())
    format_pct = correct / total * 100 if total > 0 else 0

    for cat in per_category:
        c = per_category[cat]
        c["pct"] = round(c["correct"] / c["total"] * 100, 2) if c["total"] > 0 else 0

    summary = {
        "model": args.model_id,
        "model_path": args.model_path,
        "total": total,
        "format_correct": correct,
        "format_pct": round(format_pct, 2),
        "per_category": per_category,
    }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, f"{args.model_id}_format_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    raw_path = os.path.join(args.output_dir, f"{args.model_id}_format_raw.jsonl")
    with open(raw_path, "w") as f:
        for r in raw_results:
            f.write(json.dumps(r) + "\n")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Format Evaluation: {args.model_id}")
    print(f"{'='*60}")
    print(f"Overall: {correct}/{total} = {format_pct:.2f}%")
    print(f"{'-'*60}")
    for cat in sorted(per_category.keys()):
        c = per_category[cat]
        print(f"  {cat:30s}: {c['correct']:>4}/{c['total']:>4} = {c['pct']:>6.2f}%")
    print(f"\nSaved: {summary_path}")
    print(f"Raw:   {raw_path}")


if __name__ == "__main__":
    main()

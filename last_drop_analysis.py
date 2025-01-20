import os
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from textwrap import fill
import math

from transformers import AutoTokenizer

plt.rcParams['font.family'] = ['Droid Sans Fallback', 'DejaVu Sans', 'Noto Mono', 'SimHei']

BASE_DIR = "analysis/last_drop"

def find_unanalyzed_directories(base_dir):
    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    unanalyzed = [
        d for d in subdirs
        if not os.path.exists(os.path.join(d, "last_drop_analysis.pdf"))
    ]
    return unanalyzed

def load_jsonl_file(jsonl_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze_directory(subdir, max_calls=100):
    jsonl_path = os.path.join(subdir, "last_drop.jsonl")
    pdf_path = os.path.join(subdir, "last_drop_analysis.pdf")

    if not os.path.exists(jsonl_path):
        print(f"No last_drop.jsonl in {subdir}, skipping.")
        return

    data = load_jsonl_file(jsonl_path)[:max_calls]

    model_name = data[0].get("model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pdf = None
    try:
        pdf = PdfPages(pdf_path)
        for datum in data:
            fig = generate_plot(datum, tokenizer)
            if fig:
                pdf.savefig(fig)
                plt.close(fig)
    except Exception as e:
        print(f"Error analyzing {subdir}: {e}")
    finally:
        if pdf:
            pdf.close()

    print(f"Analysis done for {subdir} => {pdf_path}")

def generate_plot(data, tokenizer):
    """
    Creates a single figure for one call with scatter plots:
      - Red scatter for original topk_logits
      - Black scatter for tokens i where (topk_logits[i] - topk_logits[i+1]) > threshold
    """

    model_name = data.get("model")
    topk_indices = data.get("topk_indices", [])
    topk_logits = data.get("topk_logits", [])
    final_topk_logits = data.get("final_topk_logits", [])
    threshold = data.get("threshold")
    diff_mask = data.get("diff_mask", [])
    entropy = float(data.get("entropy", 0))

    # If missing essential fields, skip
    if not topk_indices or not topk_logits or not final_topk_logits:
        return None

    token_strings = tokenizer.convert_ids_to_tokens(topk_indices)
    token_strings = [s.replace("$", "\\$").replace("Ġ", '_') for s in token_strings]

    fig, ax = plt.subplots(figsize=(18,6))
    x_vals = np.arange(len(topk_indices))

    ax.plot(
        x_vals,
        topk_logits,
        color="red",
        marker="o",
        label="Original top-k logits",
        zorder = 1
    )

    highlight_x = []
    highlight_y = []
    for i, mask_val in enumerate(diff_mask):
        if mask_val:
            highlight_x.append(i)
            highlight_y.append(topk_logits[i])

    ax.scatter(
        highlight_x,
        highlight_y,
        color="black",
        marker="D",
        s=50,
        label=f"Drop > {threshold}",
        zorder = 2
    )

    # Add cut off for minp comparison
    minp = 0.25
    cutoff_logit = topk_logits[0] + math.log(minp) # Best minp performance is with higher p for reasoning
    ax.axhline(
        y=cutoff_logit,
        color="gray",
        linestyle="--",
        label=f"Min-p w/ p={minp} cutoff = {cutoff_logit:.2f}"
    )

    stats_text = (
        f"Entropy: {entropy:.4f}\n"
    )
    ax.text(
        0.96, 0.96,
        stats_text,
        transform=ax.transAxes,
        ha="right", va="center",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="square")
    )

    ax.set_xticks(x_vals)
    ax.set_xticklabels(token_strings, rotation=60, ha="right", fontsize=10)
    ax.set_xlabel("Top-k tokens (descending logit order)", fontsize=12)
    ax.set_ylabel("Logit value", fontsize=12)
    ax.set_xlim(-0.5, len(topk_logits) - 0.5)

    prompt_ids = data.get("past_token_ids", [])
    if "gemma" in model_name.lower():
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False).replace("$", "\\$")
        user_marker = "<start_of_turn>user"
        assistant_marker = "<start_of_turn>model"
        user_idx = prompt.rfind(user_marker) + len(user_marker)
        assistant_idx = prompt.rfind(assistant_marker)
        user_prompt = prompt[user_idx:assistant_idx].strip()[:-13] # Remove "<end_of_turn>"
        model_response = prompt[assistant_idx + len(assistant_marker):].strip()

    elif "llama" in model_name.lower():
        prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False).replace("$", "\\$")
        user_marker = "<|start_header_id|>user<|end_header_id|>"
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
        user_idx = prompt.rfind(user_marker) + len(user_marker)
        assistant_idx = prompt.rfind(assistant_marker)
        user_prompt = prompt[user_idx:assistant_idx].strip()[:-10] # Remove "<|eot_id|>"
        model_response = prompt[assistant_idx + len(assistant_marker):].strip()

    user_prompt_wrapped = fill(f"User: {user_prompt}", width=200)
    model_response_wrapped = fill(f"Model: {model_response}", width=200)

    formatted_prompt = f"{user_prompt_wrapped}\n{model_response_wrapped}"
    ax.set_title(f"Model: {model_name}\n{formatted_prompt}", fontsize=12)

    ax.legend()
    plt.tight_layout()
    return fig

def main():
    subdirs = find_unanalyzed_directories(BASE_DIR)
    if not subdirs:
        print("No unanalyzed directories found in last_drop.")
        return
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_calls", type=int, default=100, help="Max number of calls to plot.")
    max_calls = parser.parse_args().max_calls

    for sd in subdirs:
        print(f"Analyzing {sd} ...")
        analyze_directory(sd, max_calls=max_calls)

if __name__=="__main__":
    main()

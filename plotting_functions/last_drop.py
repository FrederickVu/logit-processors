import matplotlib.pyplot as plt
import numpy as np
import math


def plot(data, metadata, tokenizer):
    """
    Creates a single figure for one call with scatter plots:
      - Red scatter for original topk_logits
      - Black scatter for tokens i where (topk_logits[i] - topk_logits[i+1]) > threshold
    """

    model_name = metadata.get("model_name", "unknown model")
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

    fig, ax = plt.subplots(figsize=(18,8))
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
        s=40,
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
        ha="right", va="top",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5")
    )

    ax.set_xticks(x_vals)
    ax.set_xticklabels(token_strings, rotation=60, ha="right", fontsize=9)
    ax.set_xlabel("Top-k tokens (descending logit order)", fontsize=12)
    ax.set_ylabel("Logit value", fontsize=12)
    ax.set_xlim(-0.5, len(topk_logits) - 0.5)

    ax.set_title(f"Model: {model_name}", fontsize=12)

    ax.legend()
    # plt.tight_layout()
    return fig
import argparse
import sys
import os
from utils.fewshot_registry import FEWSHOT_REGISTRY

def validate_args(args):
    """
    A simple validation function to check if logit processor
    and fewshot arguments are valid before loading model. 
    """
    if args.fewshot:
        if args.fewshot not in FEWSHOT_REGISTRY:
            print(f"Few-shot context key '{args.fewshot}' not found in registry.")
            sys.exit(1)

    if args.logits_processor:
        module_name, class_name = args.logits_processor.split(".")
        file_path = os.path.join('logits_processors', module_name + '.py')
        
        if not os.path.isfile(file_path):
            print(f"[ERROR] No file '{module_name}.py' found in logits_processors/ for processor.")
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if f"class {class_name}" not in content:
            print(f"[ERROR] Found '{module_name}.py' but no 'class {class_name}' in it.")
            sys.exit(1)


def create_cli_parser():    
    """
    Returns an ArgumentParser pre-configured with all the command-line
    options needed for the chat script.
    """
    parser = argparse.ArgumentParser(description="Interactive model chat using text-generation pipeline")

    # ------------------------------------------------
    # Model and device configuration
    # ------------------------------------------------
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Hugging Face model name or path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device type: 'auto', 'cpu', 'cuda', 'mps', etc."
    )

    # -------------------------------------------------
    # Generation hyperparameters
    # -------------------------------------------------
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Maximum number of newly generated tokens in each response. "
             "Recommended to increase (>10k) for reasoning-type models. "
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Defaults to 1.0, but is set to None "
             "if no sampling/processing specified."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy decoding."
             "Defaults to False, but is set True in all expected cases."
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Repetition penalty (>1 means stronger penalty)."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling (top-p) cutoff."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling cutoff."
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=None,
        help="Min-p sampling cutoff. "
             "Note: Temperature scaling is applied after min-p filtering. ")
    
    # -------------------------------------------------
    # Conversation context and few-shot settings
    # -------------------------------------------------
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="An optional initial system prompt or context string."
    )
    parser.add_argument(
        "--fewshot",
        type=str,
        default=None,
        help="A few-shot context name from FEWSHOT_REGISTRY (if any)."
    )
    parser.add_argument(
        "--fewshot_num",
        type=int,
        default=None,
        help="Number of examples to use from the few-shot registry. Defaults to all."
    )

    # -------------------------------------------------
    # Logging and I/O behavior
    # -------------------------------------------------
    parser.add_argument(
        "--log",
        action="store_true",
        help="Whether to log the entire conversation to a file."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="chat_logs",
        help="Directory where conversation logs should be saved. Defaults to `chat_logs`."
    )
    parser.add_argument(
        "--no_history_log",
        action="store_true",
        help="If set, only store top-level events (like turn_completed), not the entire message history."
    )
    parser.add_argument(
        "--no_stream",
        action="store_true",
        help="If set, do not attempt to stream model outputs; generate the entire response at once."
    )

    # -------------------------------------------------
    # Logits processor arguments
    # -------------------------------------------------
    parser.add_argument(
        "--logits_processor",
        type=str,
        default=None,
        help="Fully qualified class name for a custom logits processor, e.g., 'my_module.MyProcessor'."
    )

    parser.add_argument(
        "--analysis",
        action="store_true",
        help="If set, enable analysis/logging mode for custom logits processors."
    )

    # # Similarity processor
    # parser.add_argument(
    #     "--p_trusted",
    #     type=float,
    #     default=0.5,
    #     help="Minimum probability threshold for 'trusted' tokens in a similarity processor."
    # )
    # parser.add_argument(
    #     "--p_min",
    #     type=float,
    #     default=0.1,
    #     help="Minimum (final) probability cutoff for tokens after similarity adjustments."
    # )
    # parser.add_argument(
    #     "--sim_threshold",
    #     type=float,
    #     default=None,
    #     help="Similarity threshold for a similarity-based processor."
    # )
    # parser.add_argument(
    #     "--sim_alpha",
    #     type=float,
    #     default=1.0,
    #     help="Scaling factor for similarity-based filtering."
    # )

    # BetterMinp processor
    parser.add_argument(
        "--p",
        type=float,
        default=0.05,
        help="p value for better min-p processor."
    )

    # LastDrop processor
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Logit difference threshold for LastDropProcessor."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Top-k to consider for LastDrop-like processors."
    )

    # # LogitRatio processor
    # parser.add_argument(
    #     "--ratio_threshold",
    #     type=float,
    #     default=1.01,
    #     help="Ratio threshold for LogitRatioProcessor."
    # )

    return parser
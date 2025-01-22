import os
import json
import argparse
import importlib
from matplotlib.backends.backend_pdf import PdfPages
from transformers import AutoTokenizer
from textwrap import fill
from utils.marker_registry import get_markers
import matplotlib.pyplot as plt

# Set default for logging dir, data log file name, ane analyis output pdf name
ANALYSIS_BASE_DIR = "analyses"
LOG_FILE_NAME = "logs.jsonl"
ANALYSIS_PDF = "analysis.pdf"
PLOT_FN_DIR = "plotting_functions"

# Avoid character display issues
plt.rcParams['font.family'] = ['sans-serif', 'Droid Sans Fallback', 'DejaVu Sans', 'Noto Mono']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processor_module",
        type=str,
        default="base",
        help="Name of file containing custom logits processor class which generated "
             "the data on which analysis is to be performed. E.g., if chat was "
             "run with 'my_processor.MyProcessorClass', then provide 'my_processor'."
    )
    parser.add_argument(
        "--plot_fn",
        type=str,
        required=True,
        help="Name of plotting function <module_name>.<fn_name> for dynamic import."
    )
    parser.add_argument(
        "--max_calls",
        type=int,
        default=150,
        help="Max number of records to process from each JSONL"
    )
    args = parser.parse_args()

    plotting_fn = load_plot_fn(args.plot_fn)
    if plotting_fn is None:
        print(f"No function found at {args.plot_fn}.")
        return
    
    unanalyzed_subdirs = find_unanalyzed_directories(args.processor_module)
    if not unanalyzed_subdirs:
        print("No directories found to analyze in "
              f"{ANALYSIS_BASE_DIR}/{args.processor_module}.")
        return

    for subdir in unanalyzed_subdirs:
        print(f"Analyzing {subdir} ...")
        analyze_subdirectory(
            subdir,
            plotting_fn,
            max_calls=args.max_calls
        )

    return

def load_plot_fn(qualified_plot_fn_name: str):
    """
    Dynamically load function from 'my_plot_utils.plotting_fn'
    """
    mod_name, fn_name = qualified_plot_fn_name.rsplit(".", 1)
    fn = None
    try:
        mod = importlib.import_module(f"{PLOT_FN_DIR}.{mod_name}")
        fn = getattr(mod, fn_name)
    except Exception as e:
        print(f"[DEBUG] Error loading module '{mod_name}': {e}")
        return None
    return fn


def find_unanalyzed_directories(processor_name):
    """
    Looks inside <ANALYSIS_BASE_DIR>/<processor_name>/ for subdirectories which
    do NOT contain the file 'analysis.pdf'.
    """
    top_dir = os.path.join(ANALYSIS_BASE_DIR, processor_name)

    if not os.path.isdir(top_dir):
        print(f"No logs found under '{top_dir}'.")
        return []
    
    unanalyzed_subdirs = [
        subdir
        for f in os.listdir(top_dir)
        if os.path.isdir(subdir := os.path.join(top_dir, f))
            and not os.path.exists(os.path.join(subdir, ANALYSIS_PDF))
    ]
    return unanalyzed_subdirs


def load_jsonl(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
    

def analyze_subdirectory(subdir, plotting_fn, max_calls):
    """
    - Reads up to 'max_calls' records from <subdir>/<LOG_FILE_NAME>
    - Calls plotting_fn(record) for each record
    - Saves all resulting figures to <subdir>/<ANALYSIS_PDF>
    """
    jsonl_path = os.path.join(subdir, LOG_FILE_NAME)
    pdf_path = os.path.join(subdir, ANALYSIS_PDF)

    if not os.path.exists(jsonl_path):
        print(f"No '{LOG_FILE_NAME}' in {subdir}, skipping.")
        return

    data = load_jsonl(jsonl_path)[:max_calls]
    if not data:
        print(f"No data in {jsonl_path}, skipping.")
        return
    
    # Get model_name from metadata to load tokenizer
    metadata_path = os.path.join(subdir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_name = metadata.get("model_name")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Generate plot for each piece of data
    pdf = None
    try:
        pdf = PdfPages(pdf_path)
        for record in data:
            fig = plotting_fn(record, metadata, tokenizer)
            if fig:
                # Set User message and Assistant response in title of plot
                user_text, asst_text = extract_chat_text(record, metadata, tokenizer)
                wrapped_user = fill(f"User: {user_text}", width=200)
                wrapped_asst = fill(f"Assistant: {asst_text}", width=200)
                if fig.axes and user_text and asst_text:
                    ax = fig.axes[0]
                    old_title = ax.get_title()
                    
                    if old_title:
                        new_title = f"{old_title}\n\n{wrapped_user}\n{wrapped_asst}"
                    else:
                        new_title = f"User: {wrapped_user}\nAsst: {wrapped_asst}"
                    ax.set_title(new_title)

                pdf.savefig(fig)
                plt.close(fig)
    except Exception as e:
        print(f"Error analyzing {subdir}: {e}")
    finally:
        if pdf:
            pdf.close()

def extract_chat_text(record, metadata, tokenizer):
    """
    Use tokenizer of model associated with 'subdir' to identify user
    and assistant text recorded in past_token_ids in 'record'. 
    """
    token_ids = record.get("past_token_ids", [])
    model_name = metadata.get("model_name")

    decoded_ctx = tokenizer.decode(token_ids, skip_special_tokens=False)
    user_marker, asst_marker = get_markers(model_name)
    user_idx = decoded_ctx.rfind(user_marker)
    asst_idx = decoded_ctx.rfind(asst_marker)

    if user_idx < 0 or asst_idx < 0:
        return ("","")

    user_text = decoded_ctx[user_idx + len(user_marker):asst_idx]
    asst_text = decoded_ctx[asst_idx + len(asst_marker):]

    return (user_text, asst_text)

if __name__ == "__main__":
    main()
# Chat with Custom Logits Processors

This repository provides an interactive chat script (`chat.py`) for experimenting with:
- **Language Models** from Hugging Face,
- **Few-shot prompting** (from a simple registry in `fewshot_registry.py`),
- **Custom logit processors** that transform logits before sampling (e.g., `LastDropProcessor`),
- **(Optional) analysis** of logit processor outputs, stored in `analysis/`.

## Features

1. **Interactive multi-turn chat**:  
   - Default model is **`meta-llama/Llama-3.2-3B-Instruct`** if you don’t specify `--model`.  

2. **Few-shot prompts**:  
   - Provide a name like `hendrycks` and an integer to specify how many examples (`--fewshot_num`) from `fewshot_registry.py` to use.
   - The script prepends those user→assistant examples before your real conversation, for in-context learning.

3. **Custom logit processors**:  
   - Supply `--logits_processor <path.to.YourProcessor>` to transform logits at generation time.
   - Set parameters via command-line (e.g. `--threshold 0.9 --k 30` for `LastDropProcessor`).

4. **Analysis logs**:  
   - If run with `--analysis`, certain processors (like `LastDropProcessor`) write data to `analysis/<processor_name>/*`.
   - You can then run `last_drop_analysis.py` to produce PDF plots.

5. **Chat logs**:  
   - By default, **no** conversation logs are saved. Use `--log` to store them in `chat_logs/`.

## Usage

By default, `chat.py`:

- Uses the **`meta-llama/Llama-3.2-3B-Instruct`** model if you do not specify `--model`.
- Performs **greedy sampling** unless you enable `--do_sample` or specify parameters like `--top_p` or `--top_k` or a logits processor.
- Does **not** store conversation logs (`--log` must be set if you want them).

Below are some sample invocations:
1. **Basic chat, default model, greedy sampling**  
   ```bash
   python3 chat.py
   ```
2. **Use another model**
  ```bash
  python3 chat.py --model google/gemma-2-9b-it
  ```
3. **Fewshot prompting**
  ```bash
  python3 chat.py --fewshot hendrycks --fewshot_num 2
  ```
- Loads two user and assistant examples from the “hendrycks” set in fewshot_registry.py
- Omitting `--fewshot_num` defaults to including all fewshot examples
4. **Custom logit processor and analysis**
  ```bash
  python3 chat.py \
    --model google/gemma-2-9b-it \
    --logits_processor last_drop.LastDropProcessor \
    --threshold 0.9 \
    --k 30 \
    --fewshot hendrycks \
    --analysis
  ```
  - Writes additional data under `analysis/last_drop/` for later analysis.
  - Run `python3 last_drop_analysis.py` to perform analysis. 
5. **Chat logging**
    ```bash
    python3 chat.py --log
    ```
    - Appends each completed turn to a JSONL file in `chat_logs/`.

## Conversation Commands

Inside the chat loop, you can type:

- **`regenerate`** or **`regenerate N`**: Re-run the last user turn (or turn N, zero-indexed) and produce a new assistant response.
- **`edit N`**: Modify the user’s text at turn N, then regenerate subsequent turns accordingly.
- **`switch_model <NAME>`**: Switch to a different Hugging Face model mid-session.
- **`show_params <ClassName>`**: Display the current parameters for a given custom logit processor class.
- **`set_params param=value ...`**: Adjust logit processor parameters on the fly.
- **`clear analysis`**: If `analysis_mode` is active, clear the JSON logs for the currently loaded processor.
- **`exit`** / **`quit`**: End the session.




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
   - Supply `--logits_processor <module_name>.<ProcessorClass>` to process logits at generation time.
   - Set parameters via command-line (e.g. `--threshold 0.9 --k 30` for `LastDropProcessor`).

4. **Analysis logs**:  
   - If run with `--analysis`, processors (with a defined `_write_log` function) write data to `analyses/<processor_name>/*`.
   - You can then run `analysis_script.py` to produce PDF plots.

5. **Chat logs**:  
   - Use `--log` to store chat history in `chat_logs/`. Operates independently of the analysis logs. 

## Usage

By default, `chat.py`:

- Uses the **`meta-llama/Llama-3.2-3B-Instruct`** model if you do not specify `--model`.
- Performs **greedy sampling** unless you enable `--do_sample` or specify parameters like `--top_p` or `--top_k` or a `--logits_processor`.
- Does **not** store conversation logs unless you enable `--log`.
- Does **not** store logits processor generated data unless `_write_log` is defined in the processor's class and enable `--analysis`. 

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
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --logits_processor last_drop.LastDropProcessor \
    --threshold 0.9 \
    --k 30 \
    --fewshot hendrycks \
    --analysis
  ```
  - Writes additional data under `analyses/last_drop/DeepSeek-R1-Distill-Qwen-1.5B_<timestamp>` for later analysis.
  - Run `python3 analysis_script.py --processor_module last_drop --plot_fn last_drop.plot` to perform analysis. 
5. **Chat logging**
  ```bash
  python3 chat.py --log
  ```
  - Appends each completed turn to a JSONL file in `chat_logs/`.

## Conversation Commands

Inside the chat loop, you can type:

- **`regenerate`** or **`regenerate N`**: Re-run the last user turn (or turn N, zero-indexed) and produce a new assistant response.
- **`edit N`**: Modify the user’s text at turn N, then regenerate subsequent turns accordingly.
- **`show_params`**: Display the current parameters for a given custom logit processor class.
- **`set_params param=value ...`**: Adjust logit processor parameters on the fly.
- **`FEWSHOT ...`**: Format your prompt according to a specified fewshot template, if available.
- **`clear analysis`**: If `analysis_mode` is active, clear the JSONL logs for the currently loaded processor.
- **`exit`** / **`quit`**: End the session.

## Writing a Custom Logits Processor

1. Create a new Python file at `logits_processors/<my_processor>.py`.
2. Extend the `BaseCustomProcessor` class defined in `logits_processors/base_processor.py`, following the template provided at `logits_processors/logit_processor_template.py`.
3. Override the `__init__`, `__call__`, and `_write_log` (optional)
4. Add new parameters to the CLI parser in `utils/parser.py`.
You can now specify your processor and parameters at the command line, e.g.:
  ```bash
  python chat.py \
    --logits_processor my_processor.MyProcessor \
    --my_param 2.0 \
    --my_param2 3.03 \
    --analysis
  ```
and may additionally view and set new parameter values via `show_params` and `set_params` at runtime. 

## Plotting data for analysis
1. Define `_write_log` in your custom processor class, following the template `logit_processor_template.py` in `logits_processors/`.
2. Define a custom plotting function in `plotting_functions/<plotter_for_my_processor>.py`
3. Run `chat.py` with the `--analysis` flag to log data in `analyses/<processor_name>/<model>_<timestamp>/`
4. Run 
  ```bash
  python analysis_script.py --processor_module <processor_name> --plot_fn <plotter_file>.<my_fn> --max_calls 12345679
  ```
   to generate `analyses/<processor_name>/<model>_<timestamp>/analysis.pdf` for each subdirectory of `analyses/<processor_name>/` which lacks an `analysis.pdf` file.
   Each `analysis.pdf` will contain one plot per call, up to `max_calls`, so it is best to test your plotting function with this set to a low number. 

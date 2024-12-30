import argparse
import json
import os
import sys
import importlib
import torch
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig,
    TextStreamer,
    LogitsProcessorList,
    MinPLogitsWarper,
    TemperatureLogitsWarper,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def ensure_hf_token():
    if "HF_TOKEN" not in os.environ or not os.environ["HF_TOKEN"].strip():
        print("HF_TOKEN environment variable not found.")
        token = input("Please enter your Hugging Face Hub token (or leave blank to exit): ").strip()
        if not token:
            print("No token provided. Exiting.")
            sys.exit(1)
        os.environ["HF_TOKEN"] = token

def prompt_for_strategy():
    print("No sampling procedure or custom logits processor provided.")
    resp = input("Do you want to specify one now? (y/n) ").strip().lower()
    if resp.startswith('n'):
        print("No strategy chosen. Exiting.")
        sys.exit(1)
    print("Enter a sampling parameter or custom processor.")
    print("Examples: 'top_p=0.9', 'top_k=50', 'min_p=0.05', or 'logits_processor=module.ClassName'")
    choice = input("Your choice: ").strip()
    if not choice:
        print("No choice made. Exiting.")
        sys.exit(1)
    return choice

def parse_user_choice(choice):
    if '=' not in choice:
        print("Invalid format. Exiting.")
        sys.exit(1)
    key, val = choice.split('=', 1)
    key = key.strip()
    val = val.strip()
    if key == "top_p":
        return {"top_p": float(val), "do_sample": True}
    elif key == "top_k":
        return {"top_k": int(val), "do_sample": True}
    elif key == "min_p":
        return {"min_p": float(val), "do_sample": True}
    elif key == "logits_processor":
        return {"logits_processor": val, "do_sample": True}
    else:
        print("Unknown parameter. Exiting.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Interactive model chat using text-generation pipeline")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it", help="Hugging Face model name")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', 'cuda', 'mps'")
    parser.add_argument("--log_dir", type=str, default="chat_logs", help="Directory to store conversation logs (default: chat_logs).")
    parser.add_argument("--no_log", action="store_true", help="Do not log the conversation.")
    parser.add_argument("--system_prompt", type=str, default=None, help="Initial context.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens for responses.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty.")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling top_p.")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--min_p", type=float, default=None, help="Min-p sampling.") # Note: if temp!=1.0, need to do min_p before temp scaling
    parser.add_argument("--no_history_log", action="store_true", help="If set, only store event in logs.")
    parser.add_argument("--no_stream", action="store_true", help="If set, do not attempt to stream output. By default we try streaming.")

    parser.add_argument("--logits_processor", type=str, default=None, help="Custom logits processor class in format module.ClassName")
    parser.add_argument("--p_trusted", type=float, default=.5, help="Min p cutoff for trusted tokens for similarity processor.")
    parser.add_argument("--sim_threshold", type=float, default=None, help="Similarity threshold for SimilarityProcessor.")
    parser.add_argument("--sim_alpha", type=float, default=1.0, help="Similarity scaling hyperparameter for SimilarityProcessor.")

    parser.add_argument("--ratio_threshold", type=float, default=1.01, help="Logit ratio threshold for LogitRatioProcessor.")
    args = parser.parse_args()

    ensure_hf_token()

    model_name = args.model
    device = args.device
    log_dir = args.log_dir
    no_log = args.no_log
    system_prompt = args.system_prompt
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    do_sample = args.do_sample
    repetition_penalty = args.repetition_penalty
    no_history_log = args.no_history_log
    no_stream = args.no_stream
    top_p = args.top_p
    top_k = args.top_k
    min_p = args.min_p
    logits_processor_spec = args.logits_processor
    p_trusted = args.p_trusted
    sim_threshold = args.sim_threshold
    sim_alpha = args.sim_alpha

    strategies = [top_p, top_k, min_p, logits_processor_spec]
    provided_strategy = any([strat is not None for strat in strategies])

    if not provided_strategy and not do_sample:
        choice = prompt_for_strategy()
        extras = parse_user_choice(choice)
        if "top_p" in extras:
            top_p = extras["top_p"]
        if "top_k" in extras:
            top_k = extras["top_k"]
        if "min_p" in extras:
            min_p = extras["min_p"]
        if "logits_processor" in extras:
            logits_processor_spec = extras["logits_processor"]
        do_sample = extras.get("do_sample", do_sample)

    if provided_strategy and not do_sample:
        do_sample = True

    if not no_log:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        log_file = open(log_path, "a", encoding="utf-8")
    else:
        log_path = None
        log_file = None

    def log_event(event_data):
        if log_file:
            log_file.write(json.dumps(event_data, ensure_ascii=False) + "\n")
            log_file.flush()

    print(f"Loading model '{model_name}' on device='{device}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

    # Patching side cases in text generation.
    if not no_stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = None

    # Create custom list of LogitsProcessor objects for .generate or pipeline call
    # Note: need to manually order list of processors
    # For some reason, temperature scaling defaults to being first in HF, so change it
    processors = LogitsProcessorList()
    if logits_processor_spec:
        module_name, class_name = logits_processor_spec.rsplit(".", 1)
        sys.path.append(os.path.join(os.getcwd(), "logits_processors"))
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)

        if class_name == "SimilarityProcessor":
            if not sim_threshold:
                print("No sim_threshold for similarity logits processor provided.")
                print("Enter a valid float for sim_threshold.")
                choice = input("sim_threshold = ").strip()
                if not choice:
                    print("No choice made. Exiting.")
                sys.exit(1)
                sim_threshold = float(choice)
            if not sim_alpha:
                print("No sim_threshold for similarity logits processor provided.")
                print("Enter a valid float for sim_threshold.")
                choice = input("sim_threshold = ").strip()
                if not choice:
                    print("No choice made. Exiting.")
                sys.exit(1)
                sim_alpha = float(choice)
                custom_processor = cls(
                    p_trusted=p_trusted,
                    alpha=sim_alpha, 
                    sim_threshold=sim_threshold,
                    embedding=model.lm_head.weight,
                )

        else:
            custom_processor = cls()
        processors.append(custom_processor)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "use_cache": True
    }
    if top_p is not None:
        gen_kwargs["top_p"] = top_p
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
    if min_p is not None:
        if temperature != 1.0:
            # Place temperature warper after min-p filter in processor list
            processors.append(MinPLogitsWarper(min_p=min_p))
            processors.append(TemperatureLogitsWarper(temperature=temperature))
            gen_kwargs['temperatre']=1.0
        else:
            gen_kwargs["min_p"] = min_p
    if tokenizer.pad_token_id is None:
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    gen_config = GenerationConfig(**gen_kwargs)

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device,
        return_full_text=False,
        streamer=streamer,
        generation_config=gen_config
    )

    print("Model loaded successfully!\n")
    print("Commands:\n")
    print(" - Just type a message and press Enter for a response.")
    print(" - 'regenerate' to regenerate the last assistant response.")
    print(" - 'regenerate N' to regenerate assistant response for turn N.")
    print(" - 'edit N' to modify user input at turn N and regenerate subsequent turns.")
    print(" - 'revert N' to revert conversation to turn N (remove subsequent turns).")
    print(" - 'history' to print the current conversation.")
    print(" - 'versions N' to show all versions of turn N from logs.")
    print(" - 'switch_model <model_name>' to switch models mid-session.")
    print(" - 'set_params param1=val1 param2=val2' to update params on logit processors.")
    print(" - 'show_params <ProcessorClassName>' to show current params on logit processor.")
    print(" - 'exit' or 'quit' to end the session.\n")

    logged_params = {}
    for k, v in gen_kwargs.items():
        if k == "do_sample" and v is False:
            continue
        if k == "use_cache" and v is True:
            continue
        if k == "max_new_tokens" and v == 256:
            continue
        if k == "temperature" and v == 1.0:
            continue
        if k == "repetition_penalty" and v == 1.0:
            continue
        logged_params[k] = v

    event_data = {
        "event": "session_start",
        "model_name": model_name,
        "system_prompt_included_as_user": bool(system_prompt)
    }
    if logged_params:
        event_data["params"] = logged_params
    if logits_processor_spec:
        event_data["logits_processor"] = logits_processor_spec

    log_event(event_data)

    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "user", "content": system_prompt.strip()})

    current_version_id = 0

    def get_turns():
        turns = []
        i = 0
        turn_id = 0
        while i < len(messages):
            if messages[i]["role"] == "user":
                u = messages[i]
                a = None
                if i+1 < len(messages) and messages[i+1]["role"] == "assistant":
                    a = messages[i+1]
                    i += 2
                else:
                    i += 1
                turns.append((turn_id, u, a))
                turn_id += 1
            else:
                i += 1
        return turns

    def print_history():
        turns = get_turns()
        for t_id, user_msg, assistant_msg in turns:
            print(f"Turn {t_id}:")
            print(f"  User: {user_msg['content']}")
            if assistant_msg:
                print(f"  Assistant: {assistant_msg['content']}")
            else:
                print("  Assistant: [No response yet]")
            print("")

    def generate_assistant(msg_list):
        outputs = text_gen_pipeline(text_inputs=msg_list, logits_processor=processors if processors else None)
        assistant_content = outputs[0]["generated_text"]
        return {"role": "assistant", "content": assistant_content}

    def regenerate_turn(turn_id):
        nonlocal current_version_id
        all_turns = get_turns()
        if turn_id < 0 or turn_id >= len(all_turns):
            print("Invalid turn number.")
            return

        base_messages = []
        for t, u, a in all_turns[:turn_id]:
            base_messages.append(u)
            base_messages.append(a)
        user_msg = all_turns[turn_id][1]
        base_messages.append(user_msg)

        current_version_id += 1
        new_version_id = current_version_id
        assistant_msg = generate_assistant(base_messages)

        messages.clear()
        messages.extend(base_messages)
        messages.append(assistant_msg)

        log_event({
            "event": "regenerate",
            "turn_id": turn_id,
            "version_id": new_version_id,
            "user": user_msg["content"],
            "assistant": assistant_msg["content"],
            # "messages": None if no_history_log else messages
        })

        # If not streaming, print the model output now
        if no_stream:
            print("Model:", assistant_msg["content"], "\n")

    def edit_turn(turn_id):
        nonlocal current_version_id
        all_turns = get_turns()
        if turn_id < 0 or turn_id >= len(all_turns):
            print("Invalid turn number.")
            return
        old_user_msg = all_turns[turn_id][1]
        new_user_input = input(f"Enter new user message for turn {turn_id}: ").strip()

        base_messages = []
        for t, u, a in all_turns[:turn_id]:
            base_messages.append(u)
            base_messages.append(a)
        edited_user_msg = {"role": "user", "content": new_user_input}
        base_messages.append(edited_user_msg)

        current_version_id += 1
        new_version_id = current_version_id
        assistant_msg = generate_assistant(base_messages)

        messages.clear()
        messages.extend(base_messages)
        messages.append(assistant_msg)

        log_event({
            "event": "edit",
            "turn_id": turn_id,
            "version_id": new_version_id,
            "old_user": old_user_msg["content"],
            "new_user": new_user_input,
            "assistant": assistant_msg["content"],
            # "messages": None if no_history_log else messages
        })

        print("Model:", assistant_msg["content"], "\n")

    def revert_to(turn_id):
        all_turns = get_turns()
        if turn_id < 0 or turn_id >= len(all_turns):
            print("Invalid turn number.")
            return

        base_messages = []
        for t, u, a in all_turns[:turn_id+1]:
            base_messages.append(u)
            if a:
                base_messages.append(a)

        messages.clear()
        messages.extend(base_messages)

        log_event({
            "event": "revert",
            "to_turn_id": turn_id,
            "messages": None if no_history_log else messages
        })

        print(f"Conversation reverted to turn {turn_id}.")

    def show_versions(turn_id):
        if no_log or not log_file:
            print("No log file available for version tracking.")
            return
        log_file.flush()
        if not os.path.exists(log_path):
            print("No log file found.")
            return

        versions = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                if data.get("turn_id") == turn_id and ("assistant" in data or data.get("event") in ["edit","regenerate","turn_completed"]):
                    versions.append(data)

        if not versions:
            print(f"No versions found for turn {turn_id}.")
            return

        print(f"Versions for turn {turn_id}:")
        for v in versions:
            ev = v.get("event", "turn_completed")
            ver_id = v.get("version_id", "?")
            usr = v.get("user") or v.get("new_user")
            assistant = v.get("assistant")
            print(f" - version_id={ver_id}, event={ev}, user={usr}, assistant={assistant}")

    def show_processor_params(processor_class_name: str):
        """
        Looks through all processors in `processors` to find a match
        for `processor_class_name`, calls `get_param_names()` on it,
        and prints them out. If no match, prints error.
        """
        found = False
        for proc in processors:
            if proc.__class__.__name__ == processor_class_name:
                if hasattr(proc, "get_param_names") and callable(proc.get_param_names):
                    param_info = proc.get_param_names()
                    if isinstance(param_info, dict):
                        # If you store param name -> current value in a dict
                        print(f"Parameters for {processor_class_name}:")
                        for pname, val in param_info.items():
                            print(f"  {pname} = {val}")
                    elif isinstance(param_info, (list, set)):
                        print(f"Parameter names for {processor_class_name}: {param_info}")
                    else:
                        print(f"{processor_class_name} returned unknown type from get_param_names: {param_info}")
                else:
                    print(f"{processor_class_name} does not implement a get_param_names() method.")
                found = True
                break
        if not found:
            print(f"No processor named '{processor_class_name}' is currently in use.")

    def switch_model(new_model_name):
        nonlocal model, tokenizer, text_gen_pipeline, model_name
        print(f"Switching model to '{new_model_name}'...")
        model_name = new_model_name
        new_tokenizer = AutoTokenizer.from_pretrained(model_name)
        new_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        new_text_gen_pipeline = pipeline(
            "text-generation",
            model=new_model,
            tokenizer=new_tokenizer,
            device_map=device,
            return_full_text=False,
            generation_config=gen_config
        )
        tokenizer = new_tokenizer
        model = new_model
        text_gen_pipeline.__dict__.update(new_text_gen_pipeline.__dict__)
        print("Model switched successfully.\n")
        log_event({
            "event": "switch_model",
            "model_name": model_name
        })

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the conversation. Goodbye!")
            break
        
        if user_input.lower().startswith("show_params "):
            # e.g. "show_params LogitRatioProcessor"
            parts = user_input.split()
            if len(parts) == 2:
                target_class = parts[1].strip()
                show_processor_params(target_class)
            else:
                print("Usage: show_params <ProcessorClassName>")
            continue

        if user_input.lower().startswith("set_params "):
            param_str = user_input[len("set_params "):].strip()
            if not param_str:
                print("Usage: set_params param1=val1 param2=val2 ...")
                continue
            chunks = param_str.split()
            for chunk in chunks:
                if "=" in chunk:
                    pkey, pval = chunk.split("=",1)
                    pkey=pkey.strip(); pval=pval.strip()
                    # Try to set parameter on custom processor(s)
                    changed=False
                    for proc in processors:
                        if hasattr(proc,"set_param"):
                            try:
                                proc.set_param(pkey,pval)
                                log_event({"event":"set_params","processor":proc.__class__.__name__,"param":pkey,"value":pval})
                                print(f"Set {pkey}={pval} on {proc.__class__.__name__}")
                                changed=True
                            except ValueError:
                                pass
                    if not changed:
                        print(f"No processor recognized param '{pkey}'.")
                else:
                    print("Usage: set_params param1=val1 param2=val2 ... (space separated)")
            continue

        if user_input.lower().startswith("regenerate"):
            parts = user_input.split()
            if len(parts) == 1:
                all_turns = get_turns()
                last_turn_id = len(all_turns) - 1
                if last_turn_id >= 0:
                    regenerate_turn(last_turn_id)
                else:
                    print("No turns to regenerate.")
            else:
                try:
                    turn_id = int(parts[1])
                    regenerate_turn(turn_id)
                except ValueError:
                    print("Usage: regenerate [N]")
            continue

        if user_input.lower().startswith("edit "):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    turn_id = int(parts[1])
                    edit_turn(turn_id)
                except ValueError:
                    print("Usage: edit N")
            else:
                print("Usage: edit N")
            continue

        if user_input.lower().startswith("revert "):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    turn_id = int(parts[1])
                    revert_to(turn_id)
                except ValueError:
                    print("Usage: revert N")
            else:
                print("Usage: revert N")
            continue

        if user_input.lower() == "history":
            print_history()
            continue

        if user_input.lower().startswith("versions "):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    turn_id = int(parts[1])
                    show_versions(turn_id)
                except ValueError:
                    print("Usage: versions N")
            else:
                print("Usage: versions N")
            continue

        if user_input.lower().startswith("switch_model "):
            parts = user_input.split(" ", 1)
            if len(parts) == 2:
                new_model = parts[1].strip()
                switch_model(new_model)
            else:
                print("Usage: switch_model <model_name>")
            continue

        current_version_id += 1
        new_version_id = current_version_id

        messages.append({"role": "user", "content": user_input})
        assistant_msg = generate_assistant(messages)
        messages.append(assistant_msg)

        if no_stream:
            print("Model:", assistant_msg["content"], "\n")

        last_turn_id = len(get_turns()) - 1
        log_event({
            "event": "turn_completed",
            "turn_id": last_turn_id,
            "version_id": new_version_id,
            "user": user_input,
            "assistant": assistant_msg["content"],
            # "messages": None if no_history_log else messages
        })

    if log_file:
        log_file.close()

if __name__ == "__main__":
    main()

import sys
import os
import importlib

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TextStreamer, 
    GenerationConfig,
    pipeline,
)
from transformers import LogitsProcessorList, MinPLogitsWarper

PROCESSOR_DIR = "logits_processors"

def ensure_hf_token():
    """
    Check if the HF_TOKEN environment variable is set and non-empty.
    If not, prompt the user and exit.
    """
    if "HF_TOKEN" not in os.environ or not os.environ["HF_TOKEN"].strip():
        print("HF_TOKEN environment variable not found. Run")
        print("  export HF_TOKEN=<your token>")
        print("and rerun the script.")
        sys.exit(1)

def load_chat_pipeline(args):
    """
    Loads a model, tokenizer, and text-generation pipeline using the information
    from 'args'.

    Returns:
        text_gen_pipeline: A transformers pipeline for text generation
    """
    logits_processor_spec = args.logits_processor
    top_p = args.top_p
    top_k = args.top_k
    min_p = args.min_p
    do_sample = args.do_sample
    strategies = [top_p, top_k, min_p, logits_processor_spec]
    provided_strategy = any([strat is not None for strat in strategies])

    if provided_strategy and not do_sample:
        do_sample = True
    
    model_name = args.model
    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    no_stream = args.no_stream
    streamer = (TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                if not no_stream else None)
    
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": do_sample,
        "temperature": args.temperature if do_sample else None,
        "repetition_penalty": args.repetition_penalty,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    gen_config = GenerationConfig(**gen_kwargs)

    # Optional: set up a TextStreamer for streaming text outputs,
    # if we want interactive streaming. If the user passed --no_stream,
    # we set streamer to None.
    streamer = None
    if not args.no_stream:
        streamer = TextStreamer(
            tokenizer=tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

    # Finally, create the text-generation pipeline
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device,
        return_full_text=False,
        generation_config=gen_config,
        streamer=streamer
    )

    print("Model and pipeline loaded successfully.\n")
    return text_gen_pipeline

def build_logits_processor_list(args):
    """
    Creates and returns a LogitsProcessorList based on the arguments
    in 'args'. 
    """
    processors = LogitsProcessorList()

    # If a custom processor was specified, import and instantiate it
    if args.logits_processor:
        module_name, class_name = args.logits_processor.rsplit(".", 1)
        mod = importlib.import_module(f"{PROCESSOR_DIR}.{module_name}")
        processor_class = getattr(mod, class_name)

        custom_processor = processor_class(
            analysis_mode=args.analysis,
            model_name=args.model,
            **vars(args)
        )

        processors.append(custom_processor)

    # If min_p is specified, append a built-in HF MinPLogitsWarper
    if args.min_p is not None:
        processors.append(MinPLogitsWarper(min_p=args.min_p))

    return processors

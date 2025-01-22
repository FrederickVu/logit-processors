import os
import json
from transformers import AutoTokenizer

MARKER_REGISTRY_NAME = "marker_registry.json"
registry_path = os.path.join(os.path.dirname(__file__), MARKER_REGISTRY_NAME)


def load_marker_registry():
    with open(registry_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_marker_registry(registry: dict):
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

def get_markers(model_name):
    registry = load_marker_registry()
    if model_name in registry:
        model_markers = registry[model_name]
        return model_markers["user_marker"], model_markers["assistant_marker"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    toy_messages = [
        {"role": "user", "content": "MEOW"},
        {"role": "assistant", "content": "WOOF"},
        {"role": "user", "content": "RIBBIT"},
    ]
    templated_str = tokenizer.apply_chat_template(toy_messages, tokenize=False)
    first_user_idx = templated_str.find("MEOW")
    asst_idx = templated_str.find("WOOF")
    second_user_idx = templated_str.find("RIBBIT")
    
    # user_marker = templated_str[asst_idx + len("WOOF"):second_user_idx].strip()

    user_marker_cand = templated_str[asst_idx + len("WOOF"):second_user_idx].strip()
    chat_prefix = templated_str[:first_user_idx].strip()
    user_marker = os.path.commonprefix([user_marker_cand[::-1],chat_prefix[::-1]])[::-1]

    asst_marker = templated_str[first_user_idx + len("MEOW"):asst_idx].strip()

    if user_marker not in templated_str[:first_user_idx]:
        print("Attempting to register markers for a new model.")
        print("Please check that the user_marker and asst_marker "
            "always represent the start of a user or model message.")
        print("Chat templated toy chat:", repr(templated_str))

    registry[model_name] = {
        "user_marker": user_marker,
        "assistant_marker": asst_marker
    }
    save_marker_registry(registry)

    return user_marker, asst_marker

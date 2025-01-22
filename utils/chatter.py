import json
import os
from datetime import datetime
from utils.fewshot_registry import FEWSHOT_FORMATTING, create_base_context

class Chat:
    def __init__(
        self, *,
        logging_enabled=False,
        system_prompt=None,
        fewshot=None,
        fewshot_num=None,
        logits_processor_spec: str = None,
        pipeline=None,
        processors=None,
        model_name=None
    ):
        self.pipeline = pipeline
        self.processors=processors

        self.logging_enabled = logging_enabled
        self.log_file = None
        self.logits_processor_spec = logits_processor_spec
        self.model_name = model_name

        if fewshot:
            self.fewshot_formatter = FEWSHOT_FORMATTING[fewshot]
        else:
            self.fewshot_formatter = lambda x: x

        self.base_context = create_base_context(system_prompt, fewshot, fewshot_num)

        # self.messages is a list of dicts, each with keys "role" and "content"
        self.messages = []

        if self.logging_enabled:
            self._setup_logging()

    # ------------------------------------------------
    # Logging helpers
    # ------------------------------------------------
    def _setup_logging(self):
        # Construct logging file path from specified processor and timestamp
        if self.logits_processor_spec:
            module_name, _ = self.logits_processor_spec.rsplit(".", 1)
        else:
            module_name = "no_processor"

        model_name_clean = (self.model_name or "unknown_model").split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        chat_subdir = os.path.join(
                "chat_logs",
                module_name,
                f"{model_name_clean}_{timestamp}"
            )
        os.makedirs(chat_subdir, exist_ok=True)

        chat_log_path = os.path.join(chat_subdir, "conversation.jsonl")
        self.log_file = open(chat_log_path, "w", encoding="utf-8")

    def _log_event(self, event_data):
        if self.log_file:
            self.log_file.write(json.dumps(event_data, ensure_ascii=False) + "\n")
            self.log_file.flush()
        
    def close_log(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    # ------------------------------------------------
    # Core chat functions
    # ------------------------------------------------

    def _add_user_message(self, text):
        self.messages.append({"role": "user", "content": text})

    def _add_asst_message(self, text):
        self.messages.append({"role": "assistant", "content": text})

    def add_turn(self, user_input):
        """
        Add a user message to the chat and generate a response.
        """
        turn_id = len(self.messages) // 2
        self._add_user_message(user_input)
        asst_text = self.pipeline(
            self.base_context + self.messages, 
            logits_processor=self.processors
            )[0]["generated_text"]
        self._add_asst_message(asst_text)

        self._log_event({
            "event": "turn_completed",
            "turn_id": turn_id,
            "user": user_input,
            "assistant": asst_text
        })

    def regenerate(self, turn_id):
        if 2*turn_id >= len(self.messages) or turn_id < 0:
            print("Turn number out of range. Ignoring command.")
            return
        self.messages = self.messages[:2*turn_id + 1]

        user_text = self.messages[-1]["content"]
        asst_text = self.pipeline(
            self.base_context + self.messages, 
            logits_processor=self.processors
            )[0]["generated_text"]
        self._add_asst_message(asst_text)

        self._log_event({
            "event": "regenerate",
            "turn_id": turn_id,
            "user": user_text,
            "assistant": asst_text
        })

    def edit_turn(self, turn_id):
        if turn_id < 0 or 2*turn_id >= len(self.messages):
            print(f"Turn number out of range. Ignoring command.")
            return
        
        new_text = input("Enter new user message: ")
        self.messages[2*turn_id]["content"] = new_text
        self.messages = self.messages[:2*turn_id + 1]

        asst_text = self.pipeline(
            self.base_context + self.messages, 
            logits_processor=self.processors
            )[0]["generated_text"]
        self._add_asst_message(asst_text)

        self._log_event({
            "event": "edit_turn",
            "turn_id": turn_id,
            "new_user": new_text,
            "assistant": asst_text
        })

    def revert_to(self, turn_id):
        self.messages = self.messages[:2*turn_id]

        self._log_event({
            "event": "revert_to",
            "turn_id": turn_id
        })

    def print_history(self):
        for i, message in enumerate(self.messages):
            print(f"  {message['role']}: {message['content']}")

    def set_params(self, key, value):
        if not self.processors:
            print("No processors to set parameters on.")
            return
        processor = self.processors[0]
        processor.set_param(key, value)

    def show_params(self):
        if not self.processors:
            print("No processors to show parameters for.")
            return
        processor = self.processors[0]
        params = processor.get_param_names()
        for key, value in params.items():
            print(f"{key} = {value}")
        
    def clear_analysis(self):
        # Close and reopen the log file in write mode to clear it
        if not self.processors:
            print("No processors to clear analysis for.")
            return
        
        processor = self.processors[0]
        processor.close_log()
        log_file_path = processor.log_file_path
        processor.log_file = open(log_file_path, "w", encoding="utf-8")
        print(f"Cleared analysis logs for {processor.__class__.__name__}")

    def fewshot_format(self, user_input):
        return self.fewshot_formatter(user_input)
    
    def get_num_turns(self):
        return len(self.messages)//2

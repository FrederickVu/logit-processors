import os
import json
import inspect
from datetime import datetime
from transformers import LogitsProcessor

from analysis_script import ANALYSIS_BASE_DIR, LOG_FILE_NAME


class BaseCustomProcessor(LogitsProcessor):
    """
    A base class for custom logits processors that handles:
      - Analysis/logging mode (if analysis_mode=True).
    
    Subclasses should:
      - Include '**kwargs' in their __init__ method signature.
      - Call super().__init__(**kwargs) at the end of their __init__.
      - Set attributes self.{param_name} = param_name for all 
        parameters in their __init__ method signature.
      - Optionally implement _write_log(...) for any analysis-mode writes.
    """

    def __init__(
        self,
        analysis_mode=False,
        model_name=None,
        **kwargs
    ):
        super().__init__()
        self.analysis_mode = analysis_mode
        self.model_name = model_name

        # Internal log file info
        self.log_file = None
        self.log_file_path = None
        self.logging_subdir = None

        # Retrieve named parameters from derived class's __init__ for 
        # get/set and metadata logging
        derived_cls = type(self)
        sig = inspect.signature(derived_cls.__init__)

        self.params = {}
        for name in sig.parameters:
            if name in {"self", "kwargs"}:
                continue
            if hasattr(self, name):
                self.params[name] = getattr(self, name)
            else:
                raise ValueError(
                    f"Parameter '{name}' was in {derived_cls.__name__}.__init__ "
                    f"but self.{name} was not set!"
                )

        if self.analysis_mode:
            self._setup_analysis_directory()

    
    def __call__(self, past_token_ids, logits):
        """
        Subclasses must override this to transform 'logits'.
        Return the modified logits.
        """
        raise NotImplementedError(
            f"__call__ not implemented by {self.__class__.__name__}. "
            "Please override in a subclass."
        )

    def _setup_analysis_directory(self):
        """
        Creates a subdirectory for logging any analysis data in files
        under "analyses/<module_name>/<clean_model_name>_<timestamp>/".
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        module_name = inspect.getmodule(self).__name__.split('.')[-1].lower()
        model_name_clean = (self.model_name or "unknown_model").split("/")[-1]

        # e.g., "analysis/last_drop/modelname_20250101_123456/"
        self.logging_subdir = os.path.join(
            ANALYSIS_BASE_DIR, module_name, f"{model_name_clean}_{timestamp}"
        )
        os.makedirs(self.logging_subdir, exist_ok=True)

        # Write out static metadata to a JSON file
        metadata_path = os.path.join(self.logging_subdir, "metadata.json")
        metadata = {
            "module_name": module_name,
            "model_name": self.model_name,
            "timestamp": timestamp,
            "params": self.params
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Open a log file for per-call JSONL logs
        self.log_file_path = os.path.join(self.logging_subdir, LOG_FILE_NAME)
        self.log_file = open(self.log_file_path, "w", encoding="utf-8")

    def _write_log(self):
        raise NotImplementedError(
            f"_write_log not implemented by {self.__class__.__name__}. "
            "Please override in a subclass and write to self.log_file."
        )

    def close_log(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def set_param(self, key: str, value: str):
        if key not in self.params:
            print(f"Unknown parameter '{key}' for {self.__class__.__name__}")
            return
        
        old_val = self.params[key]

        # Attempt type conversion based on old_val's type
        try:
            if isinstance(old_val, bool):
                self.params[key] = value.lower() == "true"
            elif isinstance(old_val, int):
                self.params[key] = int(value)
            elif isinstance(old_val, float):
                self.params[key] = float(value)
        except:
            print(f"Could not match type of {key} with that of {old_val}. "
                  "Keeping {old_val}.")
            return
        print(f"Updated {key} from {old_val} to {self.params[key]}")
        

    def get_param_names(self):
        return dict(self.params)

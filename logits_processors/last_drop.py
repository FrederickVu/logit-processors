import torch
import os
import json
from datetime import datetime
from transformers import LogitsProcessor
from scipy.stats import entropy

class LastDropProcessor(LogitsProcessor):
    """
    Filter out tokens after the latest drop in logit values above a given threshold. 
    """
    def __init__(self, *, k=100, threshold=0.2, analysis_mode=None, model_name=None):
        self.k = k
        self.threshold = threshold

        self.analysis_mode = analysis_mode
        self.model_name = model_name

        if self.analysis_mode:
            self._setup_analysis_directory()
    
    def __call__(self, past_token_ids, logits):
        logits = logits.squeeze(0)
        topk_logits, topk_indices = torch.topk(logits, self.k, dim=-1)

        logit_diffs = topk_logits[:-1] - topk_logits[1:]
        diff_mask = logit_diffs > self.threshold
        nz = torch.nonzero(diff_mask, as_tuple=True)[0]
        last_drop_index = nz.max().item() if nz.numel() > 0 else -1
        last_drop_logit = topk_logits[last_drop_index]

        logits[logits < last_drop_logit] = float('-inf')

        if self.analysis_mode and self.log_file:
            self._write_log(
                topk_logits=topk_logits,
                topk_indices=topk_indices,
                final_logits=logits,
                last_drop_index=last_drop_index,
                last_drop_logit=last_drop_logit.item(),
                diff_mask=diff_mask,
                past_token_ids=past_token_ids
            )

        return logits.unsqueeze(0)
    
    def _setup_analysis_directory(self):
        """Creates a subdirectory for the logs, named based on model and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_clean = (self.model_name).split("/")[-1]
        self.current_subdir = os.path.join("analysis", "last_drop", f"{model_name_clean}_{timestamp}")
        os.makedirs(self.current_subdir, exist_ok=True)

        metadata_path = os.path.join(self.current_subdir, "metadata.json")
        metadata = {
            "model_name": self.model_name,
            "timestamp": timestamp,
            "k": self.k,
            "threshold": self.threshold,
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        self.log_file_path = os.path.join(self.current_subdir, "last_drop.jsonl")
        self.log_file = open(self.log_file_path, "a", encoding="utf-8")

    def _write_log(
            self,
            topk_logits,
            topk_indices,
            final_logits,
            last_drop_index,
            last_drop_logit,
            diff_mask,
            past_token_ids
        ):

        out_data = {
            "event": "last_drop",
            "model": self.model_name,
            "k": self.k,
            "threshold": self.threshold,
            "last_drop_index": last_drop_index,
            "last_drop_logit": last_drop_logit,
            "topk_logits": topk_logits.tolist(),
            "topk_indices": topk_indices.tolist(),
            "final_topk_logits": final_logits[topk_indices].tolist(),
            "diff_mask": diff_mask.tolist(),
            "entropy": str(entropy(torch.softmax(topk_logits, dim=-1).cpu().numpy())),
            "past_token_ids": past_token_ids.tolist()[0]
        }
        self.log_file.write(json.dumps(out_data) + "\n")
        self.log_file.flush()

    def close_log(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def set_param(self, key, value):
        if key == "k":
            self.k = int(value)
        elif key == "threshold":
            self.threshold = float(value)

    def get_param_names(self):
        return {
            "k": self.k,
            "threshold": self.threshold
        }

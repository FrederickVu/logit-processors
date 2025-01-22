import torch
import json
from logits_processors.base_processor import BaseCustomProcessor
from scipy.stats import entropy

class LastDropProcessor(BaseCustomProcessor):
    """
    Filter out tokens after the latest drop in logit values above a given threshold. 
    """
    def __init__(self, *, k=100, threshold=0.2, **kwargs):
        self.k = k
        self.threshold = threshold

        super().__init__(**kwargs)
    
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

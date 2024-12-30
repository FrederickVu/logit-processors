from transformers import LogitsProcessor
import torch

class LogitRatioProcessor(LogitsProcessor):
    """
    Filters out tokens based on ratio of logit values of tokens, sorted by logits. 
    Precisely, we filter out tokens after the last token whose logit ratio is greater
    than a provided threshold value among the fist k tokens for a provided k. 
    """
    def __init__(self, *, ratio_threshold=1.01, k=100):
        self.ratio_threshold = ratio_threshold
        self.k = k

    def __call__(self, input_ids, scores):
        sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        device = scores.device

        # Create mask for top k tokens
        logit_ratios = sorted_logits[:, :self.k] / sorted_logits[:, 1:self.k+1]
        mask = logit_ratios > self.ratio_threshold
        # Ensure that first token is always kept
        mask[:,0] = True

        # Find last occurrence of True in mask
        indices = torch.arange(self.k, device=device).unsqueeze(0).expand_as(mask)
        masked_indices = torch.where(mask, indices, torch.full_like(indices, -1))
        sorted_cutoff_indices = masked_indices.max(dim=-1).values

        # Create full mask for sorted_logits/sorted_indices
        full_sorted_mask = torch.arange(scores.size(-1), device=device).unsqueeze(0) > sorted_cutoff_indices.unsqueeze(-1)

        # Map indices to original scores tensor
        full_mask = torch.zeros_like(scores, dtype=torch.bool)
        full_mask.scatter_(dim=-1, index=sorted_indices, src=full_sorted_mask)

        # Filter out logits using full_mask
        scores = scores.masked_fill(full_mask, float('-inf'))
        
        return scores
    
    def set_param(self, key, value):
        if key == "ratio_threshold":
            self.ratio_threshold = float(value)
            return True
        elif key == "k":
            self.k = int(value)
            return True
        else:
            return False
        
    def get_param_names(self):
        # Could return a dict of name->value for clarity:
        return {
            "ratio_threshold": self.ratio_threshold,
            "k": self.k
        }
    

from transformers import AutoTokenizer
import matplotlib as plt
import numpy as np

def my_plotting_fn(data, metadata, tokenizer):
    """
    Args:
    - 'data': 
        dict containing contents of the data
        stored by user's custom _write_log function,
        defined in user's logits processor class, for
        a single forward pass of the model. 
    - 'metadata':
        Data automatically stored by `base_processor` module,
        including `model_name` and paramaters in user's
        custom logits processor's __init__ signature.
    - 'tokenizer':
        Tokenizer asssociated to language model used
        in chat session used to generate `data`. 

    Returns:
    - 'fig':
        A maptplotlib figure of user's choice. 

    Note: 
        _write_log should store past_token_ids in order to 
        include user prompt and model response in title of
        plots. This logic is handled in base_analysis_script
    """
    # interesting_logged_data = data["interesting_feature"]
    # fig, ax = plt.subplots(figsize=(18,6))
    # xvals = np.arange(len(interesting_logged_data))
    # ax.plot(
    #     interesting_logged_data,
    #     xvals,
    #     color="blue",
    #     marker="o",
    #     label="something related to language models"
    # )

    # as.set_title("")

    # ax.legend()
    # plt.tight_layout()

    # return fig
    pass
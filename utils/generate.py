import torch
from torch.nn import functional as F


def top_k_top_p_logits(logits, top_k, top_p, fill_value=-1e8):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, fill_value)

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, fill_value)
    return logits


@torch.no_grad()
def sample_sequence(
    model,
    context,
    stop_token_ids,
    max_gen_seq_length=512,
    temperature=1.0,
    repetition_penalty=1.0,
    top_k=0,
    top_p=1.0,
):
    """
    Take a conditioning sequence of indices idx and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """

    generated = context

    for _ in range(max_gen_seq_length):
        # If the sequence context is growing too long, we must crop it at block_size
        generated = generated if generated.size(1) <= model.block_size else generated[:, -model.block_size :]  # noqa: E203

        # Forward the model to get the logits for the indices in the sequence
        next_token_logits = model(generated, is_inference=True)

        # Pluck the logits at the final step and scale by the desired temperature
        next_token_logits = next_token_logits / (temperature if temperature > 0 else 1.0)

        # Repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
        if repetition_penalty > 1:
            for token in set(generated.view(-1).tolist()):
                next_token_logits[..., token] /= repetition_penalty

        next_token_logits = top_k_top_p_logits(next_token_logits, top_k, top_p)

        # Apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(next_token_logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat((generated, next_token), dim=1)

        next_token_id = next_token.squeeze().cpu().item()
        if any(next_token_id == tk for tk in stop_token_ids):
            break

    return generated

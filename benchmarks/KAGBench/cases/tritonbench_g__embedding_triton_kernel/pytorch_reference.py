import torch


@torch.no_grad()
def embedding(input_ids, weight: torch.Tensor, vob_start_id, vob_end_id, out: torch.Tensor):
    """Reference implementation for the public embedding API."""
    out.zero_()
    valid = (input_ids >= vob_start_id) & (input_ids < vob_end_id)
    rows = (input_ids[valid].to(torch.long) - int(vob_start_id))
    out[valid] = weight.index_select(0, rows)
    return out
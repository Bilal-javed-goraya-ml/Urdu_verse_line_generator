import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch


# Collate Function for Padding
def pad_batch(batch):
    inputs, outputs = zip(*batch)
    input_lens = [len(seq) for seq in inputs]
    output_lens = [len(seq) for seq in outputs]

    max_input_len = max(input_lens)
    max_output_len = max(output_lens)

    pad = lambda seqs, max_len: [torch.cat([s, torch.full((max_len - len(s),), 0)]) for s in seqs]

    padded_inputs = torch.stack(pad(inputs, max_input_len))
    padded_outputs = torch.stack(pad(outputs, max_output_len))
    return padded_inputs.long(), padded_outputs.long()

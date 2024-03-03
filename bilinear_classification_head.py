import torch
import torch.nn as nn

class BilinearClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim1=768,
        input_dim2=768,
        inner_dim=2048,
        num_classes=19,
        pooler_dropout=0.4,
    ):
        super().__init__()
        self.dense = nn.Bilinear(input_dim1, input_dim2, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states1: torch.Tensor, hidden_states2: torch.Tensor):
        hidden_states1 = self.dropout(hidden_states1)
        hidden_states2 = self.dropout(hidden_states2)
        hidden_states = self.dense(hidden_states1, hidden_states2)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
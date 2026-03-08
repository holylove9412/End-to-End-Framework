
import torch
import torch.nn as nn
class FullyConnectedNN(nn.Module):
    def __init__(
        self,
        in_dims,
        out_dims,
        hidden_dim,
        n_hidden_layers,
        with_bias=True,
        non_linearity="Identity",
        final_bias=True,
    ):

        super(FullyConnectedNN, self).__init__()
        self.with_bias = with_bias
        self.non_linearity = non_linearity
        self.linear_stack = nn.Sequential()

        self._append_block(in_dims, hidden_dim, bias=self.with_bias)##原始输入特征线性映射

        for _ in range(n_hidden_layers):
            self._append_block(hidden_dim, hidden_dim, bias=self.with_bias)

        self._append_block(hidden_dim, out_dims, bias=final_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_stack(x)
        return x

    def _append_block(self, in_dim, out_dim, bias=True):
        self.linear_stack.append(nn.Linear(in_dim, out_dim, bias=bias))
        self.linear_stack.append(getattr(nn, self.non_linearity)())

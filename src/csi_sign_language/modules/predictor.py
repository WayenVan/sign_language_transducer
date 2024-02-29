from torch import nn
from einops import rearrange
import torch
from .rnn import RnnLayer

class BasePredictor(nn.Module):
    def __init__(self, d_model, output_size, vocab_size, n_layers, dropout=0.2, share_weight=False, use_norm=True):
        super(BasePredictor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.rnn = RnnLayer(d_model, hidden_size=d_model, dropout=dropout, num_layers=n_layers, bidirectional=False)
        self.output_proj = nn.Linear(d_model, output_size)
        self.use_norm = use_norm
        if use_norm:
            self.ln = nn.LayerNorm(d_model)

        if share_weight:
            self.embedding.weight = self.output_proj.weight

    def forward(self, inputs, length=None, hidden=None):
        """
        :param inputs: [n, max_u]
        :param length: [n]
        """
        N, U = inputs.shape

        if length is None:
            #inference mode
            assert N == U == 1
            length = torch.LongTensor([1])

        embed_inputs = self.embedding(inputs)
        #n, max_u, d_model
        embed_inputs = rearrange(embed_inputs, 'n u d -> u n d')

        rnn_outputs = self.rnn(embed_inputs, length, hidden=hidden)
        outputs = rnn_outputs['out']
        hidden = rnn_outputs['hidden']
        
        if self.use_norm:
            outputs = self.ln(outputs)

        outputs = self.output_proj(outputs)

        #[u n d]
        return dict(
            out=outputs,
            hidden=hidden
        )
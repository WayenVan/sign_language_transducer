from torch import nn
import torch
from einops import repeat,rearrange
from ..utils.object import add_attributes

class JointNet(nn.Module):
    def __init__(self, enc_dim, pred_dim, d_model, vocab_size):
        super(JointNet, self).__init__()
        add_attributes(self, locals())
        self.forward_layer = nn.Linear(enc_dim+pred_dim, d_model, bias=True)
        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, enc_state, pred_state):
        """
        :param enc_state: [t, n, d] or [d] when inference
        :param pred_state: [u, n, d] or [d] when inference
        """
        is_inference = False
        if enc_state.dim() == 3 and pred_state.dim() == 3:

            t = enc_state.size(0)
            u = pred_state.size(0)
            
            enc_state = repeat(enc_state, 't n d -> t u n d', u=u)
            pred_state = repeat(pred_state, 'u n d -> t u n d', t=t)
        else:
            is_inference = True
            assert enc_state.dim() == pred_state.dim()
            assert len(enc_state.shape) == 1 

        concat_state = torch.cat((enc_state, pred_state), dim=-1)
        outputs = self.forward_layer(concat_state)

        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)
        
        #[t u n d]
        if not is_inference:
            outputs = rearrange(outputs, 't u n d -> n t u d')
        return outputs

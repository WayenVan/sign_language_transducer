import torch.nn as nn
from ..rnn import RnnLayer
from ...modules.x3d import X3d
from einops import rearrange, repeat

from csi_sign_language.utils.object import add_attributes

class X3dLSTM(nn.Module):

    def __init__(self, d_model, n_layers, x3d_type='x3d_s', dropout=0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        add_attributes(self, locals())
        self.x3d = X3d(d_model, x3d_type, dropout)
        self.rnn = RnnLayer(d_model, hidden_size=d_model, num_layers=n_layers, bidirectional=False)

    def forward(self, x, video_length):
        """
        :param x: [t, n, c, h, w]
        """
        T, N, C, H, W = x.shape
        assert (H, W) == self.x3d.input_size_spatial, f"expect size {self.x3d.input_size_spatial}, got size ({H}, {W})"

        x = rearrange(x, 't n c h w -> n c t h w')
        x, video_length = self.x3d(x, video_length)
        x = rearrange(x, 'n c t -> t n c')
        cnn_out = x
        x = self.rnn(x, video_length)['out']
        
        return dict(
            out = x,
            cnn_out = cnn_out,
            video_length = video_length
        )

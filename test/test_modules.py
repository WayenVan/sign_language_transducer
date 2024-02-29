import torch
import hydra
import sys
sys.path.append('src')
from hydra.utils import instantiate
from omegaconf import OmegaConf
from csi_sign_language.modules.rnn import RnnLayer
from torch import nn

def test_rnn():
    s1 = torch.ones(21, 100)
    s2 = torch.zeros(24, 100)
    x = nn.utils.rnn.pad_sequence([s1, s2], padding_value=0.)
    l = torch.IntTensor([21, 24])
    rnn = RnnLayer(100, num_layers=2)
    y = rnn(x, l)['out']
    y1 = y[:, 0]
    y2 = y[:, 1]
    
    print(y1)
    print(y2)
    
    return

test_rnn()

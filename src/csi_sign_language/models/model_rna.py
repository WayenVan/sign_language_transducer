import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from einops import rearrange, repeat
from ..modules.resnet.resnet import *
from ..modules.tconv import *
from ..utils.object import add_attributes

from warp_rnnt import rnnt_loss


class SLRTransducer(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        joint_net: nn.Module,
        vocab,
        blank_idx=0,
        **kwargs
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.joint_net = joint_net
        self.blank_idx = blank_idx
        self.start_token = nn.Parameter(
            torch.IntTensor([blank_idx]), requires_grad=False)
        self.vocab = vocab

    @staticmethod
    def _target_transform(target, target_length):
        # [sum(target_lengthjs)], [n]
        # return [max_u, n]
        N = target_length.size(0)
        targets = []
        start_index = 0
        for length in target_length.detach().int().cpu().numpy():
            targets.append(target[start_index:start_index+length])
            start_index += length

        assert len(targets) == N

        return nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0.0)

    def _cat_target(self, target, target_length):
        """

        :param target: [max_u, n]
        :param target_length: [n]
        """
        n = target_length.size(0)

        start_token = repeat(self.start_token, 'u -> n u', n=n)

        cat_target = torch.cat([start_token, target], dim=-1)
        target_length = target_length + 1
        return cat_target, target_length
    
    @torch.no_grad()
    def _decode(self, enc_state, lengths):
        """decode single batch

        :param enc_state: [t d]
        :param lengths: int, scaler representing the length of dimension t
        """

        token_list = []
        start_token = repeat(self.start_token, 'u -> n u', n=1)
        pred_out = self.predictor(start_token)
        pred_state, hidden = pred_out['out'], pred_out['hidden']

        for t in range(lengths):
            logits = self.joint_net(enc_state[t].view(-1), pred_state.view(-1))
            out = F.softmax(logits, dim=0).detach()
            pred = torch.argmax(out, dim=0)
            pred = int(pred.item())

            if pred != self.blank_idx:
                token_list.append(pred)
                token = torch.LongTensor([[pred]]).to(enc_state.device)
                pred_out = self.predictor(token, hidden=hidden)
                pred_state, hidden = pred_out['out'], pred_out['hidden']

        return token_list

    def forward(self, video, video_length, target, target_length):
        """
        video: [t, n, c, h, w]
        target: [sum(target_lengths)]
        """
        enc_out = self.encoder(video, video_length)

        target = self._target_transform(target, target_length)
        cat_target, target_length = self._cat_target(target, target_length)

        pred_out = self.predictor(cat_target, target_length)
        joint_out = self.joint_net(enc_out['out'], pred_out['out'])

        return dict(
            out=joint_out,
            # [n, t, u, d]
            video_length=enc_out['video_length'],
            transformed_target=target
        )

    def criterion(self, outputs, target, target_length):
        log_probs = F.log_softmax(outputs['out'], dim=-1)
        labels = outputs['transformed_target']
        video_length = outputs['video_length']
        assert labels.size(-1) == log_probs.size(-2) - 1
        loss = rnnt_loss(
            log_probs, labels.int(), video_length.int(), target_length.int(), reduction='mean')
        return loss

    @torch.no_grad()
    def inference(self, video, video_length) -> List[List[str]]:
        """
        video: [t, n, c, h, w]
        target: [sum(target_lengths)]
        """
        T, N, C, H, W = video.shape

        results = []
        enc_out = self.encoder(video, video_length)
        #t n d
        enc_state = enc_out['out']
        #n
        video_length = enc_out['video_length']
        for i in range(N):
            decoded_seq = self._decode(enc_state[:, i], video_length[i])
            assert all(idx != self.blank_idx for idx in decoded_seq)
            decoded_labels = self.vocab.lookup_tokens(decoded_seq)
            results.append(decoded_labels)
        return results
        

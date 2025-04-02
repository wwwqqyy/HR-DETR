import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register
from ..backbones.csp_darknet import BaseConv

__all__ = ['ModalityInteraction']


class ShiftShuffle(nn.Layer):
    def __init__(self, reverse=False, modalities=2):
        super(ShiftShuffle, self).__init__()
        self.pos = [[-1, 0], [0, -1], [0, 1], [1, 0]]
        if reverse:
            self.pos = self.pos[::-1]
        self.modalities = modalities

    def forward(self, x):
        if len(x) != self.modalities:
            return x, [0] * self.modalities

        shift_group = x[0].shape[1] // 5
        shuffle_channel = shift_group * 4

        x_parts_a = [x_i[:, :shuffle_channel] for x_i in x]
        x_parts_b = [x_i[:, shuffle_channel:] for x_i in x]

        shuffled = []
        for i in range(self.modalities):
            next_idx = (i + 1) % self.modalities
            shuffled.append(paddle.concat([x_parts_a[i], x_parts_b[next_idx]], axis=1))

        h, w = x_parts_a[0].shape[-2:]
        pad = [1, 1, 1, 1]

        shifted_parts = [[] for _ in range(self.modalities)]

        for mod_idx in range(self.modalities):
            for shift_idx, i in enumerate(range(0, shuffle_channel, shift_group)):
                posh, posw = self.pos[shift_idx][0] + 1, self.pos[shift_idx][1] + 1
                padded = F.pad(x_parts_a[mod_idx][:, i:i + shift_group], pad)
                shifted_part = padded[:, :, posh:h + posh, posw:w + posw]
                shifted_parts[mod_idx].append(shifted_part)

            shifted_parts[mod_idx].append(paddle.zeros_like(x_parts_b[mod_idx]))

        shifted = [paddle.concat(parts, axis=1) for parts in shifted_parts]

        return shuffled, shifted


@register
class ModalityInteraction(nn.Layer):
    def __init__(self, channels=512, bias=False):
        super(ModalityInteraction, self).__init__()

        self.shift_shuffle1 = ShiftShuffle(reverse=False)
        self.shift_shuffle2 = ShiftShuffle(reverse=True)

        self.conv1_1 = BaseConv(in_channels=channels, out_channels=channels // 2, ksize=1, stride=1,
                                bias=bias)
        self.conv1_2 = BaseConv(in_channels=channels, out_channels=channels // 2, ksize=1, stride=1,
                                bias=bias)

        self.conv2_1 = BaseConv(in_channels=channels // 2, out_channels=channels // 2, ksize=3, stride=1,
                                bias=bias)
        self.conv2_2 = BaseConv(in_channels=channels // 2, out_channels=channels // 2, ksize=3, stride=1,
                                bias=bias)

        self.conv3_1 = BaseConv(in_channels=channels // 2, out_channels=channels, ksize=1, stride=1,
                                bias=bias)
        self.conv3_2 = BaseConv(in_channels=channels // 2, out_channels=channels, ksize=1, stride=1,
                                bias=bias)

    def forward(self, vis_body_feats, ir_body_feats):
        residual = [vis_body_feats, ir_body_feats]

        vis_body_feats = self.conv1_1(vis_body_feats)
        ir_body_feats = self.conv1_2(ir_body_feats)

        out, shift = self.shift_shuffle1([vis_body_feats, ir_body_feats])

        out[0] = self.conv2_1(out[0])
        out[1] = self.conv2_2(out[1])

        out[0] = out[0] + shift[1]
        out[1] = out[1] + shift[0]
        out, _ = self.shift_shuffle2(out)

        out[0] = self.conv3_1(out[0])
        out[1] = self.conv3_2(out[1])

        return out[0] + residual[0], out[1] + residual[1]

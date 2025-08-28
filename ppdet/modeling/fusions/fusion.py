import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register, serializable
from ppdet.modeling.initializer import conv_init_
from ppdet.utils.logger import setup_logger
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
# @serializable
class ModalityInteraction(nn.Layer):
    def __init__(self, channels=512, gamma_init=10.0, bias=False, use_gamma=True):  # 新增use_gamma参数
        super(ModalityInteraction, self).__init__()
        self.use_gamma = use_gamma  # 控制是否使用gamma

        if self.use_gamma:
            self.gamma = self.create_parameter(
                shape=[1],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(gamma_init)
            )
            # 训练状态跟踪（仅gamma启用时记录）
            self.step_counter = 0
            self.log_interval = 200
            self.logger = setup_logger('fusion', './log/gamma.log')

        self.shift_shuffle1 = ShiftShuffle(reverse=False)
        self.shift_shuffle2 = ShiftShuffle(reverse=True)

        # 其余卷积层初始化保持不变...
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

    def _log_gammas(self):
        """仅当use_gamma=True时生效"""
        if self.use_gamma and hasattr(self, 'logger'):
            self.logger.info(
                "Step {}: gamma={:.8f}".format(
                    self.step_counter,
                    float(F.sigmoid(self.gamma).numpy()[0])
                )
            )

    def forward(self, vis_body_feats, ir_body_feats):
        # 训练日志记录（仅gamma启用时）
        if self.training and self.use_gamma:
            if self.step_counter % self.log_interval == 0:
                self._log_gammas()
            self.step_counter += 1

        # 保存原始特征
        vis_residual = vis_body_feats
        ir_residual = ir_body_feats

        # 特征提取
        vis_body_feats = self.conv1_1(vis_body_feats)
        ir_body_feats = self.conv1_2(ir_body_feats)

        # 模态交互
        out, shift = self.shift_shuffle1([vis_body_feats, ir_body_feats])

        # 进一步特征变换
        out[0] = self.conv2_1(out[0])
        out[1] = self.conv2_2(out[1])

        # 特征融合
        out[0] = out[0] + shift[1]
        out[1] = out[1] + shift[0]
        out, _ = self.shift_shuffle2(out)

        # 最终特征提取
        out[0] = self.conv3_1(out[0])
        out[1] = self.conv3_2(out[1])

        # 修改融合部分
        if self.use_gamma:
            gamma = F.sigmoid(self.gamma)
            vis_out = gamma * out[0] + (1 - gamma) * vis_residual
            ir_out = gamma * out[1] + (1 - gamma) * ir_residual
        else:
            vis_out = out[0] + vis_residual  # 直接相加
            ir_out = out[1] + ir_residual

        return vis_out, ir_out
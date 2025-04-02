# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
from ..fusions.fusion import ModalityInteraction

__all__ = ['HrDETR']


@register
class HrDETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone_fusion,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck_fusion=None,
                 post_process='DETRPostProcess',
                 with_mask=False,
                 exclude_post_process=False):
        super(HrDETR, self).__init__()

        self.backbone_fusion = backbone_fusion
        self.interaction = ModalityInteraction(channels=512)
        self.transformer = transformer
        self.detr_head = detr_head

        self.neck_fusion = neck_fusion

        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process


    @classmethod
    def from_config(cls, cfg, *args, **kwargs):

        backbone_fusion = create(cfg['backbone_fusion'])

        kwargs = {'input_shape': backbone_fusion.out_shape}

        neck_fusion = create(cfg['neck_fusion'], **kwargs) if cfg['neck_fusion'] else None

        if neck_fusion is not None:
            kwargs = {'input_shape': neck_fusion.out_shape}
        transformer = create(cfg['transformer'], **kwargs)

        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone_fusion.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone_fusion': backbone_fusion,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck_fusion": neck_fusion
        }

    def _forward(self):

        vis_body_feats = self.backbone_fusion(self.inputs, 1)
        ir_body_feats = self.backbone_fusion(self.inputs, 2)

        vis_body_feats[0], ir_body_feats[0] = self.interaction(vis_body_feats[0], ir_body_feats[0])

        if self.neck_fusion is not None:
            vis_body_feats = self.neck_fusion(vis_body_feats)
            ir_body_feats = self.neck_fusion(ir_body_feats)

        fusion_feats = [vis_body_feat + ir_body_feat for vis_body_feat, ir_body_feat in (zip(vis_body_feats, ir_body_feats))]

        pad_mask = self.inputs.get('pad_mask', None)

        out_transformer = self.transformer(fusion_feats, vis_body_feats, ir_body_feats, pad_mask, self.inputs)

        if self.training:
            detr_losses = self.detr_head(out_transformer, None, self.inputs)
            detr_losses.update({
                'loss': paddle.add_n(
                    [v for k, v in detr_losses.items() if 'log' not in k])
            })
            return detr_losses
        else:
            preds = self.detr_head(out_transformer, None)
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    paddle.shape(self.inputs['vis_image'])[2:])

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

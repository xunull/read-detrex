# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn.functional as F
import torch
import copy
import torch.nn as nn
from detrex.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    MultiheadAttention,
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
    get_sine_pos_embed,
)
from detrex.utils import inverse_sigmoid
from .transformer_layer import Focus_DETR_BaseTransformerLayer


class FOCUS_DETRTransformerEncoder(TransformerLayerSequence):
    def __init__(
            self,
            embed_dim: int = 256,
            num_heads: int = 8,
            feedforward_dim: int = 1024,
            attn_dropout: float = 0.1,
            ffn_dropout: float = 0.1,
            num_layers: int = 6,
            post_norm: bool = False,
            num_feature_levels: int = 4,
    ):
        super(FOCUS_DETRTransformerEncoder, self).__init__(
            transformer_layers=Focus_DETR_BaseTransformerLayer(
                attn=[
                    # 多了一个Attention，标准的多头Attention
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=True,
                    ),
                    # 正常的Deformable Attention
                    MultiScaleDeformableAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=attn_dropout,
                        batch_first=True,
                        num_levels=num_feature_levels,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    num_fcs=2,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),

                operation_order=("OESM", "encoder_cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )

        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

        # 论文中Multi-category score predictor章节
        # 就是一个分类头，各个encoder共享的
        self.enhance_MCSP = None

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(
            self,
            backbone_mask_prediction,

            focus_token_nums,
            foreground_inds,

            reference_points,
            query,
            key,
            value,
            query_pos=None,
            key_pos=None,
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
            **kwargs,
    ):
        """
        backbone_mask_prediction [bs,sum(hw)] 前景的分数
        focus_token_nums (bs,) 各个image有效的token数量
        foreground_inds list(6) 各个encoder的， [bs,cascade*foreground_topk]

        key value 是原始的token
        """
        # bs, sum(hw),采样点数量,xy
        B_, N_, S_, P_ = reference_points.shape
        ori_reference_points = reference_points
        ori_pos = query_pos
        output = query

        # 6 层encoder
        for layer_id, layer in enumerate(self.layers):
            # 从output中取出对应的前景token
            query = torch.gather(output, 1, foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, output.size(-1)))
            # 取出相对应的query_pos
            query_pos = torch.gather(ori_pos, 1,
                                     foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, query_pos.size(-1)))
            # 取出前景分数值
            foreground_pre_layer = torch.gather(backbone_mask_prediction, 1, foreground_inds[layer_id])
            # 取出参考点位
            reference_points = torch.gather(ori_reference_points.view(B_, N_, -1), 1,
                                            foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, S_ * P_)).view(B_, -1,
                                                                                                                S_,
                                                                                                                P_)
            dropflag = False
            # 每个token属于class的分数 # [bs,x,80]
            # 算法第一行
            score_tgt = self.enhance_MCSP[layer_id](query)

            query = layer(
                # 前景分数值
                foreground_pre_layer,
                # 每个token属于class的分数
                score_tgt,
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points,
                **kwargs,
            )

            outputs = []
            # 算法12行
            # for的是bs
            for i in range(foreground_inds[layer_id].shape[0]):
                # focus_token_nums限制取token的最大数量
                # 修改原始的output中token的内容
                # 以foreground_inds的id，从query中取值，更新到output中
                outputs.append(output[i].scatter(0, foreground_inds[layer_id][i][:focus_token_nums[i]].unsqueeze(
                    -1).repeat(1, query.size(-1)), query[i][:focus_token_nums[i]]))

            output = torch.stack(outputs)

        if self.post_norm_layer is not None:
            output = self.post_norm_layer(output)
        return output


def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# 与DINO一样的Decoder
class FOCUS_DETRTransformerDecoder(TransformerLayerSequence):
    def __init__(
            self,
            embed_dim: int = 256,
            num_heads: int = 8,
            feedforward_dim: int = 1024,
            attn_dropout: float = 0.1,
            ffn_dropout: float = 0.1,
            num_layers: int = 6,
            return_intermediate: bool = True,
            num_feature_levels: int = 4,
            look_forward_twice=True,
    ):
        super(FOCUS_DETRTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=True,
                    ),
                    MultiScaleDeformableAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=attn_dropout,
                        batch_first=True,
                        num_levels=num_feature_levels,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate
        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)
        self.bbox_embed = None
        self.class_embed = None
        self.look_forward_twice = look_forward_twice
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
            self,
            query,
            key,
            value,
            reference_points=None,  # num_queries, 4. normalized.
            query_pos=None,
            key_pos=None,
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
            valid_ratios=None,
            **kwargs,
    ):
        output = query
        bs, num_queries, _ = output.size()
        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)  # bs, num_queries, 4
        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                        reference_points[:, :, None]
                        * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                **kwargs,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.look_forward_twice:
                    intermediate_reference_points.append(new_reference_points)
                else:
                    intermediate_reference_points.append(reference_points)
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points


class FOCUS_DETRTransformer(nn.Module):
    """Transformer module for FOCUS_DETR

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 900.
    """

    def __init__(
            self,
            encoder=None,
            decoder=None,
            num_feature_levels=4,
            two_stage_num_proposals=900,
            learnt_init_query=True,
    ):
        super(FOCUS_DETRTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        # todo 这几行是新的
        self.focus_rho = 0.5
        # 6层encoder使用的
        self.cascade_set = torch.Tensor([1.0, 0.8, 0.6, 0.6, 0.4, 0.2])

        # 公式2中的超参数
        self.alpha = nn.Parameter(data=torch.Tensor(3), requires_grad=True)
        self.alpha.data.uniform_(-0.3, 0.3)

        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dim = self.encoder.embed_dim
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))

        # DINO那个
        self.learnt_init_query = learnt_init_query
        if self.learnt_init_query:
            self.tgt_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dim)

        self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
        self.enc_output_norm = nn.LayerNorm(self.embed_dim)

        # FTS
        self.enc_mask_predictor = MaskPredictor(self.embed_dim, self.embed_dim)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.normal_(self.level_embeds)

    # 这个方法正常只在encoder后decoder之前调用
    # 在Focus-DETR中也会在encoder之前调用一次
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes, process_output=True):
        """Make region proposals for each multi-scale features considering their shapes and padding masks,
        and project & normalize the encoder outputs corresponding to these proposals.
            - center points: relative grid coordinates in the range of [0.01, 0.99] (additional mask)
            - width/height:  2^(layer_id) * s (s=0.05) / see the appendix A.4

        Tensor shape example:
            Args:
                memory: torch.Size([2, 15060, 256])
                memory_padding_mask: torch.Size([2, 15060])
                spatial_shape: torch.Size([4, 2])
            Returns:
                output_memory: torch.Size([2, 15060, 256])
                    - same shape with memory ( + additional mask + linear layer + layer norm )
                output_proposals: torch.Size([2, 15060, 4])
                    - x, y, w, h
        """
        N_, S_, C_ = memory.shape
        proposals = []
        _cur = 0

        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # level of encoded feature scale
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  # inverse of sigmoid
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))  # sigmoid(inf) = 1

        output_memory = memory

        # 这个地方不一样,这是给encoder之前调用时准备的
        if process_output:
            output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
            output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals, (~memory_padding_mask).sum(axis=-1)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Deformable DETR的参考点
        Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    # top down score使用的
    def upsamplelike(self, inputs):
        src, size = inputs
        return F.interpolate(src, size, mode='bilinear', align_corners=True)

    def forward(
            self,
            multi_level_feats,
            multi_level_masks,
            multi_level_pos_embeds,
            query_embed,
            attn_masks,
            **kwargs,
    ):

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)

        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=feat.device)
        # 以上与DINO相同的

        # 这里是新增的
        if self.focus_rho:
            # valid_token_nums 各个image 有效的token的数量
            # 用backbone的token筛选一次
            backbone_output_memory, backbone_output_proposals, valid_token_nums = self.gen_encoder_output_proposals(
                feat_flatten + lvl_pos_embed_flatten,
                mask_flatten,
                spatial_shapes,
                process_output=bool(self.focus_rho))

            # 有效的token数量
            self.valid_token_nums = valid_token_nums
            # 有效token数量 * 比例
            focus_token_nums = (valid_token_nums * self.focus_rho).int() + 1
            # 前景token数量，选择这个batch中最大的一个数量
            foreground_topk = int(max(focus_token_nums))
            self.focus_token_nums = focus_token_nums
            # 每个encoder使用的token的数量
            # cascade_set tensor([1.0000, 0.8000, 0.6000, 0.6000, 0.4000, 0.2000])
            encoder_foreground_topk = self.cascade_set * foreground_topk

            foreground_score = []

            # 实际是从高层级特征向下进行计算的 top down score modulation
            for i in range(self.num_feature_levels):
                if i == 0:
                    # 这里取出的是最高层的特征层的token
                    backbone_lvl = backbone_output_memory[:, level_start_index[self.num_feature_levels - 1]:, :]
                    # token的分数 [bs,1,h,w]
                    # 图5中的MLP，稍微复杂一点的MLP
                    score_prediction_lvl = self.enc_mask_predictor(backbone_lvl).reshape(bs, 1, spatial_shapes[
                        self.num_feature_levels - 1][0], spatial_shapes[self.num_feature_levels - 1][1])
                    # [bs,hw,1]，自带了调整
                    foreground_score.append(score_prediction_lvl.view(bs, -1, 1))
                else:
                    backbone_lvl = backbone_output_memory[:,
                                   level_start_index[self.num_feature_levels - i - 1]:level_start_index[
                                       self.num_feature_levels - i - 0], :]
                    # 这时的score_prediction_lvl还是上一个level的，上一个高层级level的,score进行上采样
                    up_score = self.upsamplelike((score_prediction_lvl, (
                        spatial_shapes[self.num_feature_levels - i - 1][0],
                        spatial_shapes[self.num_feature_levels - i - 1][1])))
                    # [bs,hw,256] -> [bs,256,h,w]
                    re_backbone_lvl = backbone_lvl.reshape(bs, spatial_shapes[self.num_feature_levels - i - 1][0],
                                                           spatial_shapes[self.num_feature_levels - i - 1][1],
                                                           -1).permute(0, 3, 1, 2)
                    # 公式2
                    # 使用高层级的分数调制backbone的token
                    # 图5，上一个level的分数经过上采样后，乘上一个因子，然后与当前的token相乘，最后加上当前层的token
                    # [bs,hw,256]
                    backbone_lvl = backbone_lvl + (re_backbone_lvl *
                                                   up_score *
                                                   self.alpha[i - 1]).permute(0, 2, 3, 1).reshape(
                        bs, -1, self.embed_dim)

                    # 然后经过FTS得到分数
                    score_prediction_lvl = self.enc_mask_predictor(backbone_lvl).reshape(bs, 1, spatial_shapes[
                        self.num_feature_levels - i - 1][0], spatial_shapes[self.num_feature_levels - i - 1][1])

                    foreground_score.append(score_prediction_lvl)

            # 存储的是前景分数
            # 到这里 foreground_score里的内容类似为 第一个[1,hw,1] 后面的为[1,1,h,w] 这就是上面处理的差异，因此这里需要将后面的几个hw推平
            # [bs,sum(hw),1]
            backbone_mask_prediction = torch.cat(
                [foreground_score[3 - i].view(bs, -1, 1) for i in range(len(foreground_score))], dim=1)

            # 最后的返回值使用
            temp_backbone_mask_prediction = backbone_mask_prediction

            # [bs,sum(hw),1] -> [bs,sum(hw)]
            backbone_mask_prediction = backbone_mask_prediction.squeeze(-1)
            # masked_fill
            backbone_mask_prediction = backbone_mask_prediction.masked_fill(mask_flatten,
                                                                            backbone_mask_prediction.min())

            # 前景token的id
            # 选择出前景token
            foreground_inds = []

            # todo cascade 对应6层encoder
            # cascade_set 是越来越小的,因此foreground_inds中的内容是越来越少的
            for i in range(len(self.cascade_set)):
                foreground_proposal = torch.topk(backbone_mask_prediction, int(encoder_foreground_topk[i]), dim=1)[1]
                foreground_inds.append(foreground_proposal)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None
        memory = self.encoder(
            # 前三个参数是这篇论文的
            backbone_mask_prediction=backbone_mask_prediction,
            # token数量 * 比例
            focus_token_nums=focus_token_nums,
            # 前景token的id
            foreground_inds=foreground_inds,
            # Deformable DETR使用的参考点位
            reference_points=reference_points,  # bs, num_token, num_level, 2
            query=feat_flatten,
            key=None,
            value=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        # 以下内容与DINO相同
        # 这里多的返回值也没有使用，第三个返回值是在上面那个调用时被使用的
        output_memory, output_proposals, _ = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        # output_memory: bs, num_tokens, c
        # output_proposals: bs, num_tokens, 4. unsigmoided.

        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        )
        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

        # extract region proposal boxes
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )
        # unsigmoided.
        reference_points = topk_coords_unact.detach().sigmoid()
        if query_embed[1] is not None:
            reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)
        init_reference_out = reference_points

        # extract region features
        target_unact = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )

        if self.learnt_init_query:
            # 保持内容查询为网络中的参数
            target = self.tgt_embed.weight[None].repeat(bs, 1, 1)
        else:
            target = target_unact.detach()

        if query_embed[0] is not None:
            target = torch.cat([query_embed[0], target], 1)

        # decoder
        # decoder的使用没有修改，与Deformable DETR相同
        inter_states, inter_references = self.decoder(
            query=target,  # bs, num_queries, embed_dims
            key=memory,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            reference_points=reference_points,  # num_queries, 4
            query_pos=None,
            key_padding_mask=mask_flatten,  # bs, num_tokens

            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,  # bs, nlvl, 2
            attn_masks=attn_masks,
            **kwargs,
        )

        inter_references_out = inter_references
        return (
            inter_states,
            init_reference_out,
            inter_references_out,
            target_unact,
            topk_coords_unact.sigmoid(),
            # 多的返回值
            temp_backbone_mask_prediction,
        )


# 这个是FTS模块
# 最后输出维度为1，表示前景分数
class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1)
        )

    def forward(self, x):
        z = self.layer1(x)

        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)

        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)

        z = torch.cat([z_local, z_global], dim=-1)

        out = self.layer2(z)
        return out

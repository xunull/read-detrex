# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
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
# ------------------------------------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py
# ------------------------------------------------------------------------------------------------


import copy
import warnings
from typing import List
import torch
import torch.nn as nn
from detrex.layers.mlp import FFN


# 与detrex中的基本一致，多了encoder_cross_attn的处理
class BaseTransformerLayer(nn.Module):
    # TODO: add more tutorials about BaseTransformerLayer
    """The implementation of Base `TransformerLayer` used in Transformer. Modified
    from `mmcv <https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py>`_.
    It can be built by directly passing the `Attentions`, `FFNs`, `Norms`
    module, which support more flexible cusomization combined with
    `LazyConfig` system. The `BaseTransformerLayer` also supports `prenorm`
    when you specifying the `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .
    Args:
        attn (list[nn.Module] | nn.Module): nn.Module or a list
            contains the attention module used in TransformerLayer.
        ffn (nn.Module): FFN module used in TransformerLayer.
        norm (nn.Module): Normalization layer used in TransformerLayer.
        operation_order (tuple[str]): The execution order of operation in
            transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying the first element as `norm`.
            Default = None.
    """

    def __init__(
            self,
            attn: List[nn.Module],
            ffn: nn.Module,
            norm: nn.Module,
            operation_order: tuple = None,
    ):
        """
        transformer中的attn网络，ffn网络，norm网络，传入进来
        """
        super(BaseTransformerLayer, self).__init__()
        # 模型的执行顺序, 比基础的多一个encoder_cross_attn
        assert set(operation_order).issubset({"self_attn", "encoder_cross_attn", "norm", "cross_attn", "ffn"})

        # count attention nums
        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn") + operation_order.count(
            "encoder_cross_attn")

        if isinstance(attn, nn.Module):
            attn = [copy.deepcopy(attn) for _ in range(num_attn)]
        else:
            assert len(attn) == num_attn, (
                f"The length of attn (nn.Module or List[nn.Module]) {num_attn}"
                f"is not consistent with the number of attention in "
                f"operation_order {operation_order}"
            )
        self.num_attn = num_attn
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = nn.ModuleList()
        index = 0

        for operation_name in operation_order:
            # 多了一个encoder_cross_attn
            if operation_name in ["self_attn", "encoder_cross_attn", "cross_attn"]:
                self.attentions.append(attn[index])
                index += 1
        self.embed_dim = self.attentions[0].embed_dim

        # count ffn nums
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count("ffn")
        for _ in range(num_ffns):
            self.ffns.append(copy.deepcopy(ffn))

        # count norm nums
        self.norms = nn.ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(copy.deepcopy(norm))

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor = None,
            value: torch.Tensor = None,
            query_pos: torch.Tensor = None,
            key_pos: torch.Tensor = None,
            attn_masks: List[torch.Tensor] = None,
            query_key_padding_mask: torch.Tensor = None,
            key_padding_mask: torch.Tensor = None,
            reference_points: torch.Tensor = None,
            **kwargs,
    ):
        """Forward function for `BaseTransformerLayer`.
        **kwargs contains the specific arguments of attentions.
        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
                which should be specified follows the attention module used in
                `BaseTransformerLayer`.
            key (torch.Tensor): Key embeddings used in `Attention`.
            value (torch.Tensor): Value embeddings with the same shape as `key`.
            query_pos (torch.Tensor): The position embedding for `query`.
                Default: None.
            key_pos (torch.Tensor): The position embedding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): A list of 2D ByteTensor used
                in calculation the corresponding attention. The length of
                `attn_masks` should be equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape `(bs, num_query)`. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (torch.Tensor): ByteTensor for `key`, with
                shape `(bs, num_key)`. Default: None.
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )
        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                # temp_key = temp_value = value
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=reference_points,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "encoder_cross_attn":
                # temp_key = temp_value = query
                temp_key = temp_value = value
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=reference_points,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    reference_points=reference_points,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        return query


# Focus-DETR的Encoder，修改为支持Dual Attention
class Focus_DETR_BaseTransformerLayer(nn.Module):
    # TODO: add more tutorials about BaseTransformerLayer
    """The implementation of Base `TransformerLayer` used in Transformer. Modified
    from `mmcv <https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/bricks/transformer.py>`_.
    It can be built by directly passing the `Attentions`, `FFNs`, `Norms`
    module, which support more flexible cusomization combined with
    `LazyConfig` system. The `BaseTransformerLayer` also supports `prenorm`
    when you specifying the `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .
    Args:
        attn (list[nn.Module] | nn.Module): nn.Module or a list
            contains the attention module used in TransformerLayer.
        ffn (nn.Module): FFN module used in TransformerLayer.
        norm (nn.Module): Normalization layer used in TransformerLayer.
        operation_order (tuple[str]): The execution order of operation in
            transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying the first element as `norm`.
            Default = None.
    """

    def __init__(
            self,
            attn: List[nn.Module],
            ffn: nn.Module,
            norm: nn.Module,
            operation_order: tuple = None,
    ):
        super(Focus_DETR_BaseTransformerLayer, self).__init__()

        # 相比于detrex的transformer 这个地方进行了修改为了适应这篇论文 "OESM", "encoder_cross_attn"
        assert set(operation_order).issubset({"self_attn", "OESM", "encoder_cross_attn", "norm", "cross_attn", "ffn"})
        self.topk_sa = 300
        # count attention nums
        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn") + operation_order.count(
            "encoder_cross_attn") + operation_order.count("OESM")
        if isinstance(attn, nn.Module):
            attn = [copy.deepcopy(attn) for _ in range(num_attn)]
        else:
            assert len(attn) == num_attn, (
                f"The length of attn (nn.Module or List[nn.Module]) {num_attn}"
                f"is not consistent with the number of attention in "
                f"operation_order {operation_order}"
            )
        self.num_attn = num_attn
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = nn.ModuleList()
        index = 0

        for operation_name in operation_order:
            if operation_name in ["self_attn", "OESM", "encoder_cross_attn", "cross_attn"]:
                self.attentions.append(attn[index])
                index += 1

        self.embed_dim = self.attentions[0].embed_dim
        # count ffn nums
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count("ffn")
        for _ in range(num_ffns):
            self.ffns.append(copy.deepcopy(ffn))
        # count norm nums
        self.norms = nn.ModuleList()
        self.norm2 = nn.LayerNorm(self.embed_dim)
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(copy.deepcopy(norm))

    def forward(
            self,
            # 前景分数
            foreground_pre_layer: torch.Tensor,
            # 每个token的class的分数
            score_tgt: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor = None,
            value: torch.Tensor = None,
            query_pos: torch.Tensor = None,
            key_pos: torch.Tensor = None,
            attn_masks: List[torch.Tensor] = None,
            query_key_padding_mask: torch.Tensor = None,
            key_padding_mask: torch.Tensor = None,
            reference_points: torch.Tensor = None,
            **kwargs,
    ):
        """Forward function for `BaseTransformerLayer`.
        **kwargs contains the specific arguments of attentions.
        Args:
            query (torch.Tensor): Query embeddings with shape
                `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
                which should be specified follows the attention module used in
                `BaseTransformerLayer`.
            key (torch.Tensor): Key embeddings used in `Attention`.
            value (torch.Tensor): Value embeddings with the same shape as `key`.
            query_pos (torch.Tensor): The position embedding for `query`.
                Default: None.
            key_pos (torch.Tensor): The position embedding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): A list of 2D ByteTensor used
                in calculation the corresponding attention. The length of
                `attn_masks` should be equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape `(bs, num_query)`. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (torch.Tensor): ByteTensor for `key`, with
                shape `(bs, num_key)`. Default: None.
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )
        # ('OESM', 'encoder_cross_attn', 'norm', 'ffn', 'norm')
        for layer in self.operation_order:

            # encoder中没有self_attn了
            if layer == "self_attn":

                temp_key = temp_value = query
                # temp_key = temp_value = value
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=reference_points,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "OESM":
                # 第一个 attention
                ori_tgt = query
                # 最大的类别的分数和前景分数相乘
                mc_score = score_tgt.max(-1)[0] * foreground_pre_layer
                # bs, nq # 最大的300个 位置index
                select_tgt_index = torch.topk(mc_score, self.topk_sa, dim=1)[1]
                # 取出对应的query
                select_tgt = torch.gather(query, 1, select_tgt_index.unsqueeze(-1).repeat(1, 1, 256))
                # 取出对应的query index
                select_pos = torch.gather(query_pos, 1, select_tgt_index.unsqueeze(-1).repeat(1, 1, 256))
                temp_key = temp_value = select_tgt
                # 进行attention计算
                tgt2 = self.attentions[attn_index](
                    select_tgt,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=select_pos,
                    key_pos=select_pos,
                    reference_points=reference_points,
                    **kwargs,
                )
                tgt2 = self.norm2(tgt2)
                # 更新query
                query = ori_tgt.scatter(1, select_tgt_index.unsqueeze(-1).repeat(1, 1, tgt2.size(-1)), tgt2)
                attn_index += 1
                identity = query

            elif layer == "encoder_cross_attn":
                # 第二个attention，这个是Deformable Attention
                # temp_key = temp_value = query
                temp_key = temp_value = value  # 这里用的是传入进来的value，算是cross attention
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=reference_points,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    reference_points=reference_points,
                    **kwargs,
                )
                attn_index += 1
                identity = query
            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        return query


class TransformerLayerSequence(nn.Module):
    """Base class for TransformerEncoder and TransformerDecoder, which will copy
    the passed `transformer_layers` module `num_layers` time or save the passed
    list of `transformer_layers` as parameters named ``self.layers``
    which is the type of ``nn.ModuleList``.
    The users should inherit `TransformerLayerSequence` and implemente their
    own forward function.

    Args:
        transformer_layers (list[BaseTransformerLayer] | BaseTransformerLayer): A list
            of BaseTransformerLayer. If it is obj:`BaseTransformerLayer`, it
            would be repeated `num_layers` times to a list[BaseTransformerLayer]
        num_layers (int): The number of `TransformerLayer`. Default: None.
    """

    def __init__(
            self,
            transformer_layers=None,
            num_layers=None,
    ):
        super(TransformerLayerSequence, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        if isinstance(transformer_layers, nn.Module):
            for _ in range(num_layers):
                self.layers.append(copy.deepcopy(transformer_layers))
        else:
            assert isinstance(transformer_layers, list) and len(transformer_layers) == num_layers

    def forward(self):
        """Forward function of `TransformerLayerSequence`. The users should inherit
        `TransformerLayerSequence` and implemente their own forward function.
        """
        raise NotImplementedError()

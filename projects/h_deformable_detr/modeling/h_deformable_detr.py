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

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances

# 原作者
class HDeformableDETR(nn.Module):
    """Implements the Deformable DETR model.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2010.04159>`_ .

    Args:
        backbone (nn.Module): the backbone module.
        position_embedding (nn.Module): the position embedding module.
        neck (nn.Module): the neck module.
        transformer (nn.Module): the transformer module.
        embed_dim (int): the dimension of the embedding.
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): whether to use auxiliary loss. Default: True.
        with_box_refine (bool): whether to use box refinement. Default: False.
        as_two_stage (bool): whether to use two-stage. Default: False.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 100.

    """

    def __init__(
            self,
            backbone,
            position_embedding,
            neck,
            transformer,
            embed_dim,
            num_classes,
            # DeformableDETR为num_queries
            num_queries_one2one,
            num_queries_one2many,
            criterion,
            pixel_mean,
            pixel_std,
            aux_loss=True,
            with_box_refine=False,
            as_two_stage=False,
            select_box_nums_for_evaluation=100,
            device="cuda",
            # 多的参数
            mixed_selection=True,
            # 重复GT的次数
            k_one2many=6,
            # one-to-many loss的系数
            lambda_one2many=1.0,
    ):
        super().__init__()
        # 总的query数量
        num_queries = num_queries_one2one + num_queries_one2many
        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding

        # define neck module
        self.neck = neck

        # define learnable query embedding
        self.num_queries = num_queries
        if not as_two_stage:
            # 如果使用two_stage，那么query的来源是encoder的memory topk
            self.query_embedding = nn.Embedding(num_queries, embed_dim * 2)
        elif mixed_selection:
            # todo
            self.query_embedding = nn.Embedding(num_queries, embed_dim)

        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        self.num_classes = num_classes
        self.class_embed = nn.Linear(embed_dim, num_classes)
        # 3层，基本都是3层
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        # define contoller for box refinement and two-stage variants
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage

        # init parameters for heads
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # If two-stage, the last class_embed and bbox_embed is for region proposal generation
        # Decoder layers share the same heads without box refinement, while use the different
        # heads when box refinement is used.
        num_pred = (
            (transformer.decoder.num_layers + 1) if as_two_stage else transformer.decoder.num_layers
        )
        if with_box_refine:
            self.class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for i in range(num_pred)]
            )
            self.bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for i in range(num_pred)]
            )
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        # hack implementation for two-stage. The last class_embed and bbox_embed is for region proposal generation
        if as_two_stage:
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # H-DETR的参数
        self.num_queries_one2one = num_queries_one2one
        self.mixed_selection = mixed_selection
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)

        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                # mask padding regions in batched images
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)

            # todo
            # disable the one-to-many branch queries
            # save them frist
            save_num_queries = self.num_queries
            save_two_stage_num_proposals = self.transformer.two_stage_num_proposals
            # 推理时，只使用one2one的query数量
            self.num_queries = self.num_queries_one2one
            self.transformer.two_stage_num_proposals = self.num_queries

        # original features
        features = self.backbone(images.tensor)  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in deformable DETR
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # initialize object query embeddings
        query_embeds = None
        if not self.as_two_stage or self.mixed_selection:
            query_embeds = self.query_embedding.weight[0: self.num_queries, :]


        # make attn mask
        """ attention mask to prevent information leakage
        """
        # [1800, 1800]
        self_attn_mask = (torch.zeros([self.num_queries, self.num_queries, ]).bool().to(feat.device))
        # 多的1500看不到前面的300
        self_attn_mask[self.num_queries_one2one:, 0: self.num_queries_one2one, ] = True
        # 300看不到后面的1500
        self_attn_mask[0: self.num_queries_one2one, self.num_queries_one2one:, ] = True

        # 调用transformer
        (
            inter_states,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            # 多的参数
            self_attn_mask,
        )

        # todo
        # Calculate output coordinates and classes.
        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            # 输出值前面的是one2one
            outputs_classes_one2one.append(outputs_class[:, 0: self.num_queries_one2one])
            # 输出值后面的是one2many
            outputs_classes_one2many.append(outputs_class[:, self.num_queries_one2one:])
            # 输出值前面的是one2one
            outputs_coords_one2one.append(outputs_coord[:, 0: self.num_queries_one2one])
            # 输出值后面的是one2many
            outputs_coords_one2many.append(outputs_coord[:, self.num_queries_one2one:])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        # tensor shape: [num_decoder_layers, bs, num_queries_one2one, num_classes]
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        # tensor shape: [num_decoder_layers, bs, num_queries_one2one, 4]
        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        # tensor shape: [num_decoder_layers, bs, num_queries_one2many, num_classes]
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)
        # tensor shape: [num_decoder_layers, bs, num_queries_one2many, 4]

        # prepare for loss computation
        output = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            # one2many
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
        }

        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(
                outputs_classes_one2one, outputs_coords_one2one
            )
            output["aux_outputs_one2many"] = self._set_aux_loss(
                outputs_classes_one2many, outputs_coords_one2many
            )

        if self.as_two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            # two_stage时，也需要对encoder的输出进行loss计算
            output["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            if self.k_one2many > 0:
                loss_dict = self.train_hybrid(
                    output,
                    targets,
                    self.k_one2many,
                    self.criterion,
                    # 计算loss时 one2many的系数
                    self.lambda_one2many,
                )
            else:
                loss_dict = self.criterion(output, targets)

            weight_dict = self.criterion.weight_dict
            new_dict = dict()
            for key, value in weight_dict.items():
                new_dict[key] = value
                # 权重dict上补充上one2many的权重
                new_dict[key + "_one2many"] = value
            weight_dict = new_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            # recover the model parameters for next training epoch
            self.num_queries = save_num_queries
            self.transformer.two_stage_num_proposals = save_two_stage_num_proposals
            return processed_results

    # 新增的方法
    def train_hybrid(self, outputs, targets, k_one2many, criterion, lambda_one2many):
        """
        k_one2many GT的组数
        lambda_one2many one2many的loss的系数
        """
        # one-to-one-loss
        loss_dict = criterion(outputs, targets)

        multi_targets = copy.deepcopy(targets)
        # for in bs
        # repeat the targets
        for target in multi_targets:
            # 重复k次
            target["boxes"] = target["boxes"].repeat(k_one2many, 1)
            # 重复k次
            target["labels"] = target["labels"].repeat(k_one2many)

        # 取出one2many相关的输出
        outputs_one2many = dict()
        outputs_one2many["pred_logits"] = outputs["pred_logits_one2many"]
        outputs_one2many["pred_boxes"] = outputs["pred_boxes_one2many"]
        outputs_one2many["aux_outputs"] = outputs["aux_outputs_one2many"]

        # one-to-many loss
        loss_dict_one2many = criterion(outputs_one2many, multi_targets)
        # 更新到loss_dict中
        for key, value in loss_dict_one2many.items():
            if key + "_one2many" in loss_dict.keys():
                loss_dict[key + "_one2many"] += value * lambda_one2many
            else:
                loss_dict[key + "_one2many"] = value * lambda_one2many
        return loss_dict

    # 无修改
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    # 无修改
    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # Select top-k confidence boxes for inference
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        for (
                i,
                (scores_per_image, labels_per_image, box_pred_per_image, image_size),
        ) in enumerate(zip(scores, labels, boxes, image_sizes)):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    # 无修改
    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    # 无修改
    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

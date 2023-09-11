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

import torch
import torch.nn as nn

from detrex.utils import inverse_sigmoid


def apply_label_noise(
        labels: torch.Tensor,
        label_noise_prob: float = 0.2,
        num_classes: int = 80,
):
    """
    Args:
        labels (torch.Tensor): Classification labels with ``(num_labels, )``.
        label_noise_prob (float): The probability of the label being noised. Default: 0.2.
        num_classes (int): Number of total categories.

    Returns:
        torch.Tensor: The noised labels the same shape as ``labels``.
    """
    if label_noise_prob > 0:
        p = torch.rand_like(labels.float())
        noised_index = torch.nonzero(p < label_noise_prob).view(-1)
        new_lebels = torch.randint_like(noised_index, 0, num_classes)  # 随机的label
        noised_labels = labels.scatter_(0, noised_index, new_lebels)  # 放入到指定的位置
        return noised_labels
    else:
        return labels


def apply_box_noise(
        boxes: torch.Tensor,
        box_noise_scale: float = 0.4,
):
    """
    Args:
        boxes (torch.Tensor): Bounding boxes in format ``(x_c, y_c, w, h)`` with
            shape ``(num_boxes, 4)``
        box_noise_scale (float): Scaling factor for box noising. Default: 0.4.
    """
    if box_noise_scale > 0:
        diff = torch.zeros_like(boxes)
        # 宽高的一半放入头两个 xy位置 这里是diff矩阵，计算出来的是偏移量
        diff[:, :2] = boxes[:, 2:] / 2
        # 宽高还是宽高
        diff[:, 2:] = boxes[:, 2:]
        # torch.rand_like * 2 - 1 计算出来的值在 [-1,1] 内
        # 最后 *box_noise_scale 限制下范围 如限制在 [-0.4, 0.4]
        # boxes += 将计算出来的偏移量与原始坐标相加
        boxes += torch.mul((torch.rand_like(boxes) * 2 - 1.0), diff) * box_noise_scale
        boxes = boxes.clamp(min=0.0, max=1.0)
    return boxes


class GenerateDNQueries(nn.Module):
    """Generate denoising queries for DN-DETR

    Args:
        num_queries (int): Number of total queries in DN-DETR. Default: 300
        num_classes (int): Number of total categories. Default: 80.
        label_embed_dim (int): The embedding dimension for label encoding. Default: 256.
        denoising_groups (int): Number of noised ground truth groups. Default: 5.
        label_noise_prob (float): The probability of the label being noised. Default: 0.2.
        box_noise_scale (float): Scaling factor for box noising. Default: 0.4
        with_indicator (bool): If True, add indicator in noised label/box queries.

    """

    def __init__(
            self,
            num_queries: int = 300,
            num_classes: int = 80,
            label_embed_dim: int = 256,
            denoising_groups: int = 5,
            label_noise_prob: float = 0.2,
            box_noise_scale: float = 0.4,
            with_indicator: bool = False,
    ):
        super(GenerateDNQueries, self).__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.denoising_groups = denoising_groups
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale
        self.with_indicator = with_indicator

        # leave one dim for indicator mentioned in DN-DETR
        if with_indicator:
            # 这里少一位，缺少的一位为了标识是不是去噪的
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim - 1)
        else:
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim)

    def generate_query_masks(self, max_gt_num_per_image, device):
        noised_query_nums = max_gt_num_per_image * self.denoising_groups
        tgt_size = noised_query_nums + self.num_queries # 总的数量，300个query和每个image的数量（这个是max的）
        attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0
        # match query cannot see the reconstruct
        attn_mask[noised_query_nums:, :noised_query_nums] = True
        for i in range(self.denoising_groups):
            if i == 0:
                # 第一组
                # 看不到他后面的所有
                attn_mask[
                max_gt_num_per_image * i: max_gt_num_per_image * (i + 1),
                max_gt_num_per_image * (i + 1): noised_query_nums,
                ] = True
            if i == self.denoising_groups - 1:
                # 最后一组
                # 看不到他前面的所有
                attn_mask[
                max_gt_num_per_image * i: max_gt_num_per_image * (i + 1),
                : max_gt_num_per_image * i,
                ] = True
            else:
                # 中间的其他的组
                # 看不到它后面的group
                attn_mask[
                max_gt_num_per_image * i: max_gt_num_per_image * (i + 1),
                max_gt_num_per_image * (i + 1): noised_query_nums,
                ] = True
                # 看不到它前面的group
                attn_mask[
                max_gt_num_per_image * i: max_gt_num_per_image * (i + 1),
                : max_gt_num_per_image * i,
                ] = True
        return attn_mask

    def forward(
            self,
            gt_labels_list,
            gt_boxes_list,
    ):
        """
        Args:
            gt_boxes_list (list[torch.Tensor]): Ground truth bounding boxes per image
                with normalized coordinates in format ``(x, y, w, h)`` in shape ``(num_gts, 4)``
            gt_labels_list (list[torch.Tensor]): Classification labels per image in shape ``(num_gt, )``.
        """

        # concat ground truth labels and boxes in one batch
        # e.g. [tensor([0, 1, 2]), tensor([2, 3, 4])] -> tensor([0, 1, 2, 2, 3, 4])
        # bs中所有的gt label box cat到一起
        gt_labels = torch.cat(gt_labels_list)
        gt_boxes = torch.cat(gt_boxes_list)

        # For efficient denoising, repeat the original ground truth labels and boxes to
        # create more training denoising samples.
        # e.g. tensor([0, 1, 2, 2, 3, 4]) -> tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4]) if group = 2.
        # 复制多组
        gt_labels = gt_labels.repeat(self.denoising_groups, 1).flatten()
        gt_boxes = gt_boxes.repeat(self.denoising_groups, 1)

        # set the device as "gt_labels"
        device = gt_labels.device
        assert len(gt_labels_list) == len(gt_boxes_list)

        batch_size = len(gt_labels_list)

        # the number of ground truth per image in one batch
        # e.g. [tensor([0, 1]), tensor([2, 3, 4])] -> gt_nums_per_image: [2, 3]
        # means there are 2 instances in the first image and 3 instances in the second image
        gt_nums_per_image = [x.numel() for x in gt_labels_list]

        # Add noise on labels and boxes
        noised_labels = apply_label_noise(gt_labels, self.label_noise_prob, self.num_classes)

        noised_boxes = apply_box_noise(gt_boxes, self.box_noise_scale)
        # 转换回特征图的坐标范围
        noised_boxes = inverse_sigmoid(noised_boxes)

        # encoding labels
        label_embedding = self.label_encoder(noised_labels)
        query_num = label_embedding.shape[0]

        # add indicator to label encoding if with_indicator == True
        if self.with_indicator:
            # 噪声的label，最后一位添加的1
            label_embedding = torch.cat([label_embedding, torch.ones([query_num, 1]).to(device)], 1)

        # calculate the max number of ground truth in one image inside the batch.
        # e.g. gt_nums_per_image = [2, 3] which means
        # the first image has 2 instances and the second image has 3 instances
        # then the max_gt_num_per_image should be 3.
        max_gt_num_per_image = max(gt_nums_per_image)

        # the total denoising queries is depended on denoising groups and max number of instances.
        # 一个图像的query数量，当然以最大的数量计算
        noised_query_nums = max_gt_num_per_image * self.denoising_groups

        # [bs,max*group,256]
        # initialize the generated noised queries to zero.
        # And the zero initialized queries will be assigned with noised embeddings later.
        noised_label_queries = (
            torch.zeros(noised_query_nums, self.label_embed_dim).to(device).repeat(batch_size, 1, 1)
        )
        noised_box_queries = torch.zeros(noised_query_nums, 4).to(device).repeat(batch_size, 1, 1)

        # batch index per image: [0, 1, 2, 3] for batch_size == 4
        batch_idx = torch.arange(0, batch_size)
        # 每个gt的batch id
        # e.g. gt_nums_per_image = [2, 3]
        # batch_idx = [0, 1]
        # then the "batch_idx_per_instance" equals to [0, 0, 1, 1, 1]
        # which indicates which image the instance belongs to.
        # cuz the instances has been flattened before.
        batch_idx_per_instance = torch.repeat_interleave(
            batch_idx, torch.tensor(gt_nums_per_image).long()
        )

        # indicate which image the noised labels belong to. For example:
        # noised label: tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4])
        # batch_idx_per_group: tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        # which means the first label "tensor([0])"" belongs to "image_0".
        # 整合了batch id，整合了group，每组的batch id
        batch_idx_per_group = batch_idx_per_instance.repeat(self.denoising_groups, 1).flatten()

        # Cuz there might be different numbers of ground truth in each image of the same batch.
        # So there might be some padding part in noising queries.
        # Here we calculate the indexes for the valid queries and
        # fill them with the noised embeddings.
        # And leave the padding part to zeros.
        if len(gt_nums_per_image):
            valid_index_per_group = torch.cat(
                [torch.tensor(list(range(num))) for num in gt_nums_per_image]
            ) # gt数量的 range 如 0--xx,0--yy,0-zz
            valid_index_per_group = torch.cat(
                [
                    valid_index_per_group + max_gt_num_per_image * i
                    for i in range(self.denoising_groups)
                ]
            ).long() # 加上偏移量，每个组其偏移量是最大的gt数量的倍数

        if len(batch_idx_per_group):
            # 第一个维度选取batch id，第二个维度在指定的位置上放置噪声
            # 其余的位置保持0
            noised_label_queries[(batch_idx_per_group, valid_index_per_group)] = label_embedding
            noised_box_queries[(batch_idx_per_group, valid_index_per_group)] = noised_boxes

        # generate attention masks for transformer layers
        attn_mask = self.generate_query_masks(max_gt_num_per_image, device)

        return (
            noised_label_queries,
            noised_box_queries,
            attn_mask,
            self.denoising_groups,
            # bs中最大的gt的数量
            max_gt_num_per_image,
        )


class GenerateCDNQueries(nn.Module):
    def __init__(
            self,
            num_queries: int = 300,
            num_classes: int = 80,
            label_embed_dim: int = 256,
            denoising_nums: int = 100,
            label_noise_prob: float = 0.5,
            box_noise_scale: float = 1.0,
    ):
        super(GenerateCDNQueries, self).__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.denoising_nums = denoising_nums
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale

        self.label_encoder = nn.Embedding(num_classes, label_embed_dim)

    def forward(
            self,
            gt_labels_list,
            gt_boxes_list,
    ):
        denoising_nums = self.denoising_nums * 2

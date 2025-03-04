# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from ...common import LayerNorm2d

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        该类用于预测给定图像和提示（prompt）嵌入后的掩码（mask），
        采用 Transformer 结构进行特征转换和掩码预测。

        参数:
          transformer_dim (int): Transformer 的通道维度
          transformer (nn.Module): 负责掩码预测的 Transformer 模块
          num_multimask_outputs (int): 当有歧义时，模型可以预测多个掩码，此参数控制输出掩码数量
          activation (nn.Module): 掩码上采样时使用的激活函数，默认使用 GELU
          iou_head_depth (int): 用于预测掩码质量的 MLP 深度
          iou_head_hidden_dim (int): 用于预测掩码质量的 MLP 隐藏层维度
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        # IoU 评分的嵌入向量
        self.iou_token = nn.Embedding(1, transformer_dim)# 对比的iou评分为1
        # 计算掩码令牌的数量，至少为 4，用于兼容旧版本的模型
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = max(4, num_multimask_outputs)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 掩码的上采样模块，由两层反卷积（ConvTranspose2d）和激活函数组成
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # 掩码预测的超网络 MLP，每个掩码令牌对应一个 MLP
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )# M(e_prompt)	用 MLP 生成掩码权重

        # IoU 预测头部（用于预测掩码质量）
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测掩码mask,输入图像和提示嵌入，返回掩码和质量评分。

        参数:
          image_embeddings: 图像的特征嵌入。
          image_pe: 位置编码。
          sparse_prompt_embeddings: 稀疏提示（如点、框）。
          dense_prompt_embeddings: 稠密提示（如掩码）。
          multimask_output: 是否输出多个掩码
        返回:
          masks (torch.Tensor): 批量预测的掩码张量
          iou_pred (torch.Tensor): 批量预测的掩码质量评分
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # 选择正确的掩码数量进行输出
        mask_slice = slice(0, self.num_multimask_outputs)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 
        预测掩码的核心方法，在 `forward` 方法中被调用。

        参数:
          image_embeddings (torch.Tensor): 图像的特征嵌入
          image_pe (torch.Tensor): 位置编码
          sparse_prompt_embeddings (torch.Tensor): 稀疏提示嵌入
          dense_prompt_embeddings (torch.Tensor): 稠密提示嵌入

        返回:
          masks (torch.Tensor): 预测的掩码
          iou_pred (torch.Tensor): 掩码质量的预测分数
        """
        # 连接输出令牌（IoU 令牌 + 掩码令牌）
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)# 拼接 IoU 评分 token 和掩码 token
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) # 扩展 batch 维度
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 让 image_embeddings 适配 tokens 的 batch 维度
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings  # 添加稠密提示信息
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)# 让 位置编码image_pe 适配 tokens 的 batch 维度
        b, c, h, w = src.shape

        # 通过 Transformer 处理数据
        hs, src = self.transformer(src, pos_src, tokens)#输出 hs：经过 Transformer 处理后的 token 结果。输出 src：处理后的图像嵌入，通常形状不变 (B_tokens, C, H, W)
        iou_token_out = hs[:, 0, :] # hs形状（B,N,D）,N为token数量：IoU ，mask掩码，prompt提示
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # 上采样掩码嵌入，并使用掩码令牌预测掩码
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))# mask_tokens_out[:, i, :] 形状为 (B, D)，是 Transformer 计算出的第 i 个掩码 token。经过 MLP 处理后，输出形状 (B, C')
        hyper_in = torch.stack(hyper_in_list, dim=1)# 把多个 mask tokens 对应的权重 hyper_in_list 堆叠，得到形状 (B, num_mask_tokens, C')

        # 计算掩码
        b, c, h, w = upscaled_embedding.shape
        
        # e_down^(n+1) = ReLU(Norm(e_down^n ⊗ w_prompt))	利用权重调整特征嵌入
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w) 
        

        # 生成掩码质量评分
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MLP(nn.Module):
    """
    多层感知机MLP用于掩码质量评分的预测和掩码嵌入转换。

    参数:
        input_dim (int): 输入维度
        hidden_dim (int): 隐藏层维度
        output_dim (int): 输出维度
        num_layers (int): MLP 的层数
        sigmoid_output (bool): 是否在输出层应用 sigmoid 激活函数
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1) # 隐藏层维度*隐藏层数量
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x

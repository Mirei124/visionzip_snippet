from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.mm_utils import get_anyres_image_grid_shape
from transformers.generation.utils import GenerateOutput
from transformers.models.clip.modeling_clip import CLIPVisionModel

from .language_model.llava_llama import LlavaLlamaForCausalLM


def prepare_inputs_labels_for_multimodal_visionzip(
    self,
    input_ids,
    position_ids,
    attention_mask,
    past_key_values,
    labels,
    images,
    image_sizes=None,
    reserved_token_num=None,
):
    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
        concat_images = torch.cat([image for image in images], dim=0)
        image_features, keep_idxs = self.encode_images_visionzip_multi(concat_images, reserved_token_num)
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)
        keep_idxs = torch.split(keep_idxs, split_sizes, dim=0)
        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        _image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        if mm_patch_merge_type == "flat":
            image_features = [x.flatten(0, 1) for x in image_features]
        elif mm_patch_merge_type.startswith("spatial"):
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    cur_keep_idx = keep_idxs[image_idx[1:]]
                    num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                        image_sizes[image_idx],
                        self.config.image_grid_pinpoints,
                        self.get_vision_tower().config.image_size,
                    )
                    if "unpad" in mm_patch_merge_type:
                        image_feature = self.restore_image_features_sorted(
                            image_feature, cur_keep_idx, num_patch_height, num_patch_height
                        )
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                else:
                    image_feature = image_feature[0]
                    if "unpad" in mm_patch_merge_type:
                        image_feature = torch.cat(
                            (image_feature, self.model.image_newline[None].to(image_feature.device)), dim=0
                        )
                new_image_features.append(image_feature)
            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    else:
        image_features = self.encode_images_visionzip(images, reserved_token_num)

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
        raise NotImplementedError

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- FIXME
    _input_ids = input_ids
    input_ids = [
        cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
    ]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = (
            [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        )
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(
                    torch.full(
                        (cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype
                    )
                )

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full(
        (batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, "tokenizer_padding_side", "right") == "left":
            new_input_embeds_padded.append(
                torch.cat(
                    (
                        torch.zeros(
                            (max_len - cur_len, cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                        cur_new_embed,
                    ),
                    dim=0,
                )
            )
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(
                    0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                )
        else:
            new_input_embeds_padded.append(
                torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_len, cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
            )
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(
                    0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                )

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


def restore_image_features_sorted(self, image_feature, cur_keep_idx, width, height):
    num_img, total_patches, feature_dim = image_feature.shape
    num_keep = cur_keep_idx.shape[1]
    num_extra = total_patches - num_keep

    cur_keep_idx_sorted, _ = cur_keep_idx.sort(dim=1)  # [num_img, num_keep]
    cur_keep_idx_sorted_restore = cur_keep_idx_sorted[:, 1:] - 1

    restored_features = torch.zeros(
        (num_img, 576, feature_dim), device=image_feature.device, dtype=image_feature.dtype
    )  # [num_img, total_patches, feature_dim]

    mask = torch.zeros(num_img, 576, dtype=torch.bool, device=image_feature.device)
    mask.scatter_(1, cur_keep_idx_sorted_restore, True)

    kept_features = image_feature[:, 1:num_keep, :]
    restored_features[mask] = kept_features.reshape(-1, feature_dim)

    assert width * height == restored_features.shape[0], "width * height must equal num_img"
    restored_features = restored_features.view(
        height, width, 24, 24, feature_dim
    )  # [height, width, 24, 24, feature_dim]
    restored_features = restored_features.permute(0, 2, 1, 3, 4).contiguous()  # [height, 24, width, 24, feature_dim]
    restored_features = restored_features.view(
        height, 24, width * 24, feature_dim
    )  # [height, 24, width*24, feature_dim]
    restored_features = restored_features.view(
        height * 24, width * 24, feature_dim
    )  # [height*24, width*24, feature_dim]
    _image_newline_expanded = (
        self.model.image_newline.view(1, 1, feature_dim)
        .expand(height * 24, 1, feature_dim)
        .to(restored_features.device)
    )  # [height*24, 1, feature_dim]
    grid_with_newline = restored_features

    mask = mask.view(height, width, 24, 24)  # [height, width, 24, 24]
    mask = mask.permute(0, 2, 1, 3).contiguous()  # [height, 24, width, 24]
    mask = mask.view(height * 24, width * 24)  # [height*24, width*24]

    mask_all = mask

    image_feature_select = grid_with_newline[mask_all]
    raw_img_feature_merge = image_feature[
        :,
        -num_extra:,
    ].reshape(-1, feature_dim)
    cls_img_feature_merge = image_feature[
        :,
        0,
    ]

    image_feature_select = torch.cat([image_feature_select, cls_img_feature_merge, raw_img_feature_merge])
    return image_feature_select


def CLIPVisionTower_forward(self, images, **kwargs):
    x = images
    image_forward_outs = self.vision_tower(x, output_hidden_states=True, output_attentions=True)
    attn_weights = image_forward_outs.attentions[-2]
    hidden_states = image_forward_outs.hidden_states[-2]
    if "dominant" not in kwargs.keys():
        raise ValueError("dominant not found")
        return hidden_states[:, 1:]

    metric = self.vision_tower.vision_model.encoder.layers[-2].metric
    dominant_num = kwargs.pop("dominant")
    contextual_num = kwargs.pop("contextual")

    # return cls token when reserved_token_num == 1
    if dominant_num + contextual_num == 0:
        return hidden_states[:, :1].to(x.dtype), torch.zeros(
            (hidden_states.shape[0], 1), dtype=torch.int64, device=x.device
        )

    ## Dominant Visual Tokens
    cls_idx = 0
    cls_attention = attn_weights[:, :, cls_idx, cls_idx + 1 :]
    cls_attention_sum = cls_attention.sum(dim=1)
    topk_indices = cls_attention_sum.topk(dominant_num, dim=1).indices + 1
    all_indices = torch.cat(
        [torch.zeros((hidden_states.shape[0], 1), dtype=topk_indices.dtype, device=topk_indices.device), topk_indices],
        dim=1,
    )

    mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(
        1, all_indices, False
    )
    dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(
        hidden_states.shape[0], dominant_num + 1, hidden_states.shape[2]
    )

    ### Filter
    metric_filtered = metric[mask].view(
        hidden_states.shape[0], hidden_states.shape[1] - (dominant_num + 1), metric.shape[2]
    )

    hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(
        hidden_states.shape[0], hidden_states.shape[1] - (dominant_num + 1), hidden_states.shape[2]
    )

    metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True)

    ## Contextual Visual Tokens
    step = max(1, metric_normalized.shape[1] // contextual_num)
    target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
    target_tokens = metric_normalized[:, target_indices, :]

    tokens_to_merge = metric_normalized[
        :, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :
    ]
    similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
    assign_one_hot = torch.zeros(
        tokens_to_merge.shape[0],
        tokens_to_merge.shape[1],
        contextual_num,
        dtype=hidden_states_filtered.dtype,
        device=metric_normalized.device,
    )
    assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
    hidden_to_merge = hidden_states_filtered[
        :,
        ~torch.isin(
            torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices
        ),
        :,
    ]
    aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
    target_hidden = hidden_states_filtered[:, target_indices, :]

    contextual_tokens = target_hidden + aggregated_hidden

    # Merge with target hidden states and concatenate
    hidden_states_save = torch.cat([dominant_tokens, contextual_tokens], dim=1).to(x.dtype)

    return hidden_states_save, all_indices


def CLIPEncoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_attention_mask: torch.Tensor,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.FloatTensor]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            `(config.encoder_attention_heads,)`.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
    """
    residual = hidden_states

    hidden_states = self.layer_norm1(hidden_states)
    hidden_states, attn_weights, metric = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
    )
    hidden_states = residual + hidden_states

    if getattr(self, "save_metric", False):
        self.metric = metric

    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def CLIPAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""

    bsz, tgt_len, embed_dim = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states) * self.scale
    key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    raw_key_states = key_states.clone().detach()
    value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    # apply the causal_attention_mask first
    if causal_attention_mask is not None:
        if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {causal_attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if output_attentions:
        # this operation is a bit akward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights_reshaped, raw_key_states.mean(1)


def encode_images_visionzip(self, images: torch.Tensor, reserved_token_num: int):
    contextual = reserved_token_num // 7
    dominant = reserved_token_num - contextual - 1

    image_features, keep_idx = (
        self.get_model().get_vision_tower().forward(images, dominant=dominant, contextual=contextual)
    )
    # print(f'{image_features.shape=}')
    image_features = self.get_model().mm_projector(image_features)
    return image_features


def encode_images_visionzip_multi(self, images: torch.Tensor, reserved_token_num: int):
    contextual = reserved_token_num // 7
    dominant = reserved_token_num - contextual - 1

    image_features, keep_idx = (
        self.get_model().get_vision_tower().forward(images, dominant=dominant, contextual=contextual)
    )
    # print(f'{image_features.shape=}')
    image_features = self.get_model().mm_projector(image_features)
    return image_features, keep_idx


@torch.no_grad()
def generate(
    self,
    inputs: Optional[torch.Tensor] = None,
    images: Optional[torch.Tensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    position_ids = kwargs.pop("position_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    reserved_token_num = kwargs.pop("reserved_token_num", None)
    if "inputs_embeds" in kwargs:
        raise NotImplementedError("`inputs_embeds` is not supported")

    if images is not None:
        (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
            inputs,
            position_ids,
            attention_mask,
            None,
            None,
            images,
            image_sizes=image_sizes,
            reserved_token_num=reserved_token_num,
        )
    else:
        inputs_embeds = self.get_model().embed_tokens(inputs)

    return super().generate(
        position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
    )


def make_to_me_class(vision_model_class):
    class VisionZipCLIPVisionModel(vision_model_class):
        def forward(self, *args, **kwargs):
            if hasattr(self.vision_model.encoder.layers[-2], "metric"):
                self.vision_model.encoder.layers[-2].metric = None
            return super().forward(*args, **kwargs)

    return VisionZipCLIPVisionModel


def apply_info(model: "CLIPVisionModel"):
    VisionZipCLIPVisionModel = make_to_me_class(model.__class__)
    model.__class__ = VisionZipCLIPVisionModel

    model.vision_model.encoder.layers[-2].save_metric = True


def apply_vision_zip(model: "LlavaLlamaForCausalLM"):
    apply_info(model.model.vision_tower.vision_tower)

    from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer

    CLIPEncoderLayer.forward = CLIPEncoderLayer_forward
    CLIPAttention.forward = CLIPAttention_forward

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

    CLIPVisionTower.forward = CLIPVisionTower_forward

    from llava.model.llava_arch import LlavaMetaForCausalLM

    if hasattr(LlavaMetaForCausalLM, "prepare_inputs_labels_for_multimodal"):
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_visionzip
        LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted
        LlavaMetaForCausalLM.encode_images_visionzip_multi = encode_images_visionzip_multi
        LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip
        LlavaMetaForCausalLM.generate = generate

    return model

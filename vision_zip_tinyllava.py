from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.generation.utils import GenerateOutput
from transformers.models.clip import CLIPVisionModel

from ..utils.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from .modeling_tinyllava import TinyLlavaForConditionalGeneration


def TinyLlavaForConditionalGeneration_encode_images_visionzip(self, images: torch.Tensor, reserved_token_num: int):
    contextual = reserved_token_num // 7
    dominant = reserved_token_num - contextual - 1

    images = images.to(device=self.device, dtype=self.dtype)
    if self.training:
        images.requires_grad_(True)
    image_features, keep_idx = self.vision_tower(images, dominant=dominant, contextual=contextual)
    # print(f'{image_features.shape=}')
    image_features = self.connector(image_features)
    return image_features


def TinyLlavaForConditionalGeneration_prepare_inputs_labels_for_multimodal_visionzip(
    self,
    input_ids,
    position_ids,
    attention_mask,
    past_key_values,
    labels,
    images,
    image_sizes=None,
    reserved_token_num: Optional[int] = None,
):
    vision_tower = self.vision_tower
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    image_features = self.encode_images_visionzip(images, reserved_token_num)

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, "tune_mm_mlp_adapter", False):
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
            cur_input_embeds_1 = self.language_model.get_input_embeddings()(cur_input_ids)
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
        cur_input_embeds = self.language_model.get_input_embeddings()(torch.cat(cur_input_ids_noim))
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


def CLIPVisionTower_forward(self, x, **kwargs):
    image_forward_outs = self._vision_tower(x, output_hidden_states=True, output_attentions=True)
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


@torch.no_grad()
def TinyLlavaForConditionalGeneration_generate(
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
        inputs_embeds = self.language_model.get_input_embeddings()(inputs)

    return self.language_model.generate(
        position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs
    )


def make_to_me_class(vision_model_class):
    class VisionZipCLIPVisionModel(vision_model_class):
        def forward(self, *args, **kwargs):
            if hasattr(self.vision_model.encoder.layers[-2], "metric"):
                self.vision_model.encoder.layers[-2].metric = None
            return super().forward(*args, **kwargs)

    return VisionZipCLIPVisionModel


def apply_info(model: CLIPVisionModel):
    VisionZipCLIPVisionModel = make_to_me_class(model.__class__)
    model.__class__ = VisionZipCLIPVisionModel

    model.vision_model.encoder.layers[-2].save_metric = True


def apply_vision_zip(model: TinyLlavaForConditionalGeneration):
    """Apply VisionZip.

    Args:
        model: TinyLlavaForConditionalGeneration

    """
    apply_info(model.vision_tower._vision_tower)

    from transformers.models.clip.modeling_clip import CLIPAttention, CLIPEncoderLayer

    CLIPEncoderLayer.forward = CLIPEncoderLayer_forward
    CLIPAttention.forward = CLIPAttention_forward

    from .vision_tower.clip import CLIPVisionTower

    CLIPVisionTower.forward = CLIPVisionTower_forward

    TinyLlavaForConditionalGeneration.encode_images_visionzip = (
        TinyLlavaForConditionalGeneration_encode_images_visionzip
    )
    TinyLlavaForConditionalGeneration.prepare_inputs_labels_for_multimodal = (
        TinyLlavaForConditionalGeneration_prepare_inputs_labels_for_multimodal_visionzip
    )
    TinyLlavaForConditionalGeneration.generate = TinyLlavaForConditionalGeneration_generate

    return model

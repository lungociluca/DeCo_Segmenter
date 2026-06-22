import os
from enum import Enum
from typing import List


class CrossAttnType(Enum):
    Q_K = 'q-k'
    K_K = 'k-k'
    V_V = 'v-v'

class AttentionAggregateMethod(Enum):
    NO_OP = "no-op"
    MERGE_WITH_SELF_ATTN = "merge_with_self_attn"

class HeadAggregateMethod(Enum):
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"

class AttentionMapsProjection(Enum):
    NO_OP = "no-op"
    ORTHOGONAL_TO_PRINCIPAL_COMP = "orthogonal_to_principal_components"

device = "mps"

out_image_path = "out.jpg"
num_steps = 5
guidance = 3.0
image_height = 512
image_width = 512
num_images = 1
label = "a picture of a dog"
neg_label = "a picture of a background"
seed = 12
timeshift = 1
order = 2

attention_maps_dir = "attn_maps"
save_maps = True
eval = True
eval_samples_limit = 1
idx_token_of_interest = 3

dit_blocks = 1
background_threshold = 0.0004
cross_attention_types: List[CrossAttnType] = [
    CrossAttnType.Q_K,
]
attention_aggregate_method: AttentionAggregateMethod = AttentionAggregateMethod.NO_OP
head_aggregate_method: HeadAggregateMethod = HeadAggregateMethod.MEAN
attention_map_projection: AttentionMapsProjection = AttentionMapsProjection.NO_OP
import os
from enum import Enum
from typing import List

class EvalDatasets(Enum):
    ADE150 = "ade150"
    VOC12 = "voc12"

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

class MapsWeighting(Enum):
    NO_OP = "no-op"
    POOLING_OVER_COS_SIMILARITY = "pool_cos_similiarity"

class ForwardMethod(Enum):
    ATTENTION = "attention"
    COSINE_SIMILARITY = "cosine_similarity"

device = "cuda"

datasets = {
    EvalDatasets.ADE150: {
        "json": "catseg_configs/ade150.json",
        "img_dir": "images",
        "gt_dir": "annotations",
        "extention": ".png"
    },
    EvalDatasets.VOC12: {
        "json": "catseg_configs/voc20.json",
        "img_dir": "JPEGImages",
        "gt_dir": "SegmentationClassAug",
        "extention": ".png"
    }
}

run_on_textures = True
eval_dataset = EvalDatasets.VOC12
out_image_path = "out.jpg"
num_steps = 20
guidance = 10.0
image_height = 512
image_width = 512
num_images = 1
label = "A realistic scene with dog, cat, flying aeroplane"
neg_label = "Unrealistic"

seed = 13
timeshift = 1
order = 2

attention_maps_dir = "attn_maps"
save_maps = True
eval = True

background_threshold = 0.3
unbiasing_components_count = 2
no_clusters = 8
#271 595 264
self_attn_softmax_temperature = 0.7
cross_attn_softmax_temperature = 2000
aggregated_maps_softmax_temperature = 0.7

cross_attention_types: List[CrossAttnType] = [
    CrossAttnType.Q_K,
]
attention_aggregate_method: AttentionAggregateMethod = AttentionAggregateMethod.NO_OP
head_aggregate_method: HeadAggregateMethod = HeadAggregateMethod.MEAN
attention_map_projection: AttentionMapsProjection = AttentionMapsProjection.NO_OP
maps_weighting: MapsWeighting = MapsWeighting.NO_OP
forward_method: ForwardMethod = ForwardMethod.ATTENTION



dit_blocks = 1
idx_token_of_interest = 1
eval_samples_limit = 500
use_gate = True

ts = 0

# Cluster params
eps = 1
min_samples = 16
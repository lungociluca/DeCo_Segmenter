import os
from enum import Enum
from typing import List


class CrossAttnType(Enum):
    Q_K = 'q-k'
    K_K = 'k-k'
    V_V = 'v-v'

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

dit_blocks = 1
cross_attention_types: List[CrossAttnType] = [
    CrossAttnType.Q_K,
]
background_threshold = 0.0004
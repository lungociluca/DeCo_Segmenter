export DETECTRON2_DATASETS="../../Downloads"; 
cp run_configs/config_agg_mean_heads.py config.py
sh eval.sh catseg_configs/config.yaml 1 output/  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON catseg_configs/ade150.json
mv attn_maps/* prev_attn_maps/agg_mean_heads
mv output/eval/log.txt prev_attn_maps/agg_mean_heads

cp run_configs/config_no_agg_mean_heads.py config.py
sh eval.sh catseg_configs/config.yaml 1 output/  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON catseg_configs/ade150.json
mv attn_maps/* prev_attn_maps/no_agg_mean_heads
mv output/eval/log.txt prev_attn_maps/no_agg_mean_heads

cp run_configs/config_no_agg_wei_heads.py config.py
sh eval.sh catseg_configs/config.yaml 1 output/  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON catseg_configs/ade150.json
mv attn_maps/* prev_attn_maps/no_agg_wei_heads
mv output/eval/log.txt prev_attn_maps/no_agg_wei_heads
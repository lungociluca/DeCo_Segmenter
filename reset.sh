rm -r attn_maps;
rm -f sio_maps/images/*;
rm -f sio_maps/samples/*;
rm -f z_output/*;
export DETECTRON2_DATASETS="../../Downloads"; 
sh eval.sh catseg_configs/config.yaml 1 output/  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON catseg_configs/ade150.json;

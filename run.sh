OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python render.py \
    -m benchmark_360v2_ours/bicycle \
    -r 1 \
    --data_device cpu \
    --skip_train
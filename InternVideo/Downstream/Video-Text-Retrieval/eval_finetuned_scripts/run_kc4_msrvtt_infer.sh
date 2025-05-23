CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 inference.py \
    --do_eval \
    --num_thread_reader=4 \
    --n_display=50 \
    --batch_size=256 \
    --train_csv="/data/gaohaowen/workspace/InternVideo-main/InternVideo1/Dataset/ZRandom_Video/msrvtt_train_new_only_random_gen/MSRVTT_train_gen.9k.csv" \
    --val_csv="/data/gaohaowen/workspace/InternVideo-main/InternVideo1/Dataset/ZRandom_Video/msrvtt_train_new_only_random_gen/MSRVTT_JSFUSION_test_gen.csv" \
    --data_path="/data/gaohaowen/workspace/InternVideo-main/InternVideo1/Dataset/ZRandom_Video/msrvtt_train_new_only_random_gen/MSRVTT_data_gen.json" \
    --lr=1e-3 \
    --max_words=77 \
    --max_frames=12 \
    --batch_size_val=16 \
    --datatype="msrvtt" \
    --expand_msrvtt_sentences \
    --feature_framerate=1 \
    --slice_framepos=2 \
    --linear_patch=2d \
    --sim_header meanP \
    --loose_type \
    --pretrained_clip_name="ViT-L/14" \
    --clip_evl \
    --pretrained_path="/data/gaohaowen/workspace/InternVideo-main/InternVideo1/ckpt/InternVideo-MM-L-14.ckpt" \
    --finetuned_path="/data/gaohaowen/workspace/InternVideo-main/InternVideo1/output/pytorch_model_epoch5_0.02_100_42.3.bin" \
    --features_path="/data/gaohaowen/workspace/InternVideo-main/InternVideo1/Dataset/ZRandom_Video/msrvtt_train_new_only_random_gen/videos" \
    --output_dir="./log/msrvtt_infer/inference" \
    --mergeclip=True \
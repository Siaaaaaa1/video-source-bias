# srun -p video -N1 -n1 --gres=gpu:1 --cpus-per-task=16 --quotatype=auto --job-name=zs_msrvtt \
CUDA_VISIBLE_DEVICES=3 python -u -m main_task_retrieval \
    --do_eval \
    --num_thread_reader=4 \
    --epochs=5 \
    --batch_size=8 \
    --n_display=50 \
    --train_csv="/data/gaohaowen/workspace/InternVideo-main/InternVideo1/Dataset/opensora_video_extending/MSRVTT_train.9k.csv" \
    --val_csv="/data/gaohaowen/workspace/InternVideo-main/InternVideo1/Dataset/opensora_video_extending/MSRVTT_JSFUSION_test.csv" \
    --data_path="/data/gaohaowen/workspace/InternVideo-main/InternVideo1/Dataset/opensora_video_extending/MSRVTT_data.json" \
    --lr=1e-4 \
    --max_words=77 \
    --max_frames=8 \
    --batch_size_val=16 \
    --datatype="msrvtt" \
    --expand_msrvtt_sentences  \
    --feature_framerate=1 \
    --coef_lr=1e-3 \
    --freeze_layer_num=-1 \
    --slice_framepos=2 \
    --linear_patch=2d \
    --sim_header=meanP \
    --loose_type \
    --pretrained_clip_name="ViT-L/14" \
    --clip_evl \
    --pretrained_path="/data/gaohaowen/workspace/InternVideo-main/InternVideo1/ckpt/InternVideo-MM-L-14.ckpt" \
    --output_dir="/data/gaohaowen/workspace/InternVideo-main/InternVideo1/output" \
    --mergeclip=True \
    # --features_path="/data/gaohaowen/workspace/InternVideo-main/InternVideo1/Dataset/opensora_video_extending/videos" \
cd ..

export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PYTHONPATH

STEP='best'

CONFIG_PATH='config_release/msrvtt_ret.json'

TXT_DB='/data/gaohaowen/workspace/InternVideo-main/InternVideo1/Dataset/msrvtt_2b_1000_summary_org'
IMG_DB='/data/gaohaowen/workspace/InternVideo-main/InternVideo1/Dataset/ZMixed_AI_Video/20%generate/videos'

horovodrun -np 4 python src/tasks/run_video_retrieval.py \
      --do_inference 1 \
      --inference_split test \
      --inference_model_step $STEP \
      --inference_txt_db $TXT_DB \
      --inference_img_db $IMG_DB \
      --inference_batch_size 256 \
      --output_dir  output/downstreams/msrvtt_ret/public \
      --config $CONFIG_PATH

# horovodrun -np 8 python src/tasks/run_video_retrieval.py \
#       --do_inference 1 \
#       --inference_split test \
#       --inference_model_step $STEP \
#       --inference_txt_db $TXT_DB \ 
#       --inference_img_db $IMG_DB \
#       --inference_batch_size 64 \
#       --output_dir  output/downstreams/msrvtt_ret/public \
#       --config $CONFIG_PATH
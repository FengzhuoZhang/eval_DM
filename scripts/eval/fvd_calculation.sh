CUDA_VISIBLE_DEVICE=1 python -m ipdb -c continue -m opensora.eval.eval_common_metric \
    --real_video_dir "/home/aiops/zhangfz/content-debiased-fvd/mira_video_10"\
    --generated_video_dir "/home/aiops/zhangfz/Memory/results/base_length_16_ds_size_500_batch_size_8_lr_1e-06_uncond_prob_0.1_n_gpu_8_prompt_type_full/0" \
    --batch_size 5 \
    --num_frames 16 \
    --resolution 512\
    --crop_size 320 \
    --device 'cuda' \
    --metric 'fvd' \
    --fvd_method 'videogpt'
CUDA_VISIBLE_DEVICES="0,1,2" torchrun --master_port 12350 --nproc_per_node 3 av2_py/evaluation_ddp_refine-av2-multiagent.py \
  --features_dir data_av2_refine/p1_fmae_av2_final/ \
  --train_batch_size 32 \
  --val_batch_size 32 \
  --use_cuda \
  --logger_writer \
  --file_name $(basename $0) \
  --embed_dim 64 \
  --refine_num 3 \
  --seg_num 4 \
  --local_radius 10 \
  --model_path saved_ckpts/SRefiner_fmae.tar


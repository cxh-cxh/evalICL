# python icl_batch_new.py -n pi0_t10_5mm -c 10 --policy pi0_t10 --task t10 --vlm qwen3-vl-plus --queries_num 200 \
#     --envs t100000_5mm_env1 t100001_5mm_env1 t100002_5mm_env1 t100003_5mm_env1 --train_envs t100000_5mm_train t100001_5mm_train t100002_5mm_train t100003_5mm_train t100000_2mm_train t100001_2mm_train t100002_2mm_train t100003_2mm_train

# python icl_batch_new.py -n pi0_t10_2mm -c 10 --policy pi0_t10 --task t10 --vlm qwen3-vl-plus --queries_num 199 \
#     --envs t100000_2mm_env1 t100001_2mm_env1 t100002_2mm_env1 t100003_2mm_env1 --train_envs t100000_5mm_train t100001_5mm_train t100002_5mm_train t100003_5mm_train t100000_2mm_train t100001_2mm_train t100002_2mm_train t100003_2mm_train


# python icl_batch.py -n pi0_t7_135 -c 10 --policy pi0_t7 --task t7 --vlm qwen3-vl-plus --queries_num 150 \
#     --envs t70004_env1  t70002_env1  t70000_env1 --train_envs t70000_train t70001_train t70002_train t70003_train t70004_train t70005_train

# python icl_batch.py -n pi0_t7_246 -c 10 --policy pi0_t7 --task t7 --vlm qwen3-vl-plus --queries_num 150 \
#     --envs t70005_env1  t70003_env1  t70001_env1 --train_envs t70000_train t70001_train t70002_train t70003_train t70004_train t70005_train

python video_icl_batch.py -n pi0_drawer_full -c 4 --policy pi0_drawer --task drawer --vlm qwen3-vl-plus --queries_num 200 \
    --envs drawer_env1 



#  --envs t70005_env1 t70004_env1 t70002_env1 t70003_env1 t70001_env1 t70000_env1 --train_envs t70000_train t70001_train t70002_train t70003_train t70004_train t70005_train
 
 
 
 
#  --envs t10003_env1 --train_envs t10003_train 
 
 
        
# python icl_batch.py -n pi0_t10003_env_23 -c 2 --policy pi0_t10003 --task t10003 --vlm qwen3-vl-plus --queries_num 200 \
        # --envs t10003_env2 t10003_env3 --train_envs t10003_train 

# python icl_batch.py -n pi0_t10003_full -c 10 --policy pi0_t10003 --task t10003 --vlm qwen3-vl-plus --queries_num 300 \
#         --envs t10003_env1 t10003_env2 t10003_env3 --train_envs t10003_train 



# python icl_batch.py -n pi05_task40_full -c 10 --policy pi05_task40 --task task40 --vlm qwen3-vl-plus --queries_num 150 \
#         --envs task40_env1 task40_env2 task40_env3 --train_envs task40_train 


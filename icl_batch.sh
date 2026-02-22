# python icl_batch_staged.py -n pi0_task10_staged_321 -c 10 --policy pi0_t10 --task t10 --vlm qwen3-vl-plus --queries_num 30 \
#     --envs t100002_5mm_env1 t100001_5mm_env1 t100000_5mm_env1 --train_envs t100000_5mm_train t100001_5mm_train t100002_5mm_train t100003_5mm_train t100000_2mm_train t100001_2mm_train t100002_2mm_train t100003_2mm_train


#  --envs t70005_env1 t70004_env1 t70002_env1 t70003_env1 t70001_env1 t70000_env1 --train_envs t70000_train t70001_train t70002_train t70003_train t70004_train t70005_train
 
 
 
 
#  --envs t10003_env1 --train_envs t10003_train 
 
 
        
# python icl_batch.py -n pi0_t10003_env_23 -c 2 --policy pi0_t10003 --task t10003 --vlm qwen3-vl-plus --queries_num 200 \
        # --envs t10003_env2 t10003_env3 --train_envs t10003_train 

# python icl_batch.py -n pi0_t10003_full -c 10 --policy pi0_t10003 --task t10003 --vlm qwen3-vl-plus --queries_num 300 \
#         --envs t10003_env1 t10003_env2 t10003_env3 --train_envs t10003_train 



# python icl_batch.py -n pi05_task40_12 -c 10 --policy pi05_task40 --task task40 --vlm qwen3-vl-plus --queries_num 100 \
        # --envs task40_env1 task40_env2 --train_envs task40_train 

# python icl_batch.py -n pi0_t10003_sim_mixed -c 10 --policy pi0_t10003_sim --task t10003_sim --vlm qwen3-vl-plus --queries_num 300 \
#         --envs t10003_sim_env1 t10003_sim_env2 t10003_sim_env3 --train_envs t10003_sim_train 

python icl_batch.py -n pi0_t10003_sim_env_12 -c 1 --policy pi0_t10003_sim --task t10003_sim --vlm qwen3-vl-plus --queries_num 200 \
        --envs t10003_sim_env1 t10003_sim_env2  --train_envs t10003_sim_train 

python icl_batch.py -n pi0_t10003_sim_env_13 -c 1 --policy pi0_t10003_sim --task t10003_sim --vlm qwen3-vl-plus --queries_num 200 \
        --envs t10003_sim_env1 t10003_sim_env3  --train_envs t10003_sim_train 

python icl_batch.py -n pi0_t10003_sim_env_23 -c 1 --policy pi0_t10003_sim --task t10003_sim --vlm qwen3-vl-plus --queries_num 200 \
        --envs t10003_sim_env2 t10003_sim_env3  --train_envs t10003_sim_train 
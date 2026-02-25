# python icl_batch_staged.py -n pi0_task10_staged_321 -c 10 --policy pi0_t10 --task t10 --vlm qwen3-vl-plus --queries_num 30 \
#     --envs t100002_5mm_env1 t100001_5mm_env1 t100000_5mm_env1 --train_envs t100000_5mm_train t100001_5mm_train t100002_5mm_train t100003_5mm_train t100000_2mm_train t100001_2mm_train t100002_2mm_train t100003_2mm_train


#  --envs t70005_env1 t70004_env1 t70002_env1 t70003_env1 t70001_env1 t70000_env1 --train_envs t70000_train t70001_train t70002_train t70003_train t70004_train t70005_train
 
 
 
 
#  --envs t10003_env1 --train_envs t10003_train 
 
 
        
# python icl_batch.py -n pi0_t10003_000 -c 10 --policy pi0_t10003 --task t10003 --vlm qwen3-vl-plus \
    # --query_num 20 --query_envs t10003_env1 --query_sample_method random  --database_envs t10003_env2 t10003_env3 --database_num 100 --database_sample_method random
# python icl_batch.py -n pi0_t10003_001 -c 10 --policy pi0_t10003 --task t10003 --vlm qwen3-vl-plus \
#     --query_num 20 --query_envs t10003_env2 --query_sample_method random  --database_envs t10003_env1 t10003_env3 --database_num 100 --database_sample_method random
# python icl_batch.py -n pi0_t10003_002 -c 10 --policy pi0_t10003 --task t10003 --vlm qwen3-vl-plus \
#     --query_num 20 --query_envs t10003_env3 --query_sample_method random  --database_envs t10003_env1 t10003_env2 --database_num 100 --database_sample_method random

# python icl_batch.py -n pi0_t10003_007 -c 10 --policy pi0_t10003 --task t10003 --vlm qwen3-vl-plus \
#     --query_num 20 --query_envs t10003_poison --query_sample_method first  --database_envs t10003_env1 t10003_env2 t10003_env3 --database_num 100 --database_sample_method random
# python icl_batch.py -n pi0_t10003_006 -c 10 --policy pi0_t10003 --task t10003 --vlm qwen3-vl-plus \
#     --query_num 20 --query_envs t10003_poison --query_sample_method last  --database_envs t10003_env1 t10003_env2 t10003_env3 --database_num 100 --database_sample_method random
# python icl_batch.py -n pi0_t7_003 -c 10 --policy pi0_t7 --task t7 --vlm qwen3-vl-plus \
    # --query_num 20 --query_envs  t70005_env1 t70004_env1 t70002_env1 t70003_env1 t70001_env1 t70000_env1 --query_sample_method shared  --database_envs  t70005_env1 t70004_env1 t70002_env1 t70003_env1 t70001_env1 t70000_env1 --database_num 100 --database_sample_method shared
python icl_batch.py -n pi0_t10_004 -c 10 --policy pi0_t10 --task t10 --vlm qwen3-vl-plus \
    --query_num 20 --query_envs t100000_5mm_env1 t100001_5mm_env1 t100002_5mm_env1 t100003_5mm_env1  t100000_2mm_env1 t100001_2mm_env1 t100002_2mm_env1 t100003_2mm_env1 --query_sample_method shared  \
    --database_envs  t100000_5mm_env1 t100001_5mm_env1 t100002_5mm_env1 t100003_5mm_env1  t100000_2mm_env1 t100001_2mm_env1 t100002_2mm_env1 t100003_2mm_env1 --database_num 100 --database_sample_method shared

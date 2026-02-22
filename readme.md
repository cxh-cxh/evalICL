# Guide

## 文件结构

### images

    images
    |---<env_name1>
        |---imgxxx.png
        |---imgxxx.png
        ~
        |---info.json
    |---<env_name2>
    ~

### img_emb.hdf5

    /
    |---<model_name1>/
        |---<env_name1>/
            |---imgxxx.png/
                |---<embeddings>
            |---imgxxx.png/
                |---<embeddings>
            ~
        |---<env_name2>/
        ~
    |---<model_name_2>/
    ~

### video_emb.hdf5

    /
    |---<model_name1>/
        |---<env_name1>/
            |---xxx.mp4/
                |---<embeddings>
            |---xxx.mp4/
                |---<embeddings>
            ~
        |---<env_name2>/
        ~
    |---<model_name_2>/
    ~

## Test record format
`index` : begin at `0`

`is_train` : `0` or `1`, whether the record is train data

`task` : description of the task (`None` if no cross-task)

`front_img` : front image path (DEPRECATED)

`side_img` : side image path (DEPRECATED)

`success_rate` : `A/B` where `A` is success rollout count, `B` is total rollout count

`first_success` : `0` or `1`, whether the first rollout succeeds

`fail_reason` : reason for failure

`small_pos` : (for `t10003`) xy position of small cube

`big_pos` : (for `t10003`) xy position of big cube

`l2_cm` : (for `t10003`) idk

`l2_small_cm` : (for `t10003`) the distance of the small cube between the test case and the closest one in the train data

`l2_big_cm` : (for `t10003`) the distance of the big cube between the test case and the closest one in the train data

`progress_1_rate` : `A/B` where `A` is rollout count that reaches progress level 1, `B` is total rollout count

`progress_2_rate` : `A/B` where `A` is rollout count that reaches progress level 2, `B` is total rollout count

`progress_3_rate` : `A/B` where `A` is rollout count that reaches progress level 3, `B` is total rollout count


## Test record format (new)
`index` : begin at `0`

`is_train` : `0` or `1`, whether the record is train data

`task` : description of the task (`None` if no cross-task)

`progress` : the progress in each rollout. e.g. `[1, 2, 0, 1]`

`max_progress` : the progress to be regarded as success. e.g. `2`



## How to use

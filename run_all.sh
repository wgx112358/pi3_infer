#!/bin/bash

VIDEO_PATH="/inspire/hdd/global_user/zhangkaipeng-24043/lichuanhao/dataset/sekai-real-walking-hq/videos"

# 2. 在 --video-dir 参数后引用该变量
python /inspire/hdd/global_user/zhangkaipeng-24043/wgx/Pi3/prepare_job_list.py --video-dir "$VIDEO_PATH"

python /inspire/hdd/global_user/zhangkaipeng-24043/wgx/Pi3/run_distributed_worker.py  --instance-id 41 --total-instances 56 --video-dir "/inspire/hdd/global_user/zhangkaipeng-24043/lichuanhao/dataset/sekai-real-walking-hq/videos" --shared-dir /inspire/hdd/global_user/zhangkaipeng-24043/wgx/Pi3/output --output-dir /inspire/hdd/global_user/zhangkaipeng-24043/wgx/Pi3/output/sliced_videos/camera_poses

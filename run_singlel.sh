#!/bin/bash

#VIDEO_PATH="/inspire/hdd/global_user/zhangkaipeng-24043/lichuanhao/dataset/sekai-real-walking-hq/videos"
#python /inspire/hdd/global_user/zhangkaipeng-24043/wgx/Pi3/prepare_job_list.py --video-dir "$VIDEO_PATH"

# --- 1. 让脚本在遇到任何错误时立即退出 ---
# 这是一个非常重要的最佳实践，可以防止在环境配置失败后继续执行错误的指令。
#!/bin/bash

# ... (脚本前面的部分保持不变) ...
set -e

if [ -z "$1" ]; then
  echo "错误: 未提供 Instance ID。"
  echo "用法: $0 <instance_id>"
  exit 1
fi
INSTANCE_ID=$1
TOTAL_INSTANCES=56

eval "$(conda shell.bash hook)"
conda activate pi3

echo "Conda 环境 'pi3' 已激活。"
echo "启动工作实例 (Worker)... ID: $INSTANCE_ID / $TOTAL_INSTANCES"

cd /inspire/hdd/global_user/zhangkaipeng-24043/wgx/Pi3
echo "当前工作目录: $(pwd)"

# --- 使用 nohup 在后台运行，并将所有输出重定向到 /dev/null ---
# 这样就不会再生成 nohup.out 文件，因为所有输出都被丢弃了。
nohup python -u /inspire/hdd/global_user/zhangkaipeng-24043/wgx/Pi3/run_distributed_worker.py \
  --instance-id "$INSTANCE_ID" \
  --total-instances "$TOTAL_INSTANCES" \
  --video-dir "/inspire/hdd/global_user/zhangkaipeng-24043/lichuanhao/dataset/sekai-real-walking-hq/videos" \
  --shared-dir /inspire/hdd/global_user/zhangkaipeng-24043/wgx/Pi3/output \
  --output-dir /inspire/hdd/global_user/zhangkaipeng-24043/wgx/Pi3/output/sliced_videos/camera_poses \
  --timeout 7200 > /dev/null 2>&1 &


#!/bin/bash

#VIDEO_PATH="/inspire/hdd/global_user/zhangkaipeng-24043/lichuanhao/dataset/sekai-real-walking-hq/videos"
#python /inspire/hdd/global_user/zhangkaipeng-24043/wgx/Pi3/prepare_job_list.py --video-dir "$VIDEO_PATH"

# --- 1. 让脚本在遇到任何错误时立即退出 ---
# 这是一个非常重要的最佳实践，可以防止在环境配置失败后继续执行错误的指令。
set -e

# --- 1. 检查是否提供了所有必需的参数 ---
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "错误: 未提供所有必需的参数。"
  echo "用法: $0 <instance_id> <gpus_per_container>"
  exit 1
fi

INSTANCE_ID=$1
GPUS_PER_CONTAINER=$2 # 从第二个命令行参数获取 GPU 数量
TOTAL_INSTANCES=56

# --- 2. 激活 Conda 环境 ---
eval "$(conda shell.bash hook)"
conda activate pi3

echo "Conda 环境 'pi3' 已激活。"
echo "启动工作实例 (Worker)... ID: $INSTANCE_ID / $TOTAL_INSTANCES"
echo "此容器的 GPU 数量: $GPUS_PER_CONTAINER"

# --- 3. 切换工作目录 ---
cd /inspire/hdd/global_user/zhangkaipeng-24043/wgx/Pi3
echo "当前工作目录: $(pwd)"

# --- 4. 使用 nohup 在后台运行 Python 脚本 ---
# 将 --gpus-per-container 的值替换为从命令行传入的变量
nohup python -u /inspire/hdd/global_user/zhangkaipeng-24043/wgx/Pi3/run_distributed_worker.py \
  --instance-id "$INSTANCE_ID" \
  --total-instances "$TOTAL_INSTANCES" \
  --gpus-per-container "$GPUS_PER_CONTAINER" \
  --video-dir "/inspire/hdd/global_user/zhangkaipeng-24043/lichuanhao/dataset/sekai-real-walking-hq/videos" \
  --shared-dir /inspire/hdd/global_user/zhangkaipeng-24043/wgx/Pi3/output \
  --output-dir /inspire/hdd/global_user/zhangkaipeng-24043/wgx/Pi3/output/sliced_videos/camera_poses \
  --timeout 7200 > /dev/null 2>&1 &

echo "脚本已在后台启动。"
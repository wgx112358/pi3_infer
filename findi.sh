#!/bin/bash

# 检查用户是否提供了参数
if [ "$#" -ne 1 ]; then
    echo "用法: $0 <图片文件的完整路径>"
    echo "例如: $0 /inspire/hdd/global_user/zhangkaipeng-24043/wgx/Pi3_new/diff/fxBBAr9jRRQ_0022167_0023967_comparison.png"
    exit 1
fi

# 定义要搜索的基础目录
SEARCH_DIR="/inspire/hdd/global_user/zhangkaipeng-24043/lichuanhao/dataset/sekai-real-walking-hq"

# 1. 从输入的完整路径中获取文件名
# 例如: fxBBAr9jRRQ_0022167_0023967_comparison.png
input_path="$1"
filename=$(basename "$input_path")

# 2. 从文件名中提取需要搜索的核心部分
# 这里假设核心部分总是以 "_comparison.png" 结尾，并移除它
search_pattern="${filename%_comparison.png}"

# 打印将要执行的命令，方便确认
echo "--------------------------------------------------"
echo "输入路径: $input_path"
echo "提取模式: $search_pattern"
echo "搜索目录: $SEARCH_DIR"
echo "即将执行命令: find \"$SEARCH_DIR\" -name \"*${search_pattern}*\""
echo "--------------------------------------------------"

# 3. 执行 find 命令
find "$SEARCH_DIR" -name "*${search_pattern}*"
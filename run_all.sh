#!/bin/bash

# This script runs the entire parallel processing pipeline and logs all outputs.
# It ensures that the video list is generated first, then launches all
# GPU tasks in parallel, redirecting their output to separate log files.

# --- Configuration ---
# Set this to the starting physical ID of your second set of GPUs.
# For example, if you use GPUs 0,1,2,3 for full videos and 4,5,6,7 for sliced.
SLICED_GPU_OFFSET=4


# --- Step 0: Setup ---
# Stop the script immediately if any command fails.
set -e

# Create a directory for logs if it doesn't exist.
LOG_DIR="logs"
mkdir -p $LOG_DIR
echo "Logs will be saved in the '$LOG_DIR' directory."
echo "Clearing old logs..."
rm -f $LOG_DIR/*.log


# --- Step 1: Generate the unique video list ---
echo "--- Step 1: Generating video list ---"
python random_sample.py > $LOG_DIR/0_generate_list.log 2>&1
echo "Video list generated successfully. See '$LOG_DIR/0_generate_list.log' for details."


# --- Step 2: Launch all 8 parallel GPU tasks ---
echo "--- Step 2: Starting parallel processing for full and sliced videos ---"
echo "All processes will run in the background. You can monitor their progress in separate terminals using commands like:"
echo "tail -f $LOG_DIR/full_video_gpu_0.log"
echo "tail -f $LOG_DIR/sliced_video_gpu_4.log"
echo "..."

# Launch 4 Full Video processes (using physical GPUs starting from 0)
nohup python -u process_videos.py --gpu-id 0 --total-gpus 4 --gpu-offset 0 > $LOG_DIR/full_video_gpu_0.log 2>&1 &
nohup python -u process_videos.py --gpu-id 1 --total-gpus 4 --gpu-offset 0 > $LOG_DIR/full_video_gpu_1.log 2>&1 &
nohup python -u process_videos.py --gpu-id 2 --total-gpus 4 --gpu-offset 0 > $LOG_DIR/full_video_gpu_2.log 2>&1 &
nohup python -u process_videos.py --gpu-id 3 --total-gpus 4 --gpu-offset 0 > $LOG_DIR/full_video_gpu_3.log 2>&1 &

# Launch 4 Sliced Video processes (using physical GPUs starting from SLICED_GPU_OFFSET)
nohup python -u process_sliced_videos.py --gpu-id 0 --total-gpus 4 --gpu-offset $SLICED_GPU_OFFSET > $LOG_DIR/sliced_video_gpu_4.log 2>&1 &
nohup python -u process_sliced_videos.py --gpu-id 1 --total-gpus 4 --gpu-offset $SLICED_GPU_OFFSET > $LOG_DIR/sliced_video_gpu_5.log 2>&1 &
nohup python -u process_sliced_videos.py --gpu-id 2 --total-gpus 4 --gpu-offset $SLICED_GPU_OFFSET > $LOG_DIR/sliced_video_gpu_6.log 2>&1 &
nohup python -u process_sliced_videos.py --gpu-id 3 --total-gpus 4 --gpu-offset $SLICED_GPU_OFFSET > $LOG_DIR/sliced_video_gpu_7.log 2>&1 &


# --- Step 3: Wait for completion ---
echo "All 8 processes have been launched in the background."
echo "Waiting for all tasks to complete... (This may take a very long time)"
wait

echo "--- All processing tasks have completed successfully! ---"

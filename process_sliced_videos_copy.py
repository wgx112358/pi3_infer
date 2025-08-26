import os
import subprocess
import argparse

# --- Configuration ---
VIDEO_DIR = "/inspire/hdd/global_user/zhangkaipeng-24043/lichuanhao/dataset/sekai-real-walking-hq/videos"
TASK2_OUTPUT_DIR = "output/sliced_videos/camera_poses_test"
PI3_INFERENCE_SCRIPT = "run_inference_copy.py"
VIDEO_LIST_FILE = "output/full_videos/selected_videos.txt"

def get_videos_to_process():
    """ 
    Gets the list of video basenames by reading the saved list file.
    """
    try:
        with open(VIDEO_LIST_FILE, 'r') as f:
            # Read lines and strip any whitespace/newlines
            video_basenames = [line.strip() for line in f if line.strip()]
        return video_basenames
    except FileNotFoundError:
        print(f"Error: Video list file not found at '{VIDEO_LIST_FILE}'")
        print("Please run the script for Task 1 first to generate this file.")
        return None

def main():
    """
    Main function to process the sliced videos for Task 2 - Optimized version without ffmpeg.
    """
    # --- Add argument parser for parallel processing ---
    parser = argparse.ArgumentParser(description="Process sliced videos in parallel (optimized version).")
    parser.add_argument("--gpu-id", type=int, default=0, help="The GPU ID to use for this process.")
    parser.add_argument("--total-gpus", type=int, default=1, help="The total number of GPUs being used for processing.")
    parser.add_argument("--gpu-offset", type=int, default=0, help="The starting physical GPU ID for this task group.")
    args = parser.parse_args()

    # --- 1. Setup ---
    os.makedirs(TASK2_OUTPUT_DIR, exist_ok=True)

    # --- 2. Get list of videos from the saved file ---
    videos_to_process = get_videos_to_process()
    if videos_to_process is None:
        return

    print(f"Found {len(videos_to_process)} videos from list file to process for Task 2 (optimized).")

    # --- Distribute videos across GPUs ---
    if args.total_gpus > 1:
        print(f"Running in parallel mode. This is process for GPU {args.gpu_id}/{args.total_gpus}.")
        # Each process takes a slice of the video list
        videos_for_this_process = videos_to_process[args.gpu_id::args.total_gpus]
    else:
        print("Running in single GPU mode.")
        videos_for_this_process = videos_to_process

    print(f"This process will handle {len(videos_for_this_process)} videos.")

    # --- 3. Process each video ---
    total_videos = len(videos_for_this_process)
    physical_gpu_id = args.gpu_id + args.gpu_offset
    device_name = f"cuda:{physical_gpu_id}"
    print(f"This process will use physical GPU: {device_name}")

    for i, video_name in enumerate(videos_for_this_process):
        print(f"\n--- Processing video {i+1}/{total_videos}: {video_name} ---")
        original_video_path = os.path.join(VIDEO_DIR, video_name)
        base_name = os.path.splitext(video_name)[0]

        # --- Process each of the 3 slices directly without ffmpeg ---
        for slice_num in [1, 2, 3]:
            slice_name = f"{base_name}_part{slice_num}"
            output_poses_path = os.path.join(TASK2_OUTPUT_DIR, f"{slice_name}_poses.npz")
            
            print(f"  - Running inference on slice {slice_num}/3")
            
            if os.path.exists(output_poses_path):
                print(f"    Output file already exists, skipping: {output_poses_path}")
                continue

            # Use the modified run_inference.py with --slice_part parameter
            command = [
                "python", PI3_INFERENCE_SCRIPT,
                "--data_path", original_video_path,
                "--camera_poses_path", output_poses_path,
                "--interval", "1",  # Process every frame
                "--device", device_name,
                "--slice_part", str(slice_num)  # New parameter to slice the video
            ]
            
            try:
                subprocess.run(command, check=True)
                print(f"    Successfully processed slice {slice_num}")
            except subprocess.CalledProcessError as e:
                print(f"    Error processing slice {slice_num} of video {video_name}: {e}")
            except FileNotFoundError:
                print(f"Error: '{PI3_INFERENCE_SCRIPT}' not found. Make sure it's in the same directory.")
                return

    print(f"All camera poses for sliced videos have been saved in: {TASK2_OUTPUT_DIR}")
    print("Optimized processing completed - no temporary video files were created!")

if __name__ == "__main__":
    main()

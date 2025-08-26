import os
import random
import subprocess
import argparse

# --- Configuration ---
VIDEO_DIR = "/inspire/hdd/global_user/zhangkaipeng-24043/lichuanhao/dataset/sekai-real-walking-hq/videos"
OUTPUT_DIR = "output/full_videos/camera_poses"
PI3_INFERENCE_SCRIPT = "run_inference.py"
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
        print("Please run '1_generate_list.py' first to generate this file.")
        return None

def main():
    """
    Main function to process the videos.
    """
    # --- Add argument parser for parallel processing ---
    parser = argparse.ArgumentParser(description="Process a list of videos in parallel.")
    parser.add_argument("--gpu-id", type=int, default=0, help="The GPU ID to use for this process.")
    parser.add_argument("--total-gpus", type=int, default=1, help="The total number of GPUs being used for processing.")
    parser.add_argument("--gpu-offset", type=int, default=0, help="The starting physical GPU ID for this task group.")
    args = parser.parse_args()

    # --- 1. Setup ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. Get list of videos from the saved file ---
    # This script no longer selects videos; it reads the definitive list.
    selected_videos = get_videos_to_process()
    if selected_videos is None:
        return

    print(f"Found {len(selected_videos)} videos from list file to process.")
    
    # --- Distribute videos across GPUs ---
    if args.total_gpus > 1:
        print(f"Running in parallel mode. This is process for GPU {args.gpu_id}/{args.total_gpus}.")
        # Each process takes a slice of the video list
        videos_for_this_process = selected_videos[args.gpu_id::args.total_gpus]
    else:
        print("Running in single GPU mode.")
        videos_for_this_process = selected_videos

    print(f"This process will handle {len(videos_for_this_process)} videos.")

    # --- Save the list of selected videos (only master process needs to do this) ---
    # This block is now redundant as the list is read directly.
    # if args.gpu_id == 0:
    #     list_save_path = os.path.join(os.path.dirname(OUTPUT_DIR), "selected_videos.txt")
    #     with open(list_save_path, 'w') as f:
    #         for video_name in selected_videos:
    #             f.write(f"{video_name}\n")
    #     print(f"Saved selected video list to: {list_save_path}")

    # --- 3. Process each video assigned to this GPU ---
    physical_gpu_id = args.gpu_id + args.gpu_offset
    device_name = f"cuda:{physical_gpu_id}"
    print(f"This process will use physical GPU: {device_name}")

    for i, video_name in enumerate(videos_for_this_process):
        video_path = os.path.join(VIDEO_DIR, video_name)
        base_name = os.path.splitext(video_name)[0]
        camera_poses_output_path = os.path.join(OUTPUT_DIR, f"{base_name}_poses.npz")

        print(f"\n--- Processing video {i+1}/{len(videos_for_this_process)}: {video_name} ---")

        if os.path.exists(camera_poses_output_path):
            print(f"Output file already exists, skipping: {camera_poses_output_path}")
            continue

        command = [
            "python", PI3_INFERENCE_SCRIPT,
            "--data_path", video_path,
            "--camera_poses_path", camera_poses_output_path,
            "--interval", "1",  # Process every frame
            "--device", device_name
        ]

        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed and saved results to {camera_poses_output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing video {video_name}: {e}")
        except FileNotFoundError:
            print(f"Error: '{PI3_INFERENCE_SCRIPT}' not found. Make sure it's in the same directory.")
            return

    print(f"All camera poses have been saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

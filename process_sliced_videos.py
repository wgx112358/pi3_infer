import os
import subprocess
import argparse

# --- Configuration ---
VIDEO_DIR = "/inspire/hdd/global_user/zhangkaipeng-24043/lichuanhao/dataset/sekai-real-walking-hq/videos"
TASK1_OUTPUT_DIR = "output/full_videos/camera_poses"
TASK2_OUTPUT_DIR = "output/sliced_videos/camera_poses_test1vram"
TEMP_SLICE_DIR = "output/sliced_videos/temp_slices"
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
        print("Please run the script for Task 1 first to generate this file.")
        return None

def slice_video(video_path, video_name, slice_dir):
    """
    Slices a video into 3 parts of 20 seconds each using ffmpeg.
    Returns a list of paths to the sliced video parts.
    """
    base_name = os.path.splitext(video_name)[0]
    slice_paths = []
    slice_definitions = [
        {"name": f"{base_name}_part1.mp4", "start": "00:00:00", "end": "00:00:20"},
        {"name": f"{base_name}_part2.mp4", "start": "00:00:20", "end": "00:00:40"},
        {"name": f"{base_name}_part3.mp4", "start": "00:00:40", "end": "00:01:00"},
    ]

    for part in slice_definitions:
        output_path = os.path.join(slice_dir, part["name"])
        slice_paths.append(output_path)
        
        command = [
            "ffmpeg",
            "-i", video_path,
            "-ss", part["start"],
            "-to", part["end"],
            # "-c", "copy", # Removing this enables frame-accurate slicing by re-encoding
            "-y", # Overwrite output file if it exists
            output_path
        ]
        
        # Hide ffmpeg output for a cleaner log
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    return slice_paths

def main():
    """
    Main function to process the sliced videos for Task 2.
    """
    # --- Add argument parser for parallel processing ---
    parser = argparse.ArgumentParser(description="Process sliced videos in parallel.")
    parser.add_argument("--gpu-id", type=int, default=0, help="The GPU ID to use for this process.")
    parser.add_argument("--total-gpus", type=int, default=1, help="The total number of GPUs being used for processing.")
    parser.add_argument("--gpu-offset", type=int, default=0, help="The starting physical GPU ID for this task group.")
    args = parser.parse_args()

    # --- 1. Setup ---
    os.makedirs(TASK2_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_SLICE_DIR, exist_ok=True)

    # --- 2. Get list of videos from the saved file ---
    videos_to_process = get_videos_to_process()
    if videos_to_process is None:
        return

    print(f"Found {len(videos_to_process)} videos from list file to process for Task 2.")

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

        # --- 3a. Slice the video ---
        print("Slicing video into 3 parts...")
        try:
            slice_paths = slice_video(original_video_path, video_name, TEMP_SLICE_DIR)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error slicing video {video_name}. It might not exist or be corrupted. Skipping. Error: {e}")
            continue

        # --- 3b. Run inference on each slice ---
        for slice_path in slice_paths:
            slice_basename = os.path.basename(slice_path)
            slice_name_no_ext = os.path.splitext(slice_basename)[0]
            output_poses_path = os.path.join(TASK2_OUTPUT_DIR, f"{slice_name_no_ext}_poses.npz")
            
            print(f"  - Running inference on: {slice_basename}")
            
            if os.path.exists(output_poses_path):
                print(f"    Output file already exists, skipping: {output_poses_path}")
                continue

            command = [
                "python", PI3_INFERENCE_SCRIPT,
                "--data_path", slice_path,
                "--camera_poses_path", output_poses_path,
                "--interval", "1",  # Process every frame
                "--device", device_name
            ]
            
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"    Error processing slice {slice_basename}: {e}")

        # --- 3c. Clean up slices for the current video ---
        for slice_path in slice_paths:
            if os.path.exists(slice_path):
                os.remove(slice_path)
    
    # --- 4. Final Cleanup ---
    try:
        if not os.listdir(TEMP_SLICE_DIR):
             os.rmdir(TEMP_SLICE_DIR)
    except OSError as e:
        print(f"Could not remove temporary slice directory (it might not be empty): {e}")


    print(f"All camera poses for sliced videos have been saved in: {TASK2_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

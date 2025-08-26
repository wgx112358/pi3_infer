import os
import random

# --- Configuration ---
VIDEO_DIR = "/inspire/hdd/global_user/zhangkaipeng-24043/lichuanhao/dataset/sekai-real-walking-hq/videos"
OUTPUT_DIR = "output/full_videos"
NUM_VIDEOS_TO_PROCESS = 104
VIDEO_LIST_FILE = os.path.join(OUTPUT_DIR, "selected_videos.txt")

def main():
    """
    Selects a list of videos and saves it to a file. This serves as the
    single source of truth for all subsequent processing steps.
    """
    random.seed(42)  # Use a fixed seed for reproducibility
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        all_videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
        if len(all_videos) < NUM_VIDEOS_TO_PROCESS:
            selected_videos = all_videos
        else:
            selected_videos = random.sample(all_videos, NUM_VIDEOS_TO_PROCESS)
        
        with open(VIDEO_LIST_FILE, 'w') as f:
            for video_name in selected_videos:
                f.write(f"{video_name}\n")
        
        print(f"Successfully generated video list with {len(selected_videos)} videos.")
        print(f"List saved to: {VIDEO_LIST_FILE}")

    except FileNotFoundError:
        print(f"Error: The directory '{VIDEO_DIR}' was not found.")
        return

if __name__ == "__main__":
    main()
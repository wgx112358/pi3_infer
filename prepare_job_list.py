import os
import argparse
import sys
from datetime import datetime
from logger_utils import setup_simple_logging

def is_valid_video_file(file_path):
    """
    Basic validation to check if a file appears to be a valid video file.
    """
    if not os.path.exists(file_path):
        return False
    
    # Check file size (videos should be at least a few KB)
    try:
        file_size = os.path.getsize(file_path)
        if file_size < 1024:  # Less than 1KB is suspicious
            return False
    except OSError:
        return False
    
    return True

def scan_videos_with_validation(video_dir, logger):
    """
    Scan directory for video files with validation and progress reporting.
    """
    print(f"🔍 Scanning directory: {video_dir}")
    logger.info(f"Scanning video directory: {video_dir}")
    
    if not os.path.exists(video_dir):
        error_msg = f"Video directory does not exist: {video_dir}"
        print(f"❌ ERROR: {error_msg}")
        logger.error(error_msg)
        return None
    
    if not os.path.isdir(video_dir):
        error_msg = f"Path is not a directory: {video_dir}"
        print(f"❌ ERROR: {error_msg}")
        logger.error(error_msg)
        return None
    
    # Supported video extensions
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm')
    
    all_videos = []
    invalid_videos = []
    total_size = 0
    
    print("📂 Scanning for video files...")
    logger.info("Starting video file scan")
    
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.lower().endswith(video_extensions):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, video_dir)
                
                # Validate the video file
                if is_valid_video_file(full_path):
                    all_videos.append(relative_path)
                    try:
                        file_size = os.path.getsize(full_path)
                        total_size += file_size
                    except OSError:
                        pass  # Size calculation failed, but file is still valid
                else:
                    invalid_videos.append(relative_path)
                    print(f"⚠️  Invalid video file (too small or corrupted): {relative_path}")
                    logger.warning(f"Invalid video file: {relative_path}")
    
    if invalid_videos:
        print(f"\n📊 Found {len(invalid_videos)} invalid/corrupted video files:")
        logger.warning(f"Found {len(invalid_videos)} invalid video files")
        for invalid_video in invalid_videos[:10]:  # Show first 10
            print(f"   - {invalid_video}")
        if len(invalid_videos) > 10:
            print(f"   ... and {len(invalid_videos) - 10} more")
    
    print(f"\n✅ Found {len(all_videos)} valid video files")
    logger.info(f"Found {len(all_videos)} valid video files")
    
    if total_size > 0:
        size_gb = total_size / (1024**3)
        print(f"📊 Total size: {size_gb:.2f} GB")
        logger.info(f"Total video size: {size_gb:.2f} GB")
    
    return all_videos

def main():
    """
    Scans a directory for video files and creates a master list for distributed processing.
    This script should be run once before starting the container jobs.
    """
    parser = argparse.ArgumentParser(description="Create a master video list for distributed processing with validation and logging.")
    parser.add_argument("--video-dir", type=str, required=True,
                        help="The directory containing all the video files.")
    parser.add_argument("--shared-dir", type=str, required=True,
                        help="The shared, persistent directory where the master list will be saved.")
    parser.add_argument("--min-videos", type=int, default=1,
                        help="Minimum number of videos required (default: 1).")
    
    args = parser.parse_args()

    # Setup logging
    logger, log_file = setup_simple_logging(args.shared_dir, "prepare_job_list")
    
    print(f"🚀 Video List Preparation Tool")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Video directory: {args.video_dir}")
    print(f"💾 Shared directory: {args.shared_dir}")
    print(f"📝 Log file: {log_file}")

    logger.info("=== VIDEO LIST PREPARATION STARTED ===")
    logger.info(f"Video directory: {args.video_dir}")
    logger.info(f"Shared directory: {args.shared_dir}")
    logger.info(f"Minimum videos required: {args.min_videos}")

    # Validate shared directory
    try:
        os.makedirs(args.shared_dir, exist_ok=True)
        print(f"✅ Shared directory ready: {args.shared_dir}")
        logger.info(f"Shared directory created/verified: {args.shared_dir}")
    except Exception as e:
        error_msg = f"Cannot create shared directory {args.shared_dir}: {e}"
        print(f"❌ FATAL: {error_msg}")
        logger.error(f"FATAL: {error_msg}")
        sys.exit(1)

    # Find all valid video files
    all_videos = scan_videos_with_validation(args.video_dir, logger)
    
    if all_videos is None:
        logger.error("Video scanning failed")
        sys.exit(1)
    
    if len(all_videos) < args.min_videos:
        error_msg = f"Found only {len(all_videos)} videos, but minimum required is {args.min_videos}"
        print(f"❌ ERROR: {error_msg}")
        logger.error(error_msg)
        sys.exit(1)

    # Sort the list to ensure a deterministic and consistent order
    all_videos.sort()
    print(f"📋 Videos sorted for deterministic distributed processing")
    logger.info("Video list sorted for deterministic processing")

    # Calculate distribution statistics
    print(f"\n📊 Distribution Statistics (for 56 instances):")
    logger.info("=== DISTRIBUTION STATISTICS ===")
    videos_per_instance = len(all_videos) // 56
    remainder = len(all_videos) % 56
    print(f"   - Videos per instance: {videos_per_instance}")
    logger.info(f"Videos per instance: {videos_per_instance}")
    if remainder > 0:
        print(f"   - {remainder} instances will get 1 extra video")
        logger.info(f"{remainder} instances will get 1 extra video")
    print(f"   - Instance 0 will process: {videos_per_instance + (1 if remainder > 0 else 0)} videos")
    print(f"   - Instance 55 will process: {videos_per_instance + (1 if remainder > 55 else 0)} videos")

    # Save the master list to the shared directory
    master_list_path = os.path.join(args.shared_dir, "master_video_list.txt")

    try:
        with open(master_list_path, 'w') as f:
            for video_path in all_videos:
                f.write(f"{video_path}\n")
        
        print(f"\n✅ Master video list successfully saved to: {master_list_path}")
        logger.info(f"Master video list saved: {master_list_path}")
        
        # Create a metadata file with additional information
        metadata_path = os.path.join(args.shared_dir, "job_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total videos: {len(all_videos)}\n")
            f.write(f"Source directory: {args.video_dir}\n")
            f.write(f"Videos per instance (56 total): {videos_per_instance}\n")
            f.write(f"Instances with extra video: {remainder}\n")
        
        print(f"📋 Job metadata saved to: {metadata_path}")
        logger.info(f"Job metadata saved: {metadata_path}")
        
    except Exception as e:
        error_msg = f"Could not save master list to {master_list_path}: {e}"
        print(f"❌ FATAL: {error_msg}")
        logger.error(f"FATAL: {error_msg}")
        sys.exit(1)

    print(f"\n🎉 Setup complete! You can now start your 56 container instances.")
    print(f"💡 Each instance should use:")
    print(f"   --instance-id [0-55]")
    print(f"   --total-instances 56")
    print(f"   --shared-dir {args.shared_dir}")
    
    logger.info("=== VIDEO LIST PREPARATION COMPLETED SUCCESSFULLY ===")
    logger.info(f"Total videos processed: {len(all_videos)}")
    logger.info("Ready for distributed processing with 56 instances")

if __name__ == "__main__":
    main()

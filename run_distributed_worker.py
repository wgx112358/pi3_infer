import os
import argparse
import subprocess
import time
import sys
from datetime import datetime
from logger_utils import comprehensive_logging_context, log_system_info, log_processing_summary

def get_job_list(master_list_path):
    """
    Reads the master list of all video files.
    """
    try:
        with open(master_list_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"FATAL: Master video list not found at '{master_list_path}'.")
        print("Please run '1_prepare_job_list.py' first.")
        return None
    except Exception as e:
        print(f"FATAL: Error reading master list: {e}")
        return None

def get_completed_jobs(completed_dir):
    """
    Scans the 'completed' directory to find which jobs are already done.
    Returns a set of completed video basenames for fast lookups.
    """
    if not os.path.exists(completed_dir):
        return set()
    
    try:
        # The marker filename is the video's basename (without extension)
        completed_basenames = {os.path.splitext(f)[0] for f in os.listdir(completed_dir) if f.endswith('.completed')}
        return completed_basenames
    except Exception as e:
        print(f"Warning: Error reading completed jobs directory: {e}")
        return set()

def check_all_slices_exist(output_dir, video_basename):
    """
    Check if all 3 slices for a video have been successfully processed.
    This prevents race conditions where partial completion might be marked as full completion.
    """
    for slice_num in [1, 2, 3]:
        slice_output_name = f"{video_basename}_part{slice_num}_poses.npz"
        slice_output_path = os.path.join(output_dir, slice_output_name)
        if not os.path.exists(slice_output_path):
            return False
    return True

def mark_job_as_completed(completed_dir, video_relative_path, output_dir):
    """
    Creates an empty marker file to signify that a video has been fully processed.
    This operation includes verification and is atomic.
    """
    video_basename = os.path.splitext(os.path.basename(video_relative_path))[0]
    
    # Double-check that all slices actually exist before marking as complete
    if not check_all_slices_exist(output_dir, video_basename):
        print(f"ERROR: Cannot mark {video_basename} as completed - not all slice outputs exist!")
        return False
    
    marker_path = os.path.join(completed_dir, f"{video_basename}.completed")
    try:
        # Create the file in an atomic way
        with open(marker_path, 'x') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"Completed at: {timestamp}\nInstance: {os.getenv('INSTANCE_ID', 'N/A')}\n")
        print(f"Successfully marked job as completed: {video_basename}")
        return True
    except FileExistsError:
        # This can happen in rare race conditions, but it's safe since the job is done
        print(f"Warning: Marker file already existed for {video_basename} (race condition).")
        return True
    except Exception as e:
        print(f"ERROR: Could not create marker file for {video_basename}. Error: {e}")
        return False

def log_failed_job(shared_dir, video_relative_path, reason, slice_num=None):
    """
    Appends the path of a failed video to a log file for manual review.
    """
    log_path = os.path.join(shared_dir, "failed_jobs.log")
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    instance_id = os.getenv('INSTANCE_ID', 'N/A')
    slice_info = f" (Slice {slice_num})" if slice_num else ""
    log_entry = f"{timestamp} - INSTANCE {instance_id} - VIDEO: {video_relative_path}{slice_info} - REASON: {reason}\n"
    
    try:
        # Use 'a' to append to the file
        with open(log_path, 'a') as f:
            f.write(log_entry)
        print(f"Logged failed job to {log_path}")
    except Exception as e:
        print(f"ERROR: Could not write to failed jobs log. Error: {e}")

def log_progress(shared_dir, instance_id, completed_count, total_count):
    """
    Log current progress to a shared progress file for monitoring.
    """
    progress_dir = os.path.join(shared_dir, "progress")
    os.makedirs(progress_dir, exist_ok=True)
    progress_file = os.path.join(progress_dir, f"instance_{instance_id}.progress")
    
    try:
        with open(progress_file, 'w') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            progress_percentage = (completed_count / total_count * 100) if total_count > 0 else 0
            f.write(f"timestamp: {timestamp}\n")
            f.write(f"completed: {completed_count}\n")
            f.write(f"total: {total_count}\n")
            f.write(f"percentage: {progress_percentage:.1f}%\n")
    except Exception as e:
        print(f"Warning: Could not write progress file: {e}")

def validate_environment():
    """
    Validate that the environment and dependencies are properly set up.
    """
    # Check if run_inference.py exists
    if not os.path.exists("run_inference.py"):
        print("FATAL: run_inference.py not found in current directory!")
        return False
    
    # Check if we can import required modules
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("FATAL: PyTorch not available!")
        return False
    
    return True

def main():
    """
    Main processing script for a single container instance with comprehensive logging and memory optimization.
    """
    parser = argparse.ArgumentParser(description="Distributed video processing worker with memory optimization and comprehensive logging.")
    # --- Job Distribution Arguments ---
    parser.add_argument("--instance-id", type=int, required=True, help="The unique ID of this container instance (e.g., from 0 to 55).")
    parser.add_argument("--total-instances", type=int, required=True, help="The total number of container instances.")
    
    # --- Path Arguments ---
    parser.add_argument("--video-dir", type=str, required=True, help="The base directory where videos are stored.")
    parser.add_argument("--shared-dir", type=str, required=True, help="The shared, persistent directory for job management.")
    parser.add_argument("--output-dir", type=str, required=True, help="The directory to save the final output files (.npz).")

    # --- Model & Inference Arguments ---
    parser.add_argument("--gpu-id", type=int, default=0, help="The GPU ID to use within the container.")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a local model checkpoint file.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries per slice.")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds for each slice processing.")
    
    args = parser.parse_args()

    # Set environment variable for logging
    os.environ['INSTANCE_ID'] = str(args.instance_id)

    # Use comprehensive logging context
    with comprehensive_logging_context(args.shared_dir, args.instance_id) as logger:
        
        # Log system information for debugging
        log_system_info(logger)
        
        # Log startup information
        logger.info(f"Worker instance {args.instance_id}/{args.total_instances} starting")
        logger.info(f"Video directory: {args.video_dir}")
        logger.info(f"Shared directory: {args.shared_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"GPU ID: {args.gpu_id}")
        
        start_time = time.time()

        # --- 0. Environment Validation ---
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)

        # --- 1. Setup Paths ---
        master_list_path = os.path.join(args.shared_dir, "master_video_list.txt")
        completed_dir = os.path.join(args.shared_dir, "completed_markers")
        os.makedirs(completed_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)

        print(f"Worker Instance {args.instance_id}/{args.total_instances}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Shared directory: {args.shared_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"GPU ID: {args.gpu_id}")

        # --- 2. Get Full Job List ---
        all_jobs = get_job_list(master_list_path)
        if all_jobs is None:
            logger.error("Could not load job list")
            sys.exit(1)

        # --- 3. Determine Jobs for THIS Instance ---
        # Each instance gets a unique, consistent slice of the master list
        jobs_for_this_instance = all_jobs[args.instance_id::args.total_instances]
        
        if not jobs_for_this_instance:
            print("No jobs assigned to this instance. Exiting.")
            logger.info("No jobs assigned to this instance")
            return

        print(f"Total jobs in master list: {len(all_jobs)}")
        print(f"Jobs assigned to this instance: {len(jobs_for_this_instance)}")
        
        logger.info(f"Assigned {len(jobs_for_this_instance)} jobs out of {len(all_jobs)} total")

        # --- 4. Main Processing Loop ---
        completed_in_this_session = 0
        failed_in_this_session = 0
        
        for i, video_relative_path in enumerate(jobs_for_this_instance):
            video_basename = os.path.splitext(os.path.basename(video_relative_path))[0]
            
            print(f"\n[{i+1}/{len(jobs_for_this_instance)}] Processing: {video_relative_path}")
            logger.info(f"Starting video {i+1}/{len(jobs_for_this_instance)}: {video_relative_path}")
            
            # Define video_full_path first
            video_full_path = os.path.join(args.video_dir, video_relative_path)
            
            # Log basic video file information for debugging
            try:
                import cv2
                cap = cv2.VideoCapture(video_full_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"Video: {total_frames} frames, {fps:.1f} FPS")
                    logger.info(f"Video: {total_frames} frames, {fps:.1f} FPS")
                    cap.release()
                else:
                    logger.warning(f"Could not read video metadata for {video_relative_path}")
            except Exception as e:
                logger.warning(f"Error reading video metadata: {e}")

            # --- 4a. Check if job is ALREADY FULLY completed (Race Condition Safe) ---
            completed_jobs = get_completed_jobs(completed_dir)
            if video_basename in completed_jobs:
                print("Already fully completed. Skipping.")
                logger.info(f"Video {video_basename} already completed, skipping")
                continue

            # Additional check: verify all slice outputs exist
            if check_all_slices_exist(args.output_dir, video_basename):
                print("All slice outputs exist but not marked as complete. Marking now...")
                logger.info(f"Video {video_basename} has all slices but not marked complete, marking now")
                if mark_job_as_completed(completed_dir, video_relative_path, args.output_dir):
                    completed_in_this_session += 1
                continue

            # --- 4b. Check if the source video file exists (Fault Tolerance) ---
            if not os.path.exists(video_full_path):
                error_msg = f"Source video file not found: {video_full_path}"
                print(f"Video file not found: {video_full_path}")
                logger.error(f"Video file not found: {video_full_path}")
                log_failed_job(args.shared_dir, video_relative_path, error_msg)
                failed_in_this_session += 1
                continue

            # --- 4c. Process each of the 3 slices with memory optimization ---
            is_video_fully_processed = True
            failed_slice = None
            
            for slice_num in [1, 2, 3]:
                slice_output_name = f"{video_basename}_part{slice_num}_poses.npz"
                slice_output_path = os.path.join(args.output_dir, slice_output_name)

                print(f"  Processing slice {slice_num}/3...")
                logger.info(f"Processing slice {slice_num}/3 for video {video_basename}")
                
                if os.path.exists(slice_output_path):
                    print(f"    Slice output already exists, skipping: {slice_output_name}")
                    logger.info(f"Slice {slice_num} output already exists, skipping")
                    continue

                # --- Retry Logic with Improved Error Handling ---
                slice_succeeded = False
                last_error = None
                
                for attempt in range(args.max_retries):
                    print(f"    Attempt {attempt + 1}/{args.max_retries} for slice {slice_num}...")
                    logger.info(f"Slice {slice_num} attempt {attempt + 1}/{args.max_retries}")
                    
                    device_name = f"cuda:{args.gpu_id}"
                    command = [
                        "python", "run_inference.py",
                        "--data_path", video_full_path,
                        "--camera_poses_path", slice_output_path,
                        "--interval", "1",
                        "--device", device_name,
                        "--slice_part", str(slice_num)  # Memory-optimized slicing
                    ]
                    if args.ckpt:
                        command.extend(["--ckpt", args.ckpt])

                    try:
                        print(f"      Running: {' '.join(command)}")
                        logger.debug(f"Command: {' '.join(command)}")
                        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=args.timeout)
                        
                        # Verify output file was actually created
                        if os.path.exists(slice_output_path):
                            print(f"    SUCCESS: Slice {slice_num} completed on attempt {attempt + 1}")
                            logger.info(f"Slice {slice_num} completed successfully on attempt {attempt + 1}")
                            slice_succeeded = True
                            break
                        else:
                            last_error = f"Output file {slice_output_path} was not created despite successful return code"
                            print(f"    {last_error}")
                            logger.warning(last_error)

                    except subprocess.TimeoutExpired:
                        last_error = f"Process timed out after {args.timeout} seconds"
                        print(f"    TIMEOUT on attempt {attempt + 1}: {last_error}")
                        logger.warning(f"Slice {slice_num} timed out on attempt {attempt + 1}")
                        
                    except subprocess.CalledProcessError as e:
                        error_output = e.stderr.strip() if e.stderr else "N/A"
                        last_error = f"Return code {e.returncode}. Stderr: {error_output}"
                        print(f"    ERROR on attempt {attempt + 1}: {last_error}")
                        logger.warning(f"Slice {slice_num} failed on attempt {attempt + 1}: {last_error}")
                        
                    except Exception as e:
                        last_error = f"Unexpected error: {str(e)}"
                        print(f"    FATAL ERROR: {last_error}")
                        logger.error(f"Unexpected error in slice {slice_num}: {last_error}")
                        break  # Don't retry on unexpected errors

                    if attempt < args.max_retries - 1:
                        print(f"      Retrying in 5 seconds...")
                        time.sleep(5)

                # If this slice ultimately failed after all retries
                if not slice_succeeded:
                    print(f"    FAILURE: Slice {slice_num} failed after {args.max_retries} attempts")
                    logger.error(f"Slice {slice_num} failed after all retry attempts: {last_error}")
                    log_failed_job(args.shared_dir, video_relative_path, last_error or "Unknown error", slice_num)
                    is_video_fully_processed = False
                    failed_slice = slice_num
                    break

            # --- 4d. Mark Job as Completed (ATOMIC step with verification) ---
            if is_video_fully_processed:
                if mark_job_as_completed(completed_dir, video_relative_path, args.output_dir):
                    completed_in_this_session += 1
                    print(f"Video {video_basename} fully processed and marked as completed!")
                    logger.info(f"Video {video_basename} completed successfully")
                else:
                    print(f"Video {video_basename} processed but could not be marked as completed!")
                    logger.error(f"Video {video_basename} could not be marked as completed")
            else:
                failure_reason = f"Slice {failed_slice} failed after {args.max_retries} retries"
                print(f"Video {video_basename} was not fully processed: {failure_reason}")
                logger.error(f"Video {video_basename} failed: {failure_reason}")
                failed_in_this_session += 1

            # Update progress
            log_progress(args.shared_dir, args.instance_id, completed_in_this_session, len(jobs_for_this_instance))

        end_time = time.time()
        
        print(f"\nInstance {args.instance_id} completed its assigned jobs")
        print(f"Processed {completed_in_this_session} videos in this session")
        print(f"Failed {failed_in_this_session} videos in this session")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Log comprehensive summary
        log_processing_summary(
            logger, 
            len(jobs_for_this_instance), 
            completed_in_this_session, 
            failed_in_this_session, 
            start_time, 
            end_time
        )

if __name__ == "__main__":
    main()

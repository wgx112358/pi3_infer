#!/usr/bin/env python3
"""
Progress monitoring script for distributed video processing.
This script provides real-time monitoring of all worker instances.
"""

import os
import argparse
import time
import glob
from datetime import datetime, timedelta
import json
from logger_utils import setup_simple_logging

def read_progress_file(progress_file):
    """
    Read a progress file and return the parsed data.
    """
    try:
        with open(progress_file, 'r') as f:
            data = {}
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    data[key.strip()] = value.strip()
            return data
    except Exception as e:
        return None

def get_master_job_count(shared_dir):
    """
    Get the total number of jobs from the master list.
    """
    master_list_path = os.path.join(shared_dir, "master_video_list.txt")
    try:
        with open(master_list_path, 'r') as f:
            return len([line for line in f if line.strip()])
    except Exception:
        return 0

def get_completed_count(shared_dir):
    """
    Count the number of completed jobs.
    """
    completed_dir = os.path.join(shared_dir, "completed_markers")
    if not os.path.exists(completed_dir):
        return 0
    
    try:
        completed_files = [f for f in os.listdir(completed_dir) if f.endswith('.completed')]
        return len(completed_files)
    except Exception:
        return 0

def get_failed_jobs(shared_dir):
    """
    Read and parse the failed jobs log.
    """
    failed_log_path = os.path.join(shared_dir, "failed_jobs.log")
    if not os.path.exists(failed_log_path):
        return []
    
    failed_jobs = []
    try:
        with open(failed_log_path, 'r') as f:
            for line in f:
                if line.strip():
                    failed_jobs.append(line.strip())
        return failed_jobs
    except Exception:
        return []

def format_time_duration(seconds):
    """
    Format seconds into a human-readable duration.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"

def estimate_completion_time(completed, total, start_time):
    """
    Estimate when the processing will be completed.
    """
    if completed == 0:
        return "Unknown"
    
    elapsed = time.time() - start_time
    rate = completed / elapsed  # jobs per second
    remaining = total - completed
    
    if rate <= 0:
        return "Unknown"
    
    eta_seconds = remaining / rate
    eta_time = datetime.now() + timedelta(seconds=eta_seconds)
    return eta_time.strftime("%Y-%m-%d %H:%M:%S")

def main():
    """
    Monitor the progress of all distributed worker instances.
    """
    parser = argparse.ArgumentParser(description="Monitor distributed video processing progress with logging.")
    parser.add_argument("--shared-dir", type=str, required=True,
                        help="The shared directory where progress files are stored.")
    parser.add_argument("--refresh-interval", type=int, default=30,
                        help="Refresh interval in seconds (default: 30).")
    parser.add_argument("--total-instances", type=int, default=56,
                        help="Total number of worker instances (default: 56).")
    parser.add_argument("--export-json", type=str, default=None,
                        help="Export progress data to JSON file.")
    
    args = parser.parse_args()

    # Setup logging
    logger, log_file = setup_simple_logging(args.shared_dir, "monitor_progress")

    print(f"🔍 Distributed Processing Monitor")
    print(f"📂 Shared directory: {args.shared_dir}")
    print(f"🔄 Refresh interval: {args.refresh_interval} seconds")
    print(f"⚙️  Expected instances: {args.total_instances}")
    print(f"📝 Monitor log: {log_file}")
    print("=" * 80)

    logger.info("=== MONITORING SESSION STARTED ===")
    logger.info(f"Shared directory: {args.shared_dir}")
    logger.info(f"Refresh interval: {args.refresh_interval} seconds")
    logger.info(f"Expected instances: {args.total_instances}")
    if args.export_json:
        logger.info(f"JSON export file: {args.export_json}")

    start_time = time.time()
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # Clear screen (works on most terminals)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"🔍 Distributed Processing Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
            # Get overall statistics
            total_jobs = get_master_job_count(args.shared_dir)
            completed_jobs = get_completed_count(args.shared_dir)
            failed_jobs = get_failed_jobs(args.shared_dir)
            
            if total_jobs > 0:
                completion_percentage = (completed_jobs / total_jobs) * 100
            else:
                completion_percentage = 0
            
            print(f"📊 OVERALL PROGRESS:")
            print(f"   Total jobs: {total_jobs:,}")
            print(f"   Completed: {completed_jobs:,} ({completion_percentage:.1f}%)")
            print(f"   Failed: {len(failed_jobs):,}")
            print(f"   Remaining: {total_jobs - completed_jobs:,}")
            
            # Log progress every 10 iterations (to avoid too frequent logging)
            if iteration % 10 == 0:
                logger.info(f"Progress update: {completed_jobs}/{total_jobs} ({completion_percentage:.1f}%) | Failed: {len(failed_jobs)}")
            
            if total_jobs > 0 and completed_jobs > 0:
                eta = estimate_completion_time(completed_jobs, total_jobs, start_time)
                print(f"   ETA: {eta}")
                if iteration % 10 == 0:
                    logger.info(f"ETA: {eta}")
            
            # Progress bar
            bar_width = 50
            filled = int(bar_width * completion_percentage / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"   [{bar}] {completion_percentage:.1f}%")
            
            print()
            
            # Get instance-specific progress
            progress_dir = os.path.join(args.shared_dir, "progress")
            active_instances = 0
            total_instance_completed = 0
            
            if os.path.exists(progress_dir):
                progress_files = glob.glob(os.path.join(progress_dir, "instance_*.progress"))
                progress_data = []
                
                for progress_file in progress_files:
                    instance_id = os.path.basename(progress_file).replace("instance_", "").replace(".progress", "")
                    data = read_progress_file(progress_file)
                    
                    if data:
                        try:
                            completed = int(data.get('completed', 0))
                            total = int(data.get('total', 0))
                            percentage = float(data.get('percentage', 0))
                            timestamp = data.get('timestamp', 'Unknown')
                            
                            progress_data.append({
                                'instance_id': int(instance_id),
                                'completed': completed,
                                'total': total,
                                'percentage': percentage,
                                'timestamp': timestamp
                            })
                            
                            total_instance_completed += completed
                            active_instances += 1
                            
                        except ValueError:
                            pass  # Skip invalid data
                
                # Sort by instance ID
                progress_data.sort(key=lambda x: x['instance_id'])
                
                print(f"🖥️  INSTANCE STATUS ({active_instances}/{args.total_instances} reporting):")
                print("   ID  | Completed/Total  | Progress | Last Update")
                print("   " + "-" * 54)
                
                # Log instance summary every 10 iterations
                if iteration % 10 == 0:
                    logger.info(f"Active instances: {active_instances}/{args.total_instances}")
                
                for data in progress_data:
                    instance_id = data['instance_id']
                    completed = data['completed']
                    total = data['total']
                    percentage = data['percentage']
                    timestamp = data['timestamp']
                    
                    # Create mini progress bar
                    mini_bar_width = 10
                    mini_filled = int(mini_bar_width * percentage / 100) if percentage > 0 else 0
                    mini_bar = "█" * mini_filled + "░" * (mini_bar_width - mini_filled)
                    
                    print(f"   {instance_id:2d}  | {completed:4d}/{total:4d}     | [{mini_bar}] {percentage:5.1f}% | {timestamp[-8:]}")  # Show only time part
            else:
                print("⚠️  No progress data available yet")
                if iteration == 1:
                    logger.warning("No progress data available yet")
            
            # Show recent failures if any
            if failed_jobs:
                print(f"\n❌ RECENT FAILURES ({len(failed_jobs)} total):")
                for failure in failed_jobs[-5:]:  # Show last 5 failures
                    print(f"   {failure}")
                if len(failed_jobs) > 5:
                    print(f"   ... and {len(failed_jobs) - 5} more (check failed_jobs.log)")
                
                # Log failure summary every 10 iterations
                if iteration % 10 == 0 and failed_jobs:
                    logger.warning(f"Current failures: {len(failed_jobs)}")
                    # Log recent failures to structured log
                    for failure in failed_jobs[-3:]:  # Log last 3 failures
                        logger.error(f"Recent failure: {failure}")
            
            # Export to JSON if requested
            if args.export_json:
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'total_jobs': total_jobs,
                    'completed_jobs': completed_jobs,
                    'failed_jobs': len(failed_jobs),
                    'completion_percentage': completion_percentage,
                    'active_instances': active_instances,
                    'expected_instances': args.total_instances
                }
                
                try:
                    with open(args.export_json, 'w') as f:
                        json.dump(export_data, f, indent=2)
                except Exception as e:
                    print(f"\n⚠️  Could not export JSON: {e}")
                    logger.error(f"JSON export failed: {e}")
            
            print(f"\n🔄 Refreshing in {args.refresh_interval} seconds... (Ctrl+C to exit)")
            time.sleep(args.refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
        print(f"📊 Final status: {completed_jobs}/{total_jobs} jobs completed ({completion_percentage:.1f}%)")
        logger.info("=== MONITORING SESSION ENDED BY USER ===")
        logger.info(f"Final status: {completed_jobs}/{total_jobs} jobs completed ({completion_percentage:.1f}%)")
        logger.info(f"Active instances at end: {active_instances}/{args.total_instances}")
        logger.info(f"Total failures: {len(failed_jobs)}")

if __name__ == "__main__":
    main() 
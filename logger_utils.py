#!/usr/bin/env python3
"""
Comprehensive logging utilities for distributed video processing.
Captures all program output to files while maintaining console display.
"""

import os
import sys
import logging
import datetime
from logging.handlers import RotatingFileHandler
import contextlib
import io

class TeeLogger:
    """
    A class that duplicates print output to both console and file.
    """
    def __init__(self, console_stream, file_stream):
        self.console = console_stream
        self.file = file_stream
    
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.console.flush()
        self.file.flush()
    
    def flush(self):
        self.console.flush()
        self.file.flush()

def setup_comprehensive_logging(shared_dir, instance_id=None, log_level=logging.INFO):
    """
    Set up comprehensive logging system that captures:
    1. All print statements to files
    2. Structured logging with different levels
    3. Automatic log rotation
    4. Console output preservation
    
    Args:
        shared_dir (str): Directory where log files will be stored
        instance_id (int, optional): Instance ID for worker processes
        log_level: Logging level (default: INFO)
    
    Returns:
        tuple: (logger, log_file_path, stdout_redirect_file)
    """
    
    # Create logs directory
    logs_dir = os.path.join(shared_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate log file names with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if instance_id is not None:
        log_prefix = f"worker_instance_{instance_id}_{timestamp}"
    else:
        log_prefix = f"main_process_{timestamp}"
    
    # Structured logging file
    structured_log_file = os.path.join(logs_dir, f"{log_prefix}.log")
    
    # Complete output capture file (all print statements)
    stdout_log_file = os.path.join(logs_dir, f"{log_prefix}_output.log")
    
    # Set up structured logging
    logger = logging.getLogger(f"distributed_processing_{instance_id or 'main'}")
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    # File handler with rotation (max 10MB, keep 5 backups)
    file_handler = RotatingFileHandler(
        structured_log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (optional, can be disabled in production)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors on console
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Set up stdout/stderr redirection for capturing ALL output
    stdout_file = open(stdout_log_file, 'a', encoding='utf-8')
    
    # Write header to output file
    stdout_file.write(f"\n{'='*80}\n")
    stdout_file.write(f"DISTRIBUTED PROCESSING LOG - {datetime.datetime.now()}\n")
    if instance_id is not None:
        stdout_file.write(f"WORKER INSTANCE: {instance_id}\n")
    stdout_file.write(f"{'='*80}\n\n")
    stdout_file.flush()
    
    # Create tee logger for stdout
    original_stdout = sys.stdout
    tee_stdout = TeeLogger(original_stdout, stdout_file)
    
    return logger, structured_log_file, stdout_log_file, tee_stdout, stdout_file, original_stdout

def activate_output_capture(tee_stdout):
    """
    Activate stdout capture to file while preserving console output.
    """
    sys.stdout = tee_stdout

def deactivate_output_capture(original_stdout, stdout_file):
    """
    Restore original stdout and close log file.
    """
    sys.stdout = original_stdout
    if stdout_file and not stdout_file.closed:
        stdout_file.write(f"\n{'='*80}\n")
        stdout_file.write(f"LOG ENDED - {datetime.datetime.now()}\n")
        stdout_file.write(f"{'='*80}\n")
        stdout_file.close()

@contextlib.contextmanager
def comprehensive_logging_context(shared_dir, instance_id=None, log_level=logging.INFO):
    """
    Context manager for comprehensive logging.
    
    Usage:
        with comprehensive_logging_context("/shared/dir", instance_id=0) as logger:
            print("This will be logged to file and console")
            logger.info("This is structured logging")
    """
    logger, structured_log_file, stdout_log_file, tee_stdout, stdout_file, original_stdout = setup_comprehensive_logging(
        shared_dir, instance_id, log_level
    )
    
    try:
        activate_output_capture(tee_stdout)
        
        # Log session start
        logger.info(f"Logging session started")
        logger.info(f"Structured log: {structured_log_file}")
        logger.info(f"Output log: {stdout_log_file}")
        
        yield logger
        
    finally:
        logger.info("Logging session ended")
        deactivate_output_capture(original_stdout, stdout_file)

def log_system_info(logger):
    """
    Log useful system information for debugging.
    """
    import platform
    import psutil
    import torch
    
    logger.info("=== SYSTEM INFORMATION ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"CPU count: {psutil.cpu_count()}")
    logger.info(f"Memory total: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: Yes")
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name} ({props.total_memory / (1024**3):.2f} GB)")
    else:
        logger.info("CUDA available: No")

def log_processing_summary(logger, total_videos, completed_videos, failed_videos, start_time, end_time):
    """
    Log a comprehensive processing summary.
    """
    duration = end_time - start_time
    success_rate = (completed_videos / total_videos * 100) if total_videos > 0 else 0
    
    logger.info("=== PROCESSING SUMMARY ===")
    logger.info(f"Total videos: {total_videos}")
    logger.info(f"Completed: {completed_videos}")
    logger.info(f"Failed: {failed_videos}")
    logger.info(f"Success rate: {success_rate:.1f}%")
    logger.info(f"Processing time: {duration}")
    if completed_videos > 0:
        avg_time = duration / completed_videos
        logger.info(f"Average time per video: {avg_time}")

def setup_simple_logging(shared_dir, script_name):
    """
    Simplified logging setup for scripts that don't need full output capture.
    """
    logs_dir = os.path.join(shared_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"{script_name}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(script_name)
    logger.info(f"Logging initialized: {log_file}")
    
    return logger, log_file 
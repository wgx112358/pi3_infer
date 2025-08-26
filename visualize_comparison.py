import os
import numpy as np

# --- Configuration ---
ANNOTATIONS_DIR = "/inspire/hdd/global_user/zhangkaipeng-24043/lichuanhao/dataset/sekai-real-walking-hq/megasam_outputs"
# ANNOTATIONS_DIR = "/inspire/hdd/global_user/zhangkaipeng-24043/lichuanhao/dataset/sekai-real-walking-hq/annotations"
TASK1_OUTPUT_DIR = "output/full_videos/camera_poses"
TASK2_OUTPUT_DIR = "output/sliced_videos/camera_poses"
PLOT_OUTPUT_DIR = "output/comparison_plots"
VIDEO_LIST_FILE = "output/full_videos/selected_videos.txt"

# --- Matplotlib Check ---
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    print("Matplotlib is not installed. Please install it to run this script:")
    print("pip install matplotlib")
    exit()

def get_camera_positions(poses):
    """Extracts the 3D position (translation) from a list of 4x4 pose matrices."""
    # The camera position is the last column of the pose matrix
    return np.array([p[:3, 3] for p in poses])

def align_trajectory(poses_est, poses_gt):
    """
    Aligns an estimated trajectory to a ground truth trajectory using the Umeyama algorithm.
    This finds the optimal similarity transformation (scale, rotation, translation).
    
    Args:
        poses_est (np.ndarray): Array of estimated 4x4 poses.
        poses_gt (np.ndarray): Array of ground truth 4x4 poses.
    
    Returns:
        np.ndarray: The aligned 3D positions of the estimated trajectory.
    """
    # Extract 3D positions
    pos_est = get_camera_positions(poses_est)
    pos_gt = get_camera_positions(poses_gt)

    # Use the minimum length for comparison to handle potential frame count mismatches
    min_len = min(len(pos_est), len(pos_gt))
    pos_est = pos_est[:min_len]
    pos_gt = pos_gt[:min_len]

    # Center the trajectories to their centroids
    mu_est = pos_est.mean(axis=0)
    mu_gt = pos_gt.mean(axis=0)
    pos_est_centered = pos_est - mu_est
    pos_gt_centered = pos_gt - mu_gt

    # Compute the covariance matrix
    H = pos_est_centered.T @ pos_gt_centered

    # Find the optimal rotation using Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Correct for improper rotations (reflections)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        # If we have a reflection, the scale must be adjusted as well.
        # This is done by negating the smallest singular value.
        S[-1] = -S[-1]

    # Compute the optimal scale
    var_est = np.sum(pos_est_centered**2)
    s = np.sum(S) / var_est if var_est > 1e-6 else 1.0

    # Compute the optimal translation
    t = mu_gt - s * R @ mu_est

    # Apply the transformation
    aligned_pos = s * (R @ pos_est.T).T + t
    
    return aligned_pos

def stitch_sliced_poses(poses_part1, poses_part2, poses_part3):
    """
    Stitches three separate pose trajectories into a single continuous one.
    The goal is to create a single smooth trajectory by chaining the segments.
    """
    # Start with the first part of the trajectory
    stitched_poses = list(poses_part1)

    # --- Chain Part 2 to the end of Part 1 ---
    # Transformation to align the start of part 2 with the end of part 1
    transform_p1_end = stitched_poses[-1] 
    transform_p2_start_inv = np.linalg.inv(poses_part2[0])
    
    # Apply the transformation to each pose in the second part
    for p in poses_part2:
        # p_new = transform_p1_end @ (transform_p2_start_inv @ p)
        p_new = transform_p1_end.dot(transform_p2_start_inv.dot(p))
        stitched_poses.append(p_new)

    # --- Chain Part 3 to the end of the newly stitched Part 2 ---
    # Transformation to align the start of part 3 with the new end of part 2
    transform_p2_end = stitched_poses[-1]
    transform_p3_start_inv = np.linalg.inv(poses_part3[0])
    
    # Apply the transformation to each pose in the third part
    for p in poses_part3:
        # p_new = transform_p2_end @ (transform_p3_start_inv @ p)
        p_new = transform_p2_end.dot(transform_p3_start_inv.dot(p))
        stitched_poses.append(p_new)
        
    return np.array(stitched_poses)

def get_videos_to_process():
    """
    Gets the list of video basenames by reading the saved list file.
    """
    try:
        with open(VIDEO_LIST_FILE, 'r') as f:
            # Read lines and strip any whitespace/newlines
            video_basenames = [line.strip().replace('.mp4', '') for line in f if line.strip()]
        return sorted(video_basenames)
    except FileNotFoundError:
        print(f"Error: Video list file not found at '{VIDEO_LIST_FILE}'")
        print("Please run the script for Task 1 first to generate this file.")
        return None

def main():
    """
    Main function to load data, align trajectories, and plot the comparisons.
    """
    # --- 1. Setup ---
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    video_basenames = get_videos_to_process()
    if not video_basenames:
        print("Could not find any processed videos. Exiting.")
        return

    print(f"Found {len(video_basenames)} videos to visualize.")

    # --- 2. Process each video ---
    total_videos = len(video_basenames)
    for i, basename in enumerate(video_basenames):
        print(f"\n[Task 3: Progress {i+1}/{total_videos}] Visualizing: {basename}")
        
        try:
            # --- 2a. Load all data sources ---
            # Ground Truth
            gt_data = np.load(os.path.join(ANNOTATIONS_DIR, f"{basename}.npz"))
            # The ground truth data 'cam_c2w' is already in the camera-to-world format,
            # which matches the model's output. No inversion is needed.
            poses_gt = gt_data['cam_c2w']
            
            # Full Video Poses (Task 1)
            full_video_data = np.load(os.path.join(TASK1_OUTPUT_DIR, f"{basename}_poses.npz"))
            poses_full = full_video_data['camera_poses']

            # Sliced Video Poses (Task 2)
            poses_part1 = np.load(os.path.join(TASK2_OUTPUT_DIR, f"{basename}_part1_poses.npz"))['camera_poses']
            poses_part2 = np.load(os.path.join(TASK2_OUTPUT_DIR, f"{basename}_part2_poses.npz"))['camera_poses']
            poses_part3 = np.load(os.path.join(TASK2_OUTPUT_DIR, f"{basename}_part3_poses.npz"))['camera_poses']

        except FileNotFoundError as e:
            print(f"  - Could not load all required .npz files for '{basename}'. Skipping. Missing file: {e.filename}")
            continue

        # --- 2b. Stitch poses ---
        poses_stitched = stitch_sliced_poses(poses_part1, poses_part2, poses_part3)
            
        # --- 2c. Correctly Sample Ground Truth for Fair Alignment ---
        # The model was run with an interval (likely 10), so its trajectory is sparse.
        # We must align it to a similarly sparse subsample of the dense ground truth.
        
        gt_len = len(poses_gt)
        full_len = len(poses_full)
        
        if full_len == 0:
            print("  - Error: Full video trajectory has zero length. Skipping.")
            continue
            
        # Infer the sampling interval that was used during inference. Should be ~10.
        sampling_interval = round(gt_len / full_len)
        if sampling_interval < 1:
            sampling_interval = 1 # Avoid interval of 0

        print("  - Trajectory Information:")
        print(f"    - GT (Full):      {gt_len} poses")
        print(f"    - Model (Full):   {full_len} poses")
        print(f"    - Model (Sliced): {len(poses_stitched)} poses")
        print(f"    - Inferred GT Sampling Interval for Alignment: {sampling_interval}")

        # Create the downsampled GT trajectory for alignment purposes
        poses_gt_for_alignment = poses_gt[::sampling_interval]

        # --- 2d. Align Trajectories ---
        print("  - Aligning trajectories using downsampled GT...")
        
        # Align both estimated trajectories to the *sampled* ground truth
        pos_full_aligned = align_trajectory(poses_full, poses_gt_for_alignment)
        pos_stitched_aligned = align_trajectory(poses_stitched, poses_gt_for_alignment)
        
        # Get the full, dense ground truth positions for plotting
        pos_gt_dense = get_camera_positions(poses_gt)
        
        # --- 2e. Plotting ---
        print("  - Generating plot...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 为了更直观的可视化，我们将 OpenCV 坐标系 (X-右, Y-下, Z-前)
        # 映射到更常见的 3D 绘图坐标系 (X-右, Y-前, Z-上)
        # 映射关系: plot_X = data_X, plot_Y = data_Z, plot_Z = -data_Y
        ax.plot(pos_gt_dense[:, 0], pos_gt_dense[:, 2], -pos_gt_dense[:, 1], '-', label='Ground Truth (Full)', color='blue', linewidth=2, alpha=0.7)
        ax.plot(pos_full_aligned[:, 0], pos_full_aligned[:, 2], -pos_full_aligned[:, 1], 'o-', label='Full Video (Aligned)', color='red', markersize=2, alpha=0.8)
        ax.plot(pos_stitched_aligned[:, 0], pos_stitched_aligned[:, 2], -pos_stitched_aligned[:, 1], 'o-', label='Sliced Video (Stitched & Aligned)', color='green', markersize=2, alpha=0.8)
        
        ax.set_title(f'Camera Trajectory Comparison: {basename}', fontsize=16)
        ax.set_xlabel('X')
        ax.set_ylabel('Y (Forward)')
        ax.set_zlabel('Z (Up)')
        ax.legend()
        ax.view_init(elev=20, azim=-60)
        ax.set_aspect('equal', 'box')
        
        plot_path = os.path.join(PLOT_OUTPUT_DIR, f"{basename}_comparison.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig) # Close the figure to free memory

    print(f"\n--- Task 3 Finished ---")
    print(f"All comparison plots have been saved in: {PLOT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

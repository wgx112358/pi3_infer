import os
import numpy as np

# --- Configuration ---
ANNOTATIONS_DIR = "/inspire/hdd/global_user/zhangkaipeng-24043/lichuanhao/dataset/sekai-real-walking-hq/megasam_outputs"
# ANNOTATIONS_DIR = "/inspire/hdd/global_user/zhangkaipeng-24043/lichuanhao/dataset/sekai-real-walking-hq/annotations"
TASK1_OUTPUT_DIR = "output/full_videos/camera_poses"
TASK2_OUTPUT_DIR = "output/sliced_videos/camera_poses"
PLOT_OUTPUT_DIR = "output/comparison_plotsv2"
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

def compute_umeyama_transform(poses_est, poses_ref):
    """
    Computes the optimal similarity transformation (s, R, t) to align
    poses_est to poses_ref using the Umeyama algorithm.
    This function finds the scale, rotation, and translation that minimizes
    the distance between the two sets of corresponding points.
    
    Returns:
        s (float): The scale factor.
        R (np.ndarray): The 3x3 rotation matrix.
        t (np.ndarray): The 3x1 translation vector.
    """
    # Extract 3D positions
    pos_est = get_camera_positions(poses_est)
    pos_ref = get_camera_positions(poses_ref)

    # Use the minimum length for comparison to handle potential frame count mismatches
    min_len = min(len(pos_est), len(pos_ref))
    pos_est = pos_est[:min_len]
    pos_ref = pos_ref[:min_len]

    # Center the trajectories to their centroids
    mu_est = pos_est.mean(axis=0)
    mu_ref = pos_ref.mean(axis=0)
    pos_est_centered = pos_est - mu_est
    pos_ref_centered = pos_ref - mu_ref

    # Compute the covariance matrix
    H = pos_est_centered.T @ pos_ref_centered

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
    t = mu_ref - s * R @ mu_est
    
    return s, R, t

def apply_transform(poses, s, R, t):
    """Applies a pre-computed similarity transformation (s, R, t) to a set of poses."""
    pos = get_camera_positions(poses)
    aligned_pos = s * (R @ pos.T).T + t
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
    Dynamically finds video basenames that have completed processing for both
    full and sliced video tasks by checking for the existence of their output files.
    """
    if not os.path.exists(TASK1_OUTPUT_DIR) or not os.path.exists(TASK2_OUTPUT_DIR):
        print(f"Error: One of the required output directories does not exist.")
        print(f"Checked: '{TASK1_OUTPUT_DIR}' and '{TASK2_OUTPUT_DIR}'")
        return None

    # Get basenames from the full video task outputs
    try:
        full_video_files = os.listdir(TASK1_OUTPUT_DIR)
        # Extract basename, e.g., from "my_video_poses.npz" -> "my_video"
        basenames_task1 = {f.replace('_poses.npz', '') for f in full_video_files if f.endswith('_poses.npz')}
    except FileNotFoundError:
        print(f"Error: Directory not found for full video outputs: '{TASK1_OUTPUT_DIR}'")
        return None

    # Get basenames from the sliced video task outputs
    try:
        sliced_video_files = os.listdir(TASK2_OUTPUT_DIR)
        # Check for part1, part2, and part3 for each basename
        sliced_basenames_complete = set()
        # Create a dictionary to track parts found for each potential basename
        sliced_parts_found = {}
        for f in sliced_video_files:
            if f.endswith('_poses.npz'):
                # Extract basename, e.g., "my_video_part1_poses.npz" -> "my_video"
                if '_part1_' in f:
                    basename = f.replace('_part1_poses.npz', '')
                elif '_part2_' in f:
                    basename = f.replace('_part2_poses.npz', '')
                elif '_part3_' in f:
                    basename = f.replace('_part3_poses.npz', '')
                else:
                    continue # Skip files not matching the part naming convention
                
                if basename not in sliced_parts_found:
                    sliced_parts_found[basename] = set()
                
                if '_part1_' in f:
                    sliced_parts_found[basename].add('part1')
                elif '_part2_' in f:
                    sliced_parts_found[basename].add('part2')
                elif '_part3_' in f:
                    sliced_parts_found[basename].add('part3')

        # A video is considered complete for sliced task if all 3 parts are present
        for basename, parts in sliced_parts_found.items():
            if len(parts) == 3:
                sliced_basenames_complete.add(basename)

    except FileNotFoundError:
        print(f"Error: Directory not found for sliced video outputs: '{TASK2_OUTPUT_DIR}'")
        return None

    # The final list of videos to process is the intersection of the two sets
    videos_to_process = sorted(list(basenames_task1.intersection(sliced_basenames_complete)))
    
    if not videos_to_process:
        print("Warning: No videos were found with complete outputs from both full and sliced processing.")
        print(f"Checked for intersection between outputs in '{TASK1_OUTPUT_DIR}' and '{TASK2_OUTPUT_DIR}'.")

    return videos_to_process


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
            # MegaSAM Poses (Reference)
            megasam_data = np.load(os.path.join(ANNOTATIONS_DIR, f"{basename}.npz"))
            # The MegaSAM data 'cam_c2w' is in camera-to-world format, which matches the model's output.
            poses_megasam = megasam_data['cam_c2w']
            
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
            
        # --- 2c. Align Trajectories to the Reference ---
        # Both Pi3 models and the MegaSAM reference now produce trajectories of the 
        # same length, so we can directly align them without any downsampling.

        print("  - Trajectory Information:")
        print(f"    - MegaSAM (Reference): {len(poses_megasam)} poses")
        print(f"    - Pi3 (Full Video):    {len(poses_full)} poses")
        print(f"    - Pi3 (Sliced Video):  {len(poses_stitched)} poses")
        
        # --- 2d. Align Trajectories ---
        print("  - Aligning Pi3 trajectories to MegaSAM trajectory...")

        # --- Step 1: Internal Alignment (Pi3 Sliced vs. Pi3 Full) ---
        # First, align the stitched trajectory to the full trajectory. This is a critical
        # step because each Pi3 run might start in a different arbitrary coordinate system.
        # We assume the first part of the sliced video corresponds to the start of the full video.
        num_frames_part1 = len(poses_part1)
        # We need to handle cases where part1 might be longer than the full video, though unlikely.
        align_len = min(num_frames_part1, len(poses_full))
        
        # Compute transform to map the sliced coordinate system to the full video's system
        s_internal, R_internal, t_internal = compute_umeyama_transform(
            poses_part1[:align_len], poses_full[:align_len]
        )
        
        # Apply this transform to the *entire* stitched trajectory to get its 3D positions
        # in the coordinate system of the full video.
        pos_stitched_in_full_system = apply_transform(poses_stitched, s_internal, R_internal, t_internal)

        # --- Step 2: External Alignment (Both Pi3 vs. MegaSAM) ---
        # Now that both Pi3 trajectories are effectively in the same coordinate system (that of the
        # full video), compute the transform to align them to the MegaSAM reference, using the 
        # full trajectory as the baseline.
        s_external, R_external, t_external = compute_umeyama_transform(poses_full, poses_megasam)
        
        # Apply this single, external transformation to the full trajectory's poses to get aligned positions.
        pos_full_aligned = apply_transform(poses_full, s_external, R_external, t_external)
        
        # Apply the SAME external transformation to the already-aligned stitched positions.
        # We cannot use apply_transform() here as it expects 4x4 poses, not 3D points, so we do the math directly.
        pos_stitched_aligned = s_external * (R_external @ pos_stitched_in_full_system.T).T + t_external
        
        # Get the full, dense MegaSAM positions for plotting
        pos_megasam_dense = get_camera_positions(poses_megasam)
        
        # --- 2e. Plotting ---
        print("  - Generating plot...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 为了更直观的可视化，我们将 OpenCV 坐标系 (X-右, Y-下, Z-前)
        # 映射到更常见的 3D 绘图坐标系 (X-右, Y-前, Z-上)
        # 映射关系: plot_X = data_X, plot_Y = data_Z, plot_Z = -data_Y
        
        # MegaSAM: solid, thicker line to serve as a clear reference
        ax.plot(pos_megasam_dense[:, 0], pos_megasam_dense[:, 2], -pos_megasam_dense[:, 1], 
                label='MegaSAM (Full)', color='blue', linestyle='-', linewidth=2.5, alpha=0.7)

        # Pi3 Full Video: dashed line with circle markers
        ax.plot(pos_full_aligned[:, 0], pos_full_aligned[:, 2], -pos_full_aligned[:, 1], 
                label='Pi3 (Full Video - Aligned)', color='red', linestyle='--', linewidth=1.5, 
                marker='o', markersize=4, markevery=20, alpha=0.8)

        # Pi3 Sliced Video: dotted line with triangle markers
        ax.plot(pos_stitched_aligned[:, 0], pos_stitched_aligned[:, 2], -pos_stitched_aligned[:, 1], 
                label='Pi3 (Sliced Video - Aligned)', color='green', linestyle=':', linewidth=1.5, 
                marker='^', markersize=4, markevery=20, alpha=0.8)
        
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

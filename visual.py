import os
import numpy as np
import glob

# --- Configuration ---
# 请根据您的实际情况修改 `MAGASAM_DIR` 的路径
MAGASAM_DIR = "output/magasam/camera_poses"
PI3_FULL_DIR = "output/full_videos/camera_poses"
PI3_SLICED_DIR = "output/sliced_videos/camera_poses"
VIDEO_LIST_FILE = "output/full_videos/selected_videos.txt"
RESULTS_FILE = "output/trajectory_discrepancy_report.txt"

def get_video_list():
    """
    从文件中加载视频列表。
    """
    try:
        with open(VIDEO_LIST_FILE, 'r') as f:
            video_basenames = [line.strip() for line in f if line.strip()]
        print(f"Successfully loaded {len(video_basenames)} videos from '{VIDEO_LIST_FILE}'")
        return video_basenames
    except FileNotFoundError:
        print(f"Error: Video list file not found at '{VIDEO_LIST_FILE}'")
        print("Please ensure '1_generate_list.py' has been run successfully.")
        return None

def procrustes_analysis(A, B):
    """
    使用Procrustes分析对齐两个轨迹 A 和 B (仅旋转和平移)。
    返回对齐后的轨迹 B' 和两条轨迹之间的平均距离误差。
    """
    if A.shape != B.shape:
        raise ValueError("Input trajectories must have the same shape")

    # 1. 中心化
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B

    # 2. 计算最优旋转矩阵 R
    H = A_centered.T @ B_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 确保 R 是一个正常的旋转矩阵 (处理反射情况)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 3. 对齐轨迹 B (不进行缩放)
    # 通过移除缩放因子s，我们执行一个刚体变换(SE(3)对齐)。
    # 这对于捕捉尺度漂移导致的误差更敏感，因为算法无法通过缩小轨迹来“隐藏”误差。
    B_aligned = (B_centered @ R) + centroid_A
    
    # 4. 计算误差
    error = np.linalg.norm(A - B_aligned, axis=1).mean()
    
    return B_aligned, error

def load_poses(npz_path):
    """
    从 .npz 文件加载相机位姿。
    假设位姿数据存储在 'camera_poses' 键下。
    """
    try:
        data = np.load(npz_path)
        # 假设 npz 文件中的 key 是 'camera_poses'
        # 如果不是，您需要修改这里的 key
        return data['camera_poses']
    except (FileNotFoundError, KeyError) as e:
        # print(f"  - Could not load or find key in {npz_path}: {e}")
        return None

def get_trajectory(poses):
    """
    从 4x4 位姿矩阵中提取平移向量作为轨迹。
    """
    return poses[:, :3, 3]

def main():
    """
    主函数：加载、比较并排序轨迹误差。
    """
    video_basenames = get_video_list()
    if not video_basenames:
        return

    video_reports = []

    for video_name in video_basenames:
        base_name = os.path.splitext(video_name)[0]
        print(f"\n--- Processing: {base_name} ---")

        # --- 1. 加载所有可能的轨迹 ---
        trajectories = {}

        # a) Magasam
        magasam_path = os.path.join(MAGASAM_DIR, f"{base_name}_poses.npz")
        magasam_poses = load_poses(magasam_path)
        if magasam_poses is not None:
            trajectories['magasam'] = get_trajectory(magasam_poses)
            print(f"  - Loaded Magasam trajectory ({len(trajectories['magasam'])} points)")

        # b) Pi3 Full
        pi3_full_path = os.path.join(PI3_FULL_DIR, f"{base_name}_poses.npz")
        pi3_full_poses = load_poses(pi3_full_path)
        if pi3_full_poses is not None:
            trajectories['pi3_full'] = get_trajectory(pi3_full_poses)
            print(f"  - Loaded Pi3 Full trajectory ({len(trajectories['pi3_full'])} points)")
        
        # c) Pi3 Sliced (拼接)
        slice_parts = []
        for i in range(1, 4):
            slice_path = os.path.join(PI3_SLICED_DIR, f"{base_name}_part{i}_poses.npz")
            slice_poses = load_poses(slice_path)
            if slice_poses is not None:
                slice_parts.append(get_trajectory(slice_poses))
        
        if len(slice_parts) == 3:
            trajectories['pi3_sliced'] = np.concatenate(slice_parts, axis=0)
            print(f"  - Loaded and concatenated Pi3 Sliced trajectory ({len(trajectories['pi3_sliced'])} points)")
        elif slice_parts:
             print(f"  - Warning: Found only {len(slice_parts)}/3 parts for sliced video. Skipping.")


        # --- 2. 两两比较轨迹并找出最大误差 ---
        keys = list(trajectories.keys())
        
        current_video_errors = {}

        if len(keys) >= 2:
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    key1, key2 = keys[i], keys[j]
                    traj1, traj2 = trajectories[key1], trajectories[key2]

                    # 确保比较名称顺序一致
                    sorted_keys = sorted([key1, key2])
                    comparison_name = f"{sorted_keys[0]}_vs_{sorted_keys[1]}"

                    # 截断到相同长度
                    min_len = min(len(traj1), len(traj2))
                    if min_len == 0:
                        continue
                    
                    traj1_truncated = traj1[:min_len]
                    traj2_truncated = traj2[:min_len]

                    _, error = procrustes_analysis(traj1_truncated, traj2_truncated)
                    
                    print(f"  - Comparison [{comparison_name}]: Mean Error = {error:.4f}")
                    current_video_errors[comparison_name] = error
        
        if not current_video_errors:
            print("  - Not enough trajectories to compare. Skipping.")
            continue
        
        max_error = max(current_video_errors.values())
        video_reports.append({
            'video': base_name,
            'max_error': max_error,
            'all_errors': current_video_errors
        })


    # --- 3. 排序并输出结果 ---
    if not video_reports:
        print("\nNo comparisons were made. Please check your output directories.")
        return
        
    sorted_reports = sorted(video_reports, key=lambda x: x['max_error'], reverse=True)

    # --- 4. 准备报告内容 ---
    report_lines = []
    header1 = "--- Trajectory Discrepancy Report (Sorted by Max Error per Video) ---"
    header2 = "-" * 80
    header3 = f"{'Rank':<5} {'Max Error':<12} {'Video':<40}"
    
    report_lines.append(header1)
    report_lines.append(header2)
    report_lines.append(header3)
    report_lines.append(header2)

    for i, report in enumerate(sorted_reports):
        # 主行
        main_line = f"{i+1:<5} {report['max_error']:<12.4f} {report['video']:<40}"
        report_lines.append(main_line)
        
        # 详细的对比行
        sorted_individual_errors = sorted(report['all_errors'].items(), key=lambda item: item[1], reverse=True)
        for comparison, error in sorted_individual_errors:
            highlight = " (*)" if error == report['max_error'] else ""
            detail_line = f"      - {comparison:<35} Error: {error:.4f}{highlight}"
            report_lines.append(detail_line)
        report_lines.append("") # 添加空行以分隔视频

    # --- 5. 打印到终端并保存到文件 ---
    report_string = "\n".join(report_lines)
    print("\n\n" + report_string)

    # 准备输出目录
    output_dir = os.path.dirname(RESULTS_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(RESULTS_FILE, 'w') as f:
        f.write(report_string + "\n")
    
    print(f"\n--- Report successfully saved to: {RESULTS_FILE} ---")


if __name__ == "__main__":
    main()

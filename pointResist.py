import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ガウシャン座標とCT座標のデータ
gaussian_points = np.array([
    [11.7, 130.08, -46.53],
    [-12.89, 105.83, -40.85],
    [-8.45, 109.21, -162.38],
    [-8, 83.3, -161.6]
])

ct_points = np.array([
    [5.58, 330.97, 1261.75],
    [-95.98, 227.46, 1287.05],
    [-84.47, 240, 780.17],
    [-82.52, 129.31, 782.49]
])

def calculate_point_distances(source, target):
    """2つの点群間の各点の距離を計算"""
    distances = np.linalg.norm(source - target, axis=1)
    return distances

def visualize_distances(distances, title="Point Distances"):
    """距離のヒストグラムを表示"""
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=20, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel("Distance (mm)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"distance_histogram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()

def calculate_scale_factor(source, target):
    """点群間の距離比から最適なスケールファクターを計算"""
    # 各点群の重心を計算
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    
    # 重心からの距離を計算
    source_distances = np.linalg.norm(source - source_centroid, axis=1)
    target_distances = np.linalg.norm(target - target_centroid, axis=1)
    
    # 距離の比率の中央値を取る（外れ値に対してロバスト）
    scale_ratios = target_distances / source_distances
    optimal_scale = np.median(scale_ratios)
    
    return optimal_scale

def estimate_initial_threshold(source, target):
    """点群間の初期距離に基づいてICPの閾値を推定"""
    # 各点群の重心を計算
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    
    # 重心間の距離の10%を閾値として使用
    initial_distance = np.linalg.norm(source_centroid - target_centroid)
    return initial_distance * 0.1

# ================================================
# 1. 初期位置合わせ（重心を原点に）
# ================================================
source_centroid = np.mean(gaussian_points, axis=0)
target_centroid = np.mean(ct_points, axis=0)

centered_gaussian = gaussian_points - source_centroid
centered_ct = ct_points - target_centroid

# ================================================
# 2. スケール補正（自動計算）
# ================================================
scale_factor = calculate_scale_factor(centered_gaussian, centered_ct)
scaled_gaussian = centered_gaussian * scale_factor
print("計算されたスケールファクター:", scale_factor)
print("スケール補正後のガウシャン点群:\n", scaled_gaussian)

# スケール補正の誤差評価
scale_distances = calculate_point_distances(scaled_gaussian, centered_ct)
print("スケール補正後の点ごとの誤差:", scale_distances)
print("スケール補正後の平均誤差:", np.mean(scale_distances))
print("スケール補正後の標準偏差:", np.std(scale_distances))
visualize_distances(scale_distances, "Scale Correction Distances")

# ================================================
# 3. 回転補正（SVD）
# ================================================
H = np.dot(scaled_gaussian.T, centered_ct)
U, S, Vt = np.linalg.svd(H)
rotation_matrix = np.dot(Vt.T, U.T)

# 回転の反転防止
if np.linalg.det(rotation_matrix) < 0:
    Vt[-1, :] *= -1
    rotation_matrix = np.dot(Vt.T, U.T)

rotated_gaussian = np.dot(scaled_gaussian, rotation_matrix.T)
print("\n回転行列:\n", rotation_matrix)
print("回転補正後のガウシャン点群:\n", rotated_gaussian)

# 回転補正の誤差評価
rotation_distances = calculate_point_distances(rotated_gaussian, centered_ct)
print("回転補正後の点ごとの誤差:", rotation_distances)
print("回転補正後の平均誤差:", np.mean(rotation_distances))
print("回転補正後の標準偏差:", np.std(rotation_distances))
visualize_distances(rotation_distances, "Rotation Distances")

# ================================================
# 4. 平行移動（重心を元の位置に）
# ================================================
translated_gaussian = rotated_gaussian + target_centroid
print("\n最終的な平行移動量:\n", target_centroid)
print("平行移動後のガウシャン点群:\n", translated_gaussian)

# 平行移動の誤差評価
translation_distances = calculate_point_distances(translated_gaussian, ct_points)
print("平行移動後の点ごとの誤差:", translation_distances)
print("平行移動後の平均誤差:", np.mean(translation_distances))
print("平行移動後の標準偏差:", np.std(translation_distances))
visualize_distances(translation_distances, "Translation Distances")

# ================================================
# 5. ICPによる微調整
# ================================================
source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(translated_gaussian)

target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(ct_points)

# 適応的な閾値の設定
threshold = estimate_initial_threshold(translated_gaussian, ct_points)
print(f"\n推定されたICP閾値: {threshold:.2f}mm")

reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold,
    np.identity(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)

print("ICPの収束結果:\n", reg_p2p)
print("最適変換行列:\n", reg_p2p.transformation)

# ================================================
# 6. 最終変換と誤差計算
# ================================================
source.transform(reg_p2p.transformation)

# 最終的な誤差評価
final_distances = np.linalg.norm(
    np.asarray(target.points) - np.asarray(source.points), axis=1)
print("\n最終的な点ごとの誤差:", final_distances)
print("最終的な平均誤差:", np.mean(final_distances))
print("最終的な標準偏差:", np.std(final_distances))
visualize_distances(final_distances, "Final Registration Distances")

# 結果の保存
with open(f"registration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
    f.write("Registration Results\n")
    f.write("===================\n\n")
    f.write(f"Calculated scale factor: {scale_factor:.6f}\n")
    f.write(f"Initial centroid translation:\n{source_centroid}\n")
    f.write(f"Rotation matrix:\n{rotation_matrix}\n")
    f.write(f"Final translation:\n{target_centroid}\n")
    f.write(f"ICP transformation matrix:\n{reg_p2p.transformation}\n\n")
    f.write("Error Analysis\n")
    f.write("=============\n")
    f.write(f"Final mean error: {np.mean(final_distances):.3f} mm\n")
    f.write(f"Final std deviation: {np.std(final_distances):.3f} mm\n")
    f.write(f"Point-wise errors: {final_distances}\n")

# 最終的な可視化
o3d.visualization.draw_geometries([
    source.paint_uniform_color([0, 0, 1]),  # 青
    target.paint_uniform_color([0, 1, 0])   # 緑
], window_name="最終結果")

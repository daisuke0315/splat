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

# ================================================
# 1. ガウシャンのスケール補正（固定値4.2）
# ================================================
scale_factors = 4.2
scaled_gaussian = gaussian_points * scale_factors
print("スケールファクター (固定):", scale_factors)
print("スケール補正後のガウシャン点群:\n", scaled_gaussian)

# スケール補正の誤差評価
scale_distances = calculate_point_distances(scaled_gaussian, ct_points)
print("スケール補正後の点ごとの誤差:", scale_distances)
print("スケール補正後の平均誤差:", np.mean(scale_distances))
print("スケール補正後の標準偏差:", np.std(scale_distances))
visualize_distances(scale_distances, "Scale Correction Distances")

# スケール補正結果の可視化
scaled_source = o3d.geometry.PointCloud()
scaled_source.points = o3d.utility.Vector3dVector(scaled_gaussian)
scaled_source.paint_uniform_color([1, 0, 0])  # 赤

target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(ct_points)
target.paint_uniform_color([0, 1, 0])  # 緑

o3d.visualization.draw_geometries([scaled_source, target], window_name="スケール補正")

# ================================================
# 2. 1番目の点を基準に平行移動
# ================================================
translation = ct_points[0] - scaled_gaussian[0]
translated_gaussian = scaled_gaussian + translation
print("\n平行移動量 (1番目の点を一致):\n", translation)
print("平行移動後のガウシャン点群:\n", translated_gaussian)

# 平行移動の誤差評価
translation_distances = calculate_point_distances(translated_gaussian, ct_points)
print("平行移動後の点ごとの誤差:", translation_distances)
print("平行移動後の平均誤差:", np.mean(translation_distances))
print("平行移動後の標準偏差:", np.std(translation_distances))
visualize_distances(translation_distances, "Translation Distances")

# 平行移動結果の可視化
translated_source = o3d.geometry.PointCloud()
translated_source.points = o3d.utility.Vector3dVector(translated_gaussian)
translated_source.paint_uniform_color([1, 0, 0])  # 赤

o3d.visualization.draw_geometries([translated_source, target], window_name="平行移動")

# ================================================
# 3. 回転補正（最小二乗法）
# ================================================
H = np.dot((translated_gaussian - np.mean(translated_gaussian, axis=0)).T,
           (ct_points - np.mean(ct_points, axis=0)))
U, S, Vt = np.linalg.svd(H)
rotation_matrix = np.dot(Vt.T, U.T)

# 回転の反転防止
if np.linalg.det(rotation_matrix) < 0:
    Vt[-1, :] *= -1
    rotation_matrix = np.dot(Vt.T, U.T)

rotated_gaussian = np.dot(translated_gaussian, rotation_matrix.T)
print("\n回転行列:\n", rotation_matrix)
print("回転補正後のガウシャン点群:\n", rotated_gaussian)

# 回転補正の誤差評価
rotation_distances = calculate_point_distances(rotated_gaussian, ct_points)
print("回転補正後の点ごとの誤差:", rotation_distances)
print("回転補正後の平均誤差:", np.mean(rotation_distances))
print("回転補正後の標準偏差:", np.std(rotation_distances))
visualize_distances(rotation_distances, "Rotation Distances")

# 回転結果の可視化
rotated_source = o3d.geometry.PointCloud()
rotated_source.points = o3d.utility.Vector3dVector(rotated_gaussian)
rotated_source.paint_uniform_color([1, 0, 0])  # 赤

o3d.visualization.draw_geometries([rotated_source, target], window_name="回転補正")

# ================================================
# 4. ICPレジストレーション（位置合わせ）
# ================================================
source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(rotated_gaussian)

threshold = 500.0  # 閾値を適切に設定
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold,
    np.identity(4),  # 初期変換行列は単位行列（なしと同義）
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000)
)

print("\nICPの収束結果:\n", reg_p2p)
print("最適変換行列:\n", reg_p2p.transformation)

# ================================================
# 5. 最終変換と誤差計算
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
    f.write(f"Scale factor: {scale_factors}\n")
    f.write(f"Translation vector: {translation}\n")
    f.write(f"Rotation matrix:\n{rotation_matrix}\n")
    f.write(f"Final transformation matrix:\n{reg_p2p.transformation}\n\n")
    f.write("Error Analysis\n")
    f.write("=============\n")
    f.write(f"Final mean error: {np.mean(final_distances):.3f} mm\n")
    f.write(f"Final std deviation: {np.std(final_distances):.3f} mm\n")
    f.write(f"Point-wise errors: {final_distances}\n")

# 最終的な可視化
o3d.visualization.draw_geometries([
    source.paint_uniform_color([0, 0, 1]),  # 青
    target.paint_uniform_color([0, 1, 0])   # 緑
], window_name="ICP後の結果")

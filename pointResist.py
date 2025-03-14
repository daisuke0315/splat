import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.metrics import mean_squared_error

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

# たわみ補正用の2点
# [x, y, z] の形式で入力
deflection_point1 = np.array([0, -97.42, -531.8])   # 1点目の座標
deflection_point2 = np.array([0, -96.33, -1003.8])  # 2点目の座標

def calculate_point_distances(source, target):
    """2つの点群間の各点の距離を計算"""
    distances = np.linalg.norm(source - target, axis=1)
    return distances

def calculate_relative_distances(points):
    """点群内の相対距離を計算"""
    n_points = len(points)
    distances = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(i+1, n_points):
            distances[i,j] = np.linalg.norm(points[i] - points[j])
            distances[j,i] = distances[i,j]
    return distances

def calculate_distance_preservation_error(source_distances, target_distances):
    """相対距離の保存性を評価"""
    source_normalized = source_distances / np.max(source_distances)
    target_normalized = target_distances / np.max(target_distances)
    error = np.abs(source_normalized - target_normalized)
    return np.mean(error), np.std(error)

def calculate_scale_factor(source, target):
    """点群間の距離比から最適なスケールファクターを計算"""
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_distances = np.linalg.norm(source - source_centroid, axis=1)
    target_distances = np.linalg.norm(target - target_centroid, axis=1)
    scale_ratios = target_distances / source_distances
    optimal_scale = np.median(scale_ratios)
    return optimal_scale

def calculate_deflection_coefficient(point1, point2):
    """
    2点からたわみ係数を計算
    point1: [x1, y1, z1] 1点目の座標
    point2: [x2, y2, z2] 2点目の座標
    戻り値: たわみ係数 (mm/mm)
    """
    dx = point2[0] - point1[0]
    dz = point2[2] - point1[2]
    
    if dx != 0:
        deflection_coefficient = dz / dx
    else:
        deflection_coefficient = 0
        
    return deflection_coefficient

def correct_couch_deflection(points, reference_point1=None, reference_point2=None):
    """
    カウチのたわみを補正
    points: 補正する点群
    reference_point1: たわみ計算用の基準点1 [x1, y1, z1]
    reference_point2: たわみ計算用の基準点2 [x2, y2, z2]
    """
    if reference_point1 is None or reference_point2 is None:
        reference_point1 = points[0]
        reference_point2 = points[-1]
    
    deflection_coefficient = calculate_deflection_coefficient(reference_point1, reference_point2)
    print(f"   - 基準点1: X={reference_point1[0]:.1f}, Z={reference_point1[2]:.1f}")
    print(f"   - 基準点2: X={reference_point2[0]:.1f}, Z={reference_point2[2]:.1f}")
    print(f"   - 計算されたたわみ係数: {deflection_coefficient:.6f} mm/mm")
    
    corrected_points = points.copy()
    x_offset = reference_point1[0]
    corrected_points[:, 2] -= deflection_coefficient * (corrected_points[:, 0] - x_offset)
    
    # 補正前後の差分を表示
    print("\n   - 補正前後のZ座標の変化:")
    for i in range(len(points)):
        diff = corrected_points[i, 2] - points[i, 2]
        print(f"     点{i+1}: {diff:.3f} mm")
    
    return corrected_points

# ================================================
# メイン処理
# ================================================
print("\n=== 点群位置合わせプロセス開始 ===")

# ================================================
# 1. カウチのたわみ補正
# ================================================
print("\n1. カウチのたわみ補正")
print("   - CT点群に対してたわみ補正を適用")

ct_points_original = ct_points.copy()
ct_points = correct_couch_deflection(ct_points, deflection_point1, deflection_point2)

# ================================================
# 2. 初期評価
# ================================================
print("\n2. 初期評価")
print("   - 変換前の相対距離の保存性を計算")
initial_source_distances = calculate_relative_distances(gaussian_points)
initial_target_distances = calculate_relative_distances(ct_points)
initial_mean_error, initial_std_error = calculate_distance_preservation_error(
    initial_source_distances, initial_target_distances)
print(f"   - 平均誤差: {initial_mean_error:.6f}")
print(f"   - 標準偏差: {initial_std_error:.6f}")

# ================================================
# 3. 重心位置合わせ
# ================================================
print("\n3. 重心位置合わせ")
print("   - ガウシャン点群と CT点群の重心を計算")
source_centroid = np.mean(gaussian_points, axis=0)
target_centroid = np.mean(ct_points, axis=0)
centered_gaussian = gaussian_points - source_centroid
centered_ct = ct_points - target_centroid
print("   - 両点群を重心中心に移動")

# ================================================
# 4. スケール補正
# ================================================
print("\n4. スケール補正")
print("   - 相対距離に基づくスケール係数の計算")
scale_factor = calculate_scale_factor(centered_gaussian, centered_ct)
scaled_gaussian = centered_gaussian * scale_factor
print(f"   - 計算されたスケール係数: {scale_factor:.6f}")

# ================================================
# 5. 回転補正
# ================================================
print("\n5. 回転補正（SVD法）")
print("   - 最適な回転行列を計算")
H = np.dot(scaled_gaussian.T, centered_ct)
U, S, Vt = np.linalg.svd(H)
rotation_matrix = np.dot(Vt.T, U.T)

if np.linalg.det(rotation_matrix) < 0:
    print("   - 右手系の保持のため行列を調整")
    Vt[-1, :] *= -1
    rotation_matrix = np.dot(Vt.T, U.T)

r = Rotation.from_matrix(rotation_matrix)
euler_angles = r.as_euler('xyz', degrees=True)
print(f"   - 推定された回転角度 (xyz, 度): {euler_angles}")

rotated_gaussian = np.dot(scaled_gaussian, rotation_matrix.T)

# ================================================
# 6. 最終平行移動
# ================================================
print("\n6. 最終平行移動")
print("   - 重心位置に基づく平行移動を適用")
translated_gaussian = rotated_gaussian + target_centroid

# ================================================
# 7. ICPによる微調整
# ================================================
print("\n7. ICPによる微調整")
print("   - Open3Dを使用したICPアルゴリズムの適用")
source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(translated_gaussian)

target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(ct_points)

threshold = 5.0
print(f"   - 使用するICP閾値: {threshold:.2f}mm")

reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold,
    np.identity(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)

source.transform(reg_p2p.transformation)
final_points = np.asarray(source.points)

# 最終的な誤差評価
final_distances = calculate_point_distances(final_points, ct_points)
print("\n   - 点ごとの最終誤差:")
for i, dist in enumerate(final_distances):
    print(f"     点{i+1}: {dist:.3f} mm")
print(f"   - 最終平均誤差: {np.mean(final_distances):.3f} ± {np.std(final_distances):.3f} mm")

print("\n=== 処理完了 ===")

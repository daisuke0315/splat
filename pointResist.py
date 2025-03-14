import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial.transform import Rotation
from scipy.stats import norm, shapiro
from sklearn.metrics import mean_squared_error
import pandas as pd

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
    # 距離行列の正規化
    source_normalized = source_distances / np.max(source_distances)
    target_normalized = target_distances / np.max(target_distances)
    
    # 相対誤差を計算
    error = np.abs(source_normalized - target_normalized)
    return np.mean(error), np.std(error)

def calculate_procrustes_error(source, target):
    """プロクルステス距離を計算"""
    # 重心を原点に
    source_centered = source - np.mean(source, axis=0)
    target_centered = target - np.mean(target, axis=0)
    
    # フロベニウスノルムで正規化
    source_norm = np.sqrt(np.sum(source_centered**2))
    target_norm = np.sqrt(np.sum(target_centered**2))
    source_normalized = source_centered / source_norm
    target_normalized = target_centered / target_norm
    
    # プロクルステス距離を計算
    error = np.sqrt(np.sum((source_normalized - target_normalized)**2))
    return error

def visualize_distances(distances, title="Point Distances"):
    """距離のヒストグラムを表示と正規性の評価"""
    plt.figure(figsize=(10, 6))
    
    # ヒストグラムの描画
    n, bins, patches = plt.hist(distances, bins=20, density=True, 
                              color='blue', alpha=0.7, label='Observed')
    
    # 正規分布のフィッティング
    mu, std = norm.fit(distances)
    x = np.linspace(min(distances), max(distances), 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r-', lw=2, label=f'Normal fit\nμ={mu:.2f}, σ={std:.2f}')
    
    plt.title(title)
    plt.xlabel("Distance (mm)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"distance_histogram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()
    
    # Shapiro-Wilk検定
    stat, p_value = shapiro(distances)
    return p_value

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

def calculate_error_metrics(distances):
    """詳細な誤差指標を計算"""
    metrics = {
        'mean': np.mean(distances),
        'std': np.std(distances),
        'median': np.median(distances),
        'max': np.max(distances),
        'min': np.min(distances),
        'p95': np.percentile(distances, 95),
        'p99': np.percentile(distances, 99),
        'rmse': np.sqrt(mean_squared_error(np.zeros_like(distances), distances))
    }
    return metrics

def visualize_error_distribution(distances, title="Error Distribution"):
    """誤差分布の詳細な可視化"""
    plt.figure(figsize=(12, 8))
    
    # メインのヒストグラム
    plt.subplot(2, 1, 1)
    n, bins, patches = plt.hist(distances, bins='auto', density=True, 
                              color='blue', alpha=0.7, label='Observed')
    
    # 正規分布フィッティング
    mu, std = norm.fit(distances)
    x = np.linspace(min(distances), max(distances), 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r-', lw=2, label=f'Normal fit\nμ={mu:.2f}, σ={std:.2f}')
    
    plt.title(f"{title}\nShapiro-Wilk test p-value: {shapiro(distances)[1]:.4f}")
    plt.xlabel("Error (mm)")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()
    
    # Q-Qプロット
    plt.subplot(2, 1, 2)
    from scipy.stats import probplot
    probplot(distances, dist="norm", plot=plt)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"error_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()

def perform_icp_with_scale(source_points, target_points, threshold=5.0, max_iteration=2000):
    """スケール最適化を含むICPレジストレーション"""
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)
    
    # 初期位置合わせ（重心を一致）
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    init_translation = target_centroid - source_centroid
    
    # 初期変換行列の作成
    init_transform = np.eye(4)
    init_transform[:3, 3] = init_translation
    
    # ICPレジストレーション（スケール最適化含む）
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationForScale(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    
    # 変換後の点群を取得
    source.transform(reg_p2p.transformation)
    transformed_points = np.asarray(source.points)
    
    return transformed_points, reg_p2p

def analyze_registration(source, target, transformed, transformation):
    """レジストレーション結果の詳細な分析"""
    # 誤差計算
    distances = np.linalg.norm(transformed - target, axis=1)
    metrics = calculate_error_metrics(distances)
    
    # スケール係数の抽出
    scale = np.linalg.norm(transformation[:3, :3], ord='fro') / np.sqrt(3)
    
    # 回転行列の抽出と角度計算
    rotation_matrix = transformation[:3, :3] / scale
    r = Rotation.from_matrix(rotation_matrix)
    euler_angles = r.as_euler('xyz', degrees=True)
    
    # 並進ベクトルの抽出
    translation = transformation[:3, 3]
    
    return {
        'error_metrics': metrics,
        'scale': scale,
        'euler_angles': euler_angles,
        'translation': translation,
        'distances': distances
    }

def save_detailed_results(results, filename):
    """詳細な結果をファイルに保存"""
    with open(filename, 'w') as f:
        f.write("Registration Analysis Results\n")
        f.write("===========================\n\n")
        
        f.write("Error Metrics\n")
        f.write("-------------\n")
        metrics = results['error_metrics']
        for key, value in metrics.items():
            f.write(f"{key}: {value:.3f} mm\n")
        
        f.write("\nTransformation Parameters\n")
        f.write("------------------------\n")
        f.write(f"Scale factor: {results['scale']:.6f}\n")
        f.write(f"Euler angles (xyz, degrees): {results['euler_angles']}\n")
        f.write(f"Translation vector: {results['translation']}\n")
        
        # 詳細な統計情報
        distances = results['distances']
        df = pd.DataFrame(distances, columns=['Error'])
        f.write("\nDetailed Statistics\n")
        f.write("-----------------\n")
        f.write(df.describe().to_string())

def calculate_deflection_coefficient(point1, point2):
    """
    2点からたわみ係数を計算
    point1: [x1, y1, z1] 1点目の座標
    point2: [x2, y2, z2] 2点目の座標
    戻り値: たわみ係数 (mm/mm)
    """
    # X座標の差分
    dx = point2[0] - point1[0]
    # Z座標の差分
    dz = point2[2] - point1[2]
    
    # たわみ係数を計算 (dz/dx)
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
        # デフォルトでは最初と最後の点を使用
        reference_point1 = points[0]
        reference_point2 = points[-1]
    
    # たわみ係数を計算
    deflection_coefficient = calculate_deflection_coefficient(reference_point1, reference_point2)
    print(f"   - 基準点1: X={reference_point1[0]:.1f}, Z={reference_point1[2]:.1f}")
    print(f"   - 基準点2: X={reference_point2[0]:.1f}, Z={reference_point2[2]:.1f}")
    print(f"   - 計算されたたわみ係数: {deflection_coefficient:.6f} mm/mm")
    
    corrected_points = points.copy()
    # X座標に基づいてZ座標を補正
    x_offset = reference_point1[0]  # 基準点1のX座標を基準にする
    corrected_points[:, 2] -= deflection_coefficient * (corrected_points[:, 0] - x_offset)
    return corrected_points

def visualize_deflection_correction(points, corrected_points):
    """たわみ補正の可視化"""
    plt.figure(figsize=(10, 6))
    plt.scatter(points[:, 0], points[:, 2], label='補正前', c='blue')
    plt.scatter(corrected_points[:, 0], corrected_points[:, 2], label='補正後', c='red')
    
    # 補正前後の線形フィット
    coef_orig = np.polyfit(points[:, 0], points[:, 2], 1)
    coef_corr = np.polyfit(corrected_points[:, 0], corrected_points[:, 2], 1)
    
    x_range = np.linspace(min(points[:, 0]), max(points[:, 0]), 100)
    plt.plot(x_range, np.polyval(coef_orig, x_range), '--', c='blue', label='補正前の傾向')
    plt.plot(x_range, np.polyval(coef_corr, x_range), '--', c='red', label='補正後の傾向')
    
    plt.xlabel('X位置 (mm)')
    plt.ylabel('Z位置 (mm)')
    plt.title('カウチのたわみ補正')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"deflection_correction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()

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
initial_source_distances = calculate_relative_distances(gaussian_points)
initial_target_distances = calculate_relative_distances(ct_points)
initial_mean_error, initial_std_error = calculate_distance_preservation_error(
    initial_source_distances, initial_target_distances)

# ================================================
# 3. 重心位置合わせ
# ================================================
print("\n3. 重心位置合わせ")
source_centroid = np.mean(gaussian_points, axis=0)
target_centroid = np.mean(ct_points, axis=0)
centered_gaussian = gaussian_points - source_centroid
centered_ct = ct_points - target_centroid

# ================================================
# 4. スケール補正
# ================================================
print("\n4. スケール補正")
scale_factor = calculate_scale_factor(centered_gaussian, centered_ct)
scaled_gaussian = centered_gaussian * scale_factor

# ================================================
# 5. 回転補正
# ================================================
print("\n5. 回転補正（SVD法）")
H = np.dot(scaled_gaussian.T, centered_ct)
U, S, Vt = np.linalg.svd(H)
rotation_matrix = np.dot(Vt.T, U.T)

if np.linalg.det(rotation_matrix) < 0:
    print("   - 右手系の保持のため行列を調整")
    Vt[-1, :] *= -1
    rotation_matrix = np.dot(Vt.T, U.T)

rotated_gaussian = np.dot(scaled_gaussian, rotation_matrix.T)

# ================================================
# 6. 最終平行移動
# ================================================
print("\n6. 最終平行移動")
translated_gaussian = rotated_gaussian + target_centroid

# ================================================
# 7. ICPによる微調整
# ================================================
print("\n7. ICPによる微調整")
source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(translated_gaussian)

target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(ct_points)

threshold = 5.0
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold,
    np.identity(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)

source.transform(reg_p2p.transformation)
final_points = np.asarray(source.points)

print("\n=== 処理完了 ===")

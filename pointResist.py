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

# ================================================
# 1. 初期評価（変換前の相対距離の保存性）
# ================================================
initial_source_distances = calculate_relative_distances(gaussian_points)
initial_target_distances = calculate_relative_distances(ct_points)
initial_mean_error, initial_std_error = calculate_distance_preservation_error(
    initial_source_distances, initial_target_distances)
print("初期の相対距離保存性:")
print(f"平均誤差: {initial_mean_error:.6f}")
print(f"標準偏差: {initial_std_error:.6f}")

# ================================================
# 2. 重心位置合わせ
# ================================================
source_centroid = np.mean(gaussian_points, axis=0)
target_centroid = np.mean(ct_points, axis=0)
centered_gaussian = gaussian_points - source_centroid
centered_ct = ct_points - target_centroid

# ================================================
# 3. スケール補正（相対距離に基づく）
# ================================================
scale_factor = calculate_scale_factor(centered_gaussian, centered_ct)
scaled_gaussian = centered_gaussian * scale_factor
print("\n計算されたスケールファクター:", scale_factor)

# スケール補正後の相対距離保存性
scaled_distances = calculate_relative_distances(scaled_gaussian)
scale_mean_error, scale_std_error = calculate_distance_preservation_error(
    scaled_distances, initial_target_distances)
print("スケール補正後の相対距離保存性:")
print(f"平均誤差: {scale_mean_error:.6f}")
print(f"標準偏差: {scale_std_error:.6f}")

# ================================================
# 4. 回転補正（SVD）
# ================================================
H = np.dot(scaled_gaussian.T, centered_ct)
U, S, Vt = np.linalg.svd(H)
rotation_matrix = np.dot(Vt.T, U.T)

if np.linalg.det(rotation_matrix) < 0:
    Vt[-1, :] *= -1
    rotation_matrix = np.dot(Vt.T, U.T)

# 回転角度の計算（オイラー角）
r = Rotation.from_matrix(rotation_matrix)
euler_angles = r.as_euler('xyz', degrees=True)
print("\n推定された回転角度 (xyz, 度):", euler_angles)

rotated_gaussian = np.dot(scaled_gaussian, rotation_matrix.T)

# 回転後の相対距離保存性
rotated_distances = calculate_relative_distances(rotated_gaussian)
rotation_mean_error, rotation_std_error = calculate_distance_preservation_error(
    rotated_distances, initial_target_distances)
print("回転補正後の相対距離保存性:")
print(f"平均誤差: {rotation_mean_error:.6f}")
print(f"標準偏差: {rotation_std_error:.6f}")

# ================================================
# 5. 最終平行移動
# ================================================
translated_gaussian = rotated_gaussian + target_centroid

# プロクルステス誤差の計算
procrustes_error = calculate_procrustes_error(translated_gaussian, ct_points)
print(f"\nプロクルステス誤差: {procrustes_error:.6f}")

# ================================================
# 6. ICPによる微調整
# ================================================
source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(translated_gaussian)

target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(ct_points)

threshold = estimate_initial_threshold(translated_gaussian, ct_points)
print(f"\n推定されたICP閾値: {threshold:.2f}mm")

reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold,
    np.identity(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)

# ================================================
# 7. 最終評価
# ================================================
source.transform(reg_p2p.transformation)
final_points = np.asarray(source.points)

# 最終的な誤差評価
final_distances = calculate_point_distances(final_points, ct_points)
print("\n最終的な点ごとの誤差:", final_distances)
print(f"最終的な平均誤差: {np.mean(final_distances):.3f} ± {np.std(final_distances):.3f} mm")

# 正規性の評価
p_value = visualize_distances(final_distances, "Final Registration Distances")
print(f"誤差分布の正規性検定 p値: {p_value:.4f}")

# 結果の保存
with open(f"registration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
    f.write("Registration Results\n")
    f.write("===================\n\n")
    f.write("Initial Analysis\n")
    f.write(f"Initial relative distance preservation error: {initial_mean_error:.6f} ± {initial_std_error:.6f}\n\n")
    
    f.write("Scale Analysis\n")
    f.write(f"Calculated scale factor: {scale_factor:.6f}\n")
    f.write(f"Scale relative distance preservation error: {scale_mean_error:.6f} ± {scale_std_error:.6f}\n\n")
    
    f.write("Rotation Analysis\n")
    f.write(f"Euler angles (xyz, degrees): {euler_angles}\n")
    f.write(f"Rotation matrix:\n{rotation_matrix}\n")
    f.write(f"Rotation relative distance preservation error: {rotation_mean_error:.6f} ± {rotation_std_error:.6f}\n\n")
    
    f.write("Final Analysis\n")
    f.write(f"Procrustes error: {procrustes_error:.6f}\n")
    f.write(f"Final mean error: {np.mean(final_distances):.3f} ± {np.std(final_distances):.3f} mm\n")
    f.write(f"Error distribution normality test p-value: {p_value:.4f}\n")
    f.write(f"Point-wise errors: {final_distances}\n")

# 最終的な可視化
o3d.visualization.draw_geometries([
    source.paint_uniform_color([0, 0, 1]),  # 青
    target.paint_uniform_color([0, 1, 0])   # 緑
], window_name="最終結果")

# メイン処理
print("Registration Analysis Started")
print("===========================")

# ICPレジストレーション（スケール最適化含む）
transformed_points, reg_result = perform_icp_with_scale(
    gaussian_points, ct_points, threshold=5.0)

# 結果の分析
results = analyze_registration(
    gaussian_points, ct_points, transformed_points, reg_result.transformation)

# 結果の表示
print("\nError Metrics:")
for key, value in results['error_metrics'].items():
    print(f"{key}: {value:.3f} mm")

print("\nTransformation Parameters:")
print(f"Scale factor: {results['scale']:.6f}")
print(f"Euler angles (xyz, degrees): {results['euler_angles']}")
print(f"Translation vector: {results['translation']}")

# 誤差分布の可視化
visualize_error_distribution(results['distances'], "Registration Error Distribution")

# 詳細な結果の保存
save_detailed_results(results, 
    f"detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# 最終的な可視化
source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(transformed_points)
target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(ct_points)

o3d.visualization.draw_geometries([
    source.paint_uniform_color([0, 0, 1]),  # 青
    target.paint_uniform_color([0, 1, 0])   # 緑
], window_name="最終結果")

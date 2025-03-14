# Gaussian Splat Registration Analysis

このプロジェクトは、CTをリファレンスとしてガウシャンスプラット（GS）の精度を検証するためのツールです。

## 概要

放射線治療において、位置合わせのための3Dリファレンスモデルの作成にはCTが広く用いられています。
本プロジェクトでは、被ばくを伴わない代替手法としてGaussian Splattingを用いた3Dモデル生成の精度検証を行います。

## 機能

- スケール補正
- 平行移動補正
- 回転補正（最小二乗法）
- ICPレジストレーション
- 詳細な誤差分析
- 結果の可視化

## 必要なパッケージ

```
open3d
numpy
matplotlib
```

## 使用方法

1. 必要なパッケージをインストール
```bash
pip install open3d numpy matplotlib
```

2. スクリプトを実行
```bash
python pointResist.py
```

## 出力

- 各変換ステップでの誤差分析
- 誤差分布のヒストグラム（PNG形式）
- 詳細な結果レポート（テキストファイル）
- 3D可視化結果

## ライセンス

MIT License 
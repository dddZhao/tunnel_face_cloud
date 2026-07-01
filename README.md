# tunnel_face_cloud

连续隧道掌子面点云相对坐标构建与几何分析程序。第一阶段面向两组相邻开挖循环，优先处理 LAZ，保留 E57 作为归档数据，不覆盖原始文件。

## 坐标与矩阵约定

- 统一隧道局部坐标：`X` 沿掘进方向向前，`Y` 面向掘进方向向右，`Z` 竖直向上。
- 点采用列向量，齐次变换左乘：

```text
p_tunnel_h = T_raw_to_tunnel @ p_raw_h
T = [R, t; 0, 1]
t = -R @ face_center_raw + [s_i, 0, 0]
```

- 距离单位为米，角度单位为度。
- 已知进尺是硬约束。第一阶段不执行无约束 ICP。
- 算法只使用 XYZ 几何坐标，不使用颜色、强度或回波字段。

## 第一阶段数据

默认配置位于 `configs/dataset_ky2xd_phase1.yaml`：

- `C01`: `DYK980+904.0.laz`, 累计进尺 `0.0 m`
- `C02`: `DYK980+909.2.laz`, 相对上一循环进尺 `5.2 m`

## 环境

建议创建独立 conda 环境：

```bash
conda create -n tf_pointcloud python=3.10 -y
conda activate tf_pointcloud
python -m pip install -e ".[dev]"
```

## 运行

```bash
tunnel-cloud inspect --config configs/dataset_ky2xd_phase1.yaml
tunnel-cloud estimate-frames --config configs/dataset_ky2xd_phase1.yaml
tunnel-cloud place-cycles --config configs/dataset_ky2xd_phase1.yaml
tunnel-cloud features --config configs/dataset_ky2xd_phase1.yaml
tunnel-cloud project --config configs/dataset_ky2xd_phase1.yaml
tunnel-cloud run-all --config configs/dataset_ky2xd_phase1.yaml
```

脚本形式也可运行：

```bash
python scripts/run_pipeline.py --config configs/dataset_ky2xd_phase1.yaml
```

## 第一阶段输出

```text
outputs/
├── 01_quality/
├── 02_frames/
├── 03_placement/
├── 04_features/
└── 05_projection/
```

`outputs/03_placement/transforms` 会保存每个循环的：

- `Cxx_raw_to_tunnel.npy`
- `Cxx_raw_to_tunnel.txt`
- `Cxx_tunnel_to_raw.npy`

## 当前边界

第一阶段不做裂隙识别、产状反演、深度学习、自动多循环处理、无约束 ICP。稳定区域约束精化保留命令接口，待双循环基础结果确认后扩展。

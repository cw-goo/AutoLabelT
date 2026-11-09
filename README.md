# AutoLabel

视频与图像智能标注工作台。将视频采样、模型推理、质量筛选、YOLO 数据集输出和人工复核整合在同一桌面界面中。

![AutoLabel 工作台](docs/images/workbench.png)

## 主要功能

- 按目标数量动态抽帧，按目标高度、小目标占比筛选视频样本。
- 生成 JPG、YOLO TXT 和原生复核 JSON，保存源视频、源帧索引及运行配置。
- 新样本实时加入画布；人工修改保存后，通过“同步更新标签”写回 TXT。
- 支持本地 YOLO 检测权重，以及注册的 COCO YOLO、YOLO-World、Grounding DINO、Florence-2、SAM 3/3.1 适配器。
- 提供图片标注、类别管理、矩形编辑、图片模型选择及低置信度目标辅助复核。
- 标签导入、同步和保存包含类别校验、文件访问预检及失败恢复。

## 安装与启动

推荐 Python 3.11，项目支持 Python >=3.11、<3.13。GPU 用户应根据显卡驱动安装匹配的 PyTorch/CUDA；CPU 用户也需确认所选模型后端能够运行。

```bash
python -m pip install -e .
python -m autolabel
```

安装后也可使用 `autolabel` 命令。`autolabel version` 查看版本，`autolabel config` 查看配置路径，`autolabel convert` 查看标签转换任务。默认用户配置为 `~/.autolabelrc`，检测到旧配置时只复制迁移，不删除原文件。

Windows 安装包和 Linux 安装包的构建、安装说明分别位于 [packaging/windows](packaging/windows/INSTALL.txt) 和 [packaging/linux](packaging/linux/INSTALL.txt)。当前分发形式为 Python wheel 和安装脚本，首次安装需要网络，并非免安装单文件 EXE。Linux 桌面、视频解码及 CUDA 仍需在目标设备验收。

部分模型需要可选依赖，例如：

```bash
python -m pip install -e ".[multimodal]"
python -m pip install -e ".[yolo-world]"
```

图片模型目录中的其他后端可能另需 ONNX Runtime 等运行组件，应按所选模型要求安装。

## 一次完整的视频标注流程

1. 在“视频数据集”面板选择视频和新的可写输出目录。
2. 打开“模型加载”，选择视频模型并填写检测类别。选择本身不触发下载；受管模型需明确下载并测试后使用。
3. 在“调整参数”中确认检测、画面筛选和动态抽帧设置，可保存或加载 JSON 配置。
4. 点击“开始构建”，查看源帧位置、保存数和淘汰原因。已保存图片会逐步进入复核列表。
5. 在画布修改并保存标注，再点击“同步更新标签”，将人工修改写回训练用 TXT。

视频本地 YOLO 基线固定检测类别 ID 0，并命名为 `person`。图片本地 YOLO 导入可按权重 ID 顺序输入多个类别。COCO 模型只能选择已有类别；开放词汇模型的文本类别顺序决定输出 ID。

## 数据格式与默认参数

```text
dataset/
  images/       JPG 与同名复核 JSON
  labels/       YOLO TXT
  classes.txt   类别映射
  configs/      当前配置及每次运行快照
  metadata/     采样来源、筛选记录及逐框分数
```

YOLO TXT 每行保持 `class_id center_x center_y width height`，不混入分数或业务字段。已有 JSON 保留人工修改；类别冲突或同前缀样本会拒绝构建，避免覆盖数据。

| 参数 | 默认值 |
| --- | ---: |
| 检测置信度 / 推理尺寸 | 0.35 / 1280 |
| 主目标 / 伴随目标最小高度 | 100 / 40 像素 |
| 小目标比例阈值 / 唯一主目标高度下限 | 0.60 / 110 像素 |
| 淘汰 / 单主目标 / 2至3主目标 / 4及以上主目标步长 | 500 / 300 / 160 / 100 帧 |

步长单位为源视频帧，实际时间间隔需除以视频 FPS。具体边界、数据保护和任务流程见 [架构说明](docs/architecture.zh-CN.md)。

## 模型与运行限制

视频模型默认保存在本次启动的独立临时目录，构建结束或正常退出后清理；不会删除用户导入的本地权重或图片模型缓存。下载期间仍占用网络和磁盘，强制结束进程可能留下临时文件。

Florence-2 需要显式信任固定版本模型代码，其占位分数不能用于衡量置信度。SAM 3/3.1 需相应权重授权，分割结果转换为矩形框。模型加载测试仅验证接口可运行，不代表检测质量。所有自动标注结果均需按业务要求人工复核。

当前视频工作流不提供自动转码、断点续跑或跨帧近重复分析。取消采用协作方式，底层加载和单次推理返回后才能完成停止。应用更新检查在未配置 AutoLabel 发布源时保持关闭，模型下载地址单独管理。

## 开发与测试

```bash
python -m pip install -e ".[test]"
python -m pytest -q tests
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
```

Qt 测试使用 `QT_QPA_PLATFORM=offscreen`。若 Windows 默认临时目录不可写，可将 TEMP/TMP 设置到仓库可写目录，并指定 `--basetemp=artifacts/test_runs/current`；先创建父目录。

修改图片、翻译或 Qt 资源后，使用 Qt 的 `rcc` 和 `lrelease` 重建：

```bash
python tools/build_resources.py --rcc /path/to/rcc --lrelease /path/to/lrelease
```

仓库自带运行必需的 `resources.rcc`，最终用户无需安装 Qt 构建工具。GitHub Actions 提供 Windows/Linux 测试与 wheel 构建配置；工作流通过不替代目标设备上的 GPU 和真实视频验收。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `autolabel/` | 集成入口、视频工作流、模型管理和标签桥接 |
| `autolabel/workbench/` | 标注内核、画布、图片模型和 Qt 资源 |
| `tests/` | 算法、转换、事务、桌面和发布结构回归 |
| `packaging/` | Windows/Linux 分发脚本 |
| `tools/` | 资源构建和发布检查工具 |
| `docs/` | 架构说明、界面截图及发布说明 |
| `databuild.py` | 已验证的视频算法基线 |

模型权重、数据集、测试输出、构建目录和本地软著材料不纳入 Git。发布前可运行 `python tools/check_release.py --check-index` 检查暂存内容；详细整理说明见 [发布清理记录](docs/release-preparation.zh-CN.md)。

## 来源与许可

标注内核基于 **X-AnyLabeling 4.2.0**，现已迁移至 `autolabel/workbench/`；原有作者及许可声明保留。许可文本见 [LICENSE](LICENSE)，来源与组件边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。品牌和命名空间调整不改变第三方代码或权重的权利归属，分发时须遵守相应许可。

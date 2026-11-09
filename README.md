<div align="center">
  <img src="autolabel/assets/logo.png" alt="AutoLabel 简约立体 A Logo" width="128" height="128" />
  <h1>AutoLabel</h1>
  <p><strong>从视频素材，到可复核的训练数据集。</strong></p>
  <p>视频动态采样 · AI 辅助标注 · 人工复核 · YOLO 数据集构建</p>
  <p>
    <a href="pyproject.toml"><img src="https://img.shields.io/badge/version-0.1.0-087f8c?style=flat-square" alt="Version 0.1.0" /></a>
    <a href="#快速开始"><img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776ab?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.11 / 3.12" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0-16263c?style=flat-square" alt="License GPL-3.0" /></a>
  </p>
  <p>
    <a href="#核心能力">核心能力</a> ·
    <a href="#快速开始">快速开始</a> ·
    <a href="#使用流程">使用流程</a> ·
    <a href="#模型支持">模型支持</a> ·
    <a href="docs/architecture.zh-CN.md">架构文档</a> ·
    <a href="https://github.com/cw-goo/Autolabel/issues">问题反馈</a>
  </p>
</div>

---

## 简介

**AutoLabel** 是面向视频与图像的智能标注桌面工作台，将视频抽帧、模型推理、样本筛选和人工复核串成一条完整流程，帮助你把原始视频整理为可用于训练的 YOLO 数据集。

视频工作流依据目标数量调整采样步长，按目标高度与小目标占比筛选样本；保存的图片逐步进入标注画布，供人工检查和修正。每个样本保留源视频、帧索引及运行配置，方便追溯。

**工作流：** 选择视频 → 模型检测与动态采样 → 样本筛选 → 画布复核 → 同步 YOLO 标签

## 核心能力

| 能力 | 你可以做什么 |
| --- | --- |
| **视频数据集构建** | 按目标数量动态抽帧，用目标高度、小目标占比等条件筛选画面，查看保存数与淘汰原因。 |
| **AI 辅助标注** | 导入本地 YOLO 权重，或选择已适配的预训练、开放词汇模型生成初始标注。 |
| **画布人工复核** | 编辑矩形框、管理类别，使用低置信度目标辅助复核；保存后同步更新训练标签。 |
| **图像标注** | 使用集成标注工作台的图片模型面板与标注工具，处理已有图片数据。 |
| **标准输出与追溯** | 同时输出 JPG、YOLO TXT 和原生复核 JSON，记录采样来源、运行配置及逐框分数。 |
| **标注数据保护** | 导入与同步前校验类别、预检文件访问；批量写入支持普通失败与取消时回滚。 |

## 快速开始

### 1. 准备环境

推荐使用 **Python 3.11**，支持 Python `>=3.11,<3.13`。建议在独立虚拟环境中安装。GPU 推理需根据显卡驱动配置匹配的 PyTorch/CUDA；CPU 推理也需满足所选模型后端的要求。

```bash
git clone https://github.com/cw-goo/Autolabel.git
cd Autolabel
python -m venv .venv
```

激活虚拟环境：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS shell
source .venv/bin/activate
```

### 2. 安装并启动

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m autolabel
```

安装后也可使用 `autolabel` 命令。`autolabel version` 查看版本，`autolabel config` 查看配置路径，`autolabel convert` 查看标签转换任务。默认用户配置为 `~/.autolabelrc`，检测到旧配置时只复制迁移，不删除原文件。

> **安装包说明：** 当前提供 Python wheel 与安装脚本，首次安装需要网络。构建与安装步骤见 [Windows 安装说明](packaging/windows/INSTALL.txt) 和 [Linux 安装说明](packaging/linux/INSTALL.txt)。Linux 桌面、视频解码及 CUDA 仍需在目标设备验收。

## 使用流程

1. **选择素材。** 在“视频数据集”面板选择视频与新的可写输出目录。
2. **准备模型。** 打开“模型加载”，选择视频模型并设置检测类别。选择模型本身不触发下载；受管模型需明确下载并完成加载测试后使用。
3. **配置采样。** 在“调整参数”中设置检测、画面筛选和动态抽帧参数，可保存或加载 JSON 配置。
4. **开始构建。** 查看源帧位置、保存数量及淘汰原因；已保存样本会逐步进入复核列表。
5. **复核并同步。** 在画布修改、保存标注，再点击“同步更新标签”，把人工修改写回训练用 YOLO TXT。

> **类别设置：** 视频本地 YOLO 基线固定检测类别 ID `0`，并命名为 `person`。图片本地 YOLO 可按权重 ID 顺序输入多个类别。COCO 模型只能选择已有类别；开放词汇模型的文本类别顺序决定输出 ID。

## 模型支持

下表为**视频工作流已注册的适配器**。图片模型由工作台的独立模型目录管理，不能直接等同于视频支持范围。

| 模型 | 用途 | 使用要点 |
| --- | --- | --- |
| 本地 YOLO | 使用自己的 `.pt` 检测权重 | 视频基线检测 `person`；图片导入支持按权重 ID 配置类别。 |
| COCO YOLO | 使用预训练目标检测模型 | 仅选择权重已有的 COCO 类别。 |
| YOLO-World | 文本提示的开放词汇检测 | 需要 `yolo-world` 可选依赖。 |
| Grounding DINO | 按文本类别定位目标 | 需要 `multimodal` 可选依赖。 |
| Florence-2 | 视觉语言模型辅助检测 | 需要 `multimodal` 依赖及显式信任固定版本模型代码；占位分数不用于衡量置信度。 |
| SAM 3 / 3.1 | 文本概念分割转检测框 | 需要相应权重访问授权；当前视频流程使用转换后的矩形框。 |

按所选模型安装可选依赖：

```bash
python -m pip install -e ".[multimodal]"
python -m pip install -e ".[yolo-world]"
```

图片模型目录中的其他后端可能另需 ONNX Runtime 等运行组件，请按对应模型要求配置。模型加载测试只验证接口可运行，自动生成的标注仍需人工复核。

## 数据集输出

```text
dataset/
├── images/       # JPG 图片与同名复核 JSON
├── labels/       # YOLO TXT 标签
├── classes.txt   # 类别映射
├── configs/      # 当前配置与每次运行快照
└── metadata/     # 采样来源、筛选记录与逐框分数
```

YOLO TXT 每行保持 `class_id center_x center_y width height`，坐标归一化。分数与来源信息单独存储，不混入训练标签。已有复核 JSON 保留人工修改；类别冲突或同前缀样本会拒绝构建，避免覆盖数据。

<details>
<summary><strong>查看视频采样默认参数</strong></summary>

| 参数 | 默认值 |
| --- | ---: |
| 检测置信度 / 推理尺寸 | 0.35 / 1280 |
| 主目标 / 伴随目标最小高度 | 100 / 40 像素 |
| 小目标比例阈值 / 唯一主目标高度下限 | 0.60 / 110 像素 |
| 淘汰画面步长 | 500 帧 |
| 单主目标 / 2–3 主目标 / 4 及以上主目标步长 | 300 / 160 / 100 帧 |

步长单位为源视频帧，实际时间间隔需除以视频 FPS。具体边界、数据保护和任务流程见 [架构说明](docs/architecture.zh-CN.md)。

</details>

## 常见问题与运行边界

<details>
<summary><strong>视频模型下载后保存在什么位置？</strong></summary>

视频模型默认保存在本次启动的独立临时目录，构建结束或正常退出后清理；不会删除用户导入的本地权重或图片模型缓存。下载期间仍占用网络和磁盘，强制结束进程可能留下临时文件。

</details>

<details>
<summary><strong>能否断点续跑、自动转码或去除近重复帧？</strong></summary>

当前视频流程尚不提供断点续跑、自动转码或跨帧近重复分析。动态抽帧基于目标数量和画面筛选规则，并不等同于近重复检测。

</details>

<details>
<summary><strong>点击取消后为什么还需要等待？</strong></summary>

取消采用协作方式，底层模型加载或单次推理返回后才能完成停止。请等待后台任务结束。

</details>

<details>
<summary><strong>应用更新与模型下载如何管理？</strong></summary>

未配置 AutoLabel 发布源时，应用更新检查保持关闭。模型下载地址独立管理；各模型代码、运行组件与权重适用各自的许可和访问条件。

</details>

## 开发与贡献

欢迎通过 [Issues](https://github.com/cw-goo/Autolabel/issues) 反馈问题或提出改进建议。报告问题时请附上系统、Python 版本、模型名称、复现步骤与相关日志。

```bash
python -m pip install -e ".[test]"
python -m pytest -q tests
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
```

Qt 测试使用 `QT_QPA_PLATFORM=offscreen`。若 Windows 默认临时目录不可写，可将 TEMP/TMP 设置到仓库可写目录，并指定 `--basetemp=artifacts/test_runs/current`；先创建父目录。

<details>
<summary><strong>Qt 资源重建与发布检查</strong></summary>

修改图片、翻译或 Qt 资源后，使用 Qt 的 `rcc` 和 `lrelease` 重建：

```bash
python tools/build_resources.py --rcc /path/to/rcc --lrelease /path/to/lrelease
```

仓库自带运行必需的 `resources.rcc`，最终用户无需安装 Qt 构建工具。GitHub Actions 提供 Windows/Linux 测试与 wheel 构建配置；工作流通过不替代目标设备上的 GPU 和真实视频验收。

模型权重、数据集、测试输出、构建目录和本地软著材料不纳入 Git。发布前可运行 `python tools/check_release.py --check-index` 检查暂存内容；详细整理说明见 [发布清理记录](docs/release-preparation.zh-CN.md)。

</details>

### 项目结构

| 路径 | 用途 |
| --- | --- |
| `autolabel/` | 集成入口、视频工作流、模型管理和标签桥接 |
| `autolabel/workbench/` | 标注内核、画布、图片模型和 Qt 资源 |
| `tests/` | 算法、转换、事务、桌面和发布结构回归 |
| `packaging/` | Windows/Linux 分发脚本 |
| `tools/` | 资源构建和发布检查工具 |
| `docs/` | 架构说明、界面截图及发布说明 |
| `databuild.py` | 已验证的视频算法基线 |

提交前请阅读 [贡献指南](AGENTS.md)，为行为变更补充回归测试；PR 中说明用户可见的变化、验证方法，并为界面改动附上截图。

## 致谢与许可

AutoLabel 的标注内核基于 [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) **4.2.0**，现位于 `autolabel/workbench/`。感谢上游项目及其贡献者，原有作者与许可声明均予以保留。

本项目采用 **GPL-3.0** 许可，详见 [LICENSE](LICENSE)。第三方组件来源与边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)；品牌和命名空间调整不改变第三方代码与权重的权利归属，分发时须遵守相应许可。

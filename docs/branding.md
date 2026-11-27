# AutoLabelT 品牌标识

AutoLabelT 中的 T 代表 Training。主工具栏用原图标的暖橙色标出“标注 / 训练”，继续使用原图标和现有布局。安装注册的终端命令为 `autolabelt`，Python 包和 Conda 环境均使用 `autolabelt`，Qt 设置沿用旧版标识以读取已有偏好。

Logo 使用用户桌面原图去除背景后的高保真 SVG 转换版：橙色冠羽、蓝色鸟身、铜色枝干和两片绿叶构成艺术字母 **A**。保留原图造型、配色及 1254 × 1254 画布和留白，不重新设计。背景透明，白色鸟头保留。

## 资源与引用

- [SVG 矢量主图](../autolabelt/assets/logo.svg)：从桌面移入的原图转换版，保留其路径和颜色渐变，无嵌入位图。
- [PNG 导出版](../autolabelt/assets/logo.png)：从 SVG 渲染的 1024 × 1024 透明图片，供软件工具栏、窗口、设置页及 README 预览使用；README 中点击 Logo 可查看 SVG。
- `autolabelt/workbench/resources/resources.qrc`：通过 `images/icon.svg`、`images/logo.svg` 与原有 PNG 兼容别名引用资源。
- `autolabelt/workbench/resources/images/icon.ico`、`icon.icns`：从主图导出的平台图标。

更新 SVG 主图后，执行 `python tools/export_brand_icons.py` 一并导出 PNG 和平台图标，再执行 `python tools/build_resources.py --rcc /path/to/rcc --lrelease /path/to/lrelease`，并将生成的 `resources.rcc` 一并提交。SVG 与 PNG 均须包含在 `pyproject.toml` 和 `MANIFEST.in` 的资源规则中。可用 `--size 2048` 导出更大的 PNG。

执行 `python tools/verify_brand_ui.py` 检查浅色、深色工作台和不同尺寸的图标，图片输出到 `artifacts/branding/`，使用独立临时配置。执行 `python tools/review_readme.py` 检查 README 本地链接并生成网页预览。

## 编辑方式

本版直接转换用户原图，通过密集的路径与渐变保留原有颜色变化；没有按造型重新绘制，也没有裁切原图留白。复杂渐变使 SVG 文件较大，README 使用导出的 PNG 预览以减少加载量。

导出工具先按 SVG 原生坐标网格渲染，再缩放成 PNG，避免 Qt 在小尺寸下直接渲染密集渐变路径时产生透明接缝和颜色偏淡。

可使用 Inkscape、Illustrator 等矢量编辑器打开主图。更新兼容图片时只运行导出与资源构建工具，不修改主图的轮廓、配色和画布。

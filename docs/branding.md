# AutoLabel 品牌标识

Logo 根据用户提供的蓝橙啄木鸟参考图重绘：橙色冠羽、蓝色鸟身、铜色枝干和两片绿叶构成艺术字母 **A**。调整了边缘衔接、渐变高光与画布留白，保留原图的主要造型。背景透明，可直接用于浅色和深色界面。

## 资源与引用

- [SVG 矢量主图](../autolabel/assets/logo.svg)：由贝塞尔路径、圆形和渐变组成，供 README、工具栏、窗口和设置页使用；无嵌入位图，可直接编辑、缩放。
- [PNG 导出版](../autolabel/assets/logo.png)：从 SVG 渲染的 1024 × 1024 透明图片，供兼容位图的入口使用。
- `autolabel/workbench/resources/resources.qrc`：通过 `images/icon.svg`、`images/logo.svg` 与原有 PNG 兼容别名引用资源。
- `autolabel/workbench/resources/images/icon.ico`、`icon.icns`：从主图导出的平台图标。

更新 SVG 主图后，执行 `python tools/export_brand_icons.py` 一并导出 PNG 和平台图标，再执行 `python tools/build_resources.py --rcc /path/to/rcc --lrelease /path/to/lrelease`，并将生成的 `resources.rcc` 一并提交。SVG 与 PNG 均须包含在 `pyproject.toml` 和 `MANIFEST.in` 的资源规则中。可用 `--size 2048` 导出更大的 PNG。

执行 `python tools/verify_brand_ui.py` 检查浅色、深色工作台和不同尺寸的图标，图片输出到 `artifacts/branding/`，使用独立临时配置。执行 `python tools/review_readme.py` 检查 README 本地链接并生成网页预览。

## 编辑方式

本版直接依据用户参考图进行矢量重绘。SVG 中的 `branch-a`、`woodpecker` 和 `crossbar-and-leaves` 分组分别控制枝干、鸟形、横画与绿叶；`defs` 中的命名渐变用于调整色彩及立体层次。

可使用 Inkscape、Illustrator 等矢量编辑器打开主图。编辑时保留方形 `viewBox`、透明背景及足够留白，避免细化到小图标尺寸下无法辨认的装饰。

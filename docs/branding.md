# AutoLabel 品牌标识

Logo 是独立的加粗艺术字母 **A**，采用啄木鸟仿生造型：冠羽与直喙融入顶部，翼羽、颈部和尾羽构成字母笔画，羽毛横画连接两侧。红金冠羽搭配青绿、宝蓝渐变，以柔和的层叠羽片与立体高光呈现华丽质感。无底板、边框或背景，字母外部及内部留白均保持透明，可直接用于浅色和深色界面。

## 资源与引用

- [主图](../autolabel/assets/logo.png)：图像生成工具输出的 PNG，供 README、工具栏和应用窗口共用。
- `autolabel/workbench/resources/resources.qrc`：通过 `images/icon.png` 与 `images/logo.png` 两个兼容别名引用主图，供独立工作台、设置页与模型菜单使用。
- `autolabel/workbench/resources/images/icon.ico`、`icon.icns`：从主图导出的平台图标。

更新主图后，执行 `python tools/export_brand_icons.py` 导出平台图标，再执行 `python tools/build_resources.py --rcc /path/to/rcc --lrelease /path/to/lrelease`，并将生成的 `resources.rcc` 一并提交。PNG 需同时包含在 `pyproject.toml` 和 `MANIFEST.in` 的资源规则中。

执行 `python tools/verify_brand_ui.py` 检查浅色、深色工作台和不同尺寸的图标，图片输出到 `artifacts/branding/`，使用独立临时配置。执行 `python tools/review_readme.py` 检查 README 本地链接并生成网页预览。

## 设计来源与提示词

通过内置 `imagegen` 工具编辑原 AutoLabel Logo，生成 PNG；平台图标仅作尺寸与格式导出，不另行重绘。以下为生成时使用的完整提示词：

```text
Use case: logo-brand.
Asset type: production AutoLabel desktop app icon and README logo.
Input image: current AutoLabel logo is the edit target, used only as the reference for the elegant capital A gesture.
Primary request: remove the entire navy tile and ALL background. Create ONLY ONE thick artistic capital "A", with a biomimetic woodpecker form integrated into its letter anatomy and luxurious colorful plumage. The user wants a substantially bolder A, richer coloration, and a crested woodpecker-inspired silhouette.
Letter anatomy: preserve an immediately readable uppercase A with two stout sweeping legs, a strong connecting crossbar, and a clearly open triangular counter. Make the strokes about 40 percent thicker and more substantial than the reference, with smooth rounded dimensional contours. The top apex subtly becomes a woodpecker's sleek crested head with a small dark eye and a distinctive straight pointed beak facing right. One A diagonal is formed by the sweeping folded wing, the other by the flowing neck/body into elegant tail-feather tips. The crossbar is a broad clean sweeping feather. The bird and letter must be ONE unified monogram, not a bird sitting on a letter, not a separate bird illustration. Prioritize the A's silhouette.
Plumage: luminous emerald and peacock turquoise transitioning through deep sapphire and royal blue, with restrained amber-gold highlights and a vivid coral/scarlet woodpecker crest. Rich harmonious color gradients, distinctive and jewel-like rather than one-color ivory. Use a few broad layered feather shapes to suggest luxurious plumage, without dense tiny realistic feather texture.
Style/material: polished contemporary biomorphic brand sculpture, shallow 2.5D feather layering, soft satin sheen, restrained sculpted highlights, smoothly rounded beveled edges, clean antialiasing, elegant and fluid. A commanding logo silhouette still legible at 28px. No angular jagged feather fringe, no hairy fine lines, no glossy plastic, no heavy outline.
Composition: single centered standalone A monogram occupying around 84 percent of a square canvas, all crest/beak/tail tips fully inside with a small uniform safe margin. Genuinely transparent background with alpha everywhere outside the letter AND inside its triangular counter. No navy tile, no rounded square, no plaque, no circle, no frame, no background glow, no ground plane, no cast shadow, no branch, no scenery, no wordmark, no other text, no watermark, no mockup. Only the colorful thick woodpecker-inspired capital A.
```

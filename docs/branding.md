# AutoLabel 品牌标识

Logo 仅保留优雅的艺术字母 **A** 与深海军蓝圆角底板，去掉取景框和播放标记。珍珠白笔画采用浅浮雕、柔和倒角与少量青色侧光，底板边缘圆润平滑。生成图的圆角外部保留透明度，可用于浅色和深色界面。

## 资源与引用

- [主图](../autolabel/assets/logo.png)：图像生成工具输出的 PNG，供 README、工具栏和应用窗口共用。
- `autolabel/workbench/resources/resources.qrc`：通过 `images/icon.png` 与 `images/logo.png` 两个兼容别名引用主图，供独立工作台、设置页与模型菜单使用。
- `autolabel/workbench/resources/images/icon.ico`、`icon.icns`：从主图导出的平台图标。

更新主图或平台图标后，执行 `python tools/build_resources.py --rcc /path/to/rcc --lrelease /path/to/lrelease`，并将生成的 `resources.rcc` 一并提交。PNG 需同时包含在 `pyproject.toml` 和 `MANIFEST.in` 的资源规则中。

## 设计来源与提示词

通过内置 `imagegen` 工具编辑原 AutoLabel Logo，生成 PNG；平台图标仅作尺寸与格式导出，不另行重绘。以下为生成时使用的完整提示词：

```text
Use case: logo-brand.
Asset type: final AutoLabel desktop application icon and GitHub README logo.
Input image: the supplied current AutoLabel logo is the edit target and identity reference.
Primary request: make this identity noticeably MORE MINIMAL, gently THREE-DIMENSIONAL, and SOFTER at every edge while keeping an elegant artistic capital "A".
Simplify the composition to just one beautifully drawn A on the deep navy rounded-square tile. Remove all four turquoise corner brackets and remove the small amber play triangle. Simplify the A's long looping flourish into a shorter, clean flowing crossbar; keep a graceful slight italic, generous interior negative space and clearly recognizable A silhouette. Do not add any symbols or lettering.
Material and depth: the A is a shallow sculpted satin ivory/pearl ribbon with softly rounded bevels, subtle thickness, delicate cool turquoise reflected light along one side, soft ambient contact shading and a restrained upper-left highlight. Elegant, quiet premium industrial design, softly rounded edges and smooth precise curves. No harsh sharp points, no chrome, no mirror reflections, no glitter, no exaggerated glass effects, no heavy extrusion.
Background tile: deep midnight navy, softly pillowed squircle with a very gentle edge highlight and quiet smooth tonal shading. Rounded and tactile, not a hard flat slab. The letter remains the clear hero, with balanced breathing room, large enough to remain readable at 28 pixels.
Composition: a single square icon viewed straight on, centered, tile nearly fills square canvas with only a tiny safe margin. Actual transparent background outside the rounded tile, preserving alpha; no surrounding colored canvas, no detached drop shadow.
Constraints: a usable production asset, not a presentation mockup or scene, no grid of variants, no text other than the single capital A, no watermark. Smooth antialiased contours, strong small-size readability, restrained two-color palette of navy and pearl with subtle cyan reflected light.
```

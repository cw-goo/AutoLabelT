# AutoLabel 品牌标识

Logo 以带书写感的艺术字母 **A** 为主体，沿用深海军蓝底色、青色取景框和琥珀色播放标记。生成图的圆角外部保留透明度，可用于浅色和深色界面。

## 资源与引用

- [主图](../autolabel/assets/logo.png)：图像生成工具输出的 PNG，供 README、工具栏和应用窗口共用。
- `autolabel/workbench/resources/resources.qrc`：通过 `images/icon.png` 与 `images/logo.png` 两个兼容别名引用主图，供独立工作台、设置页与模型菜单使用。
- `autolabel/workbench/resources/images/icon.ico`、`icon.icns`：从主图导出的平台图标。

更新主图或平台图标后，执行 `python tools/build_resources.py --rcc /path/to/rcc --lrelease /path/to/lrelease`，并将生成的 `resources.rcc` 一并提交。PNG 需同时包含在 `pyproject.toml` 和 `MANIFEST.in` 的资源规则中。

## 设计来源与提示词

通过内置 `imagegen` 工具编辑原 AutoLabel Logo，生成 PNG；平台图标仅作尺寸与格式导出，不另行重绘。以下为生成时使用的完整提示词：

```text
Use case: logo-brand.
Asset type: production application icon for AutoLabel, a video/image annotation desktop workbench; also GitHub README logo.
Input image: the supplied icon is the edit target and identity reference.
Primary request: redesign the central capital "A" into an elegant, artistic custom calligraphic monogram. An exquisite contemporary italic A, sculptural thick-to-thin strokes, graceful tapered terminals, confident sweeping crossbar, optical balance and beautiful negative space. Clearly and immediately readable as a capital A. Refined rather than ornate; strong silhouette at 28 pixels.
Preserve the established deep navy rounded-square tile, luminous turquoise/cyan annotation-frame corner motif and ivory monogram palette. Refine the four framing corners to be thinner and more restrained so the A is the hero. Use a very small warm amber accent integrated tastefully into the sweeping crossbar, recalling the original video-play accent without obscuring the letter.
Composition: a single centered icon, square canvas, icon tile almost fills the canvas with only a tiny safe margin, balanced breathing room inside. Transparent pixels outside the rounded tile; tile itself remains solid deep navy.
Style: exceptionally clean flat vector-like brand artwork, precise smooth contours, subtle restrained color transitions only. No photograph, no mockup, no typography other than the A, no extra wordmark, no multiple options, no border around the canvas, no shadows, no bevel, no watermark. Keep important monogram strokes sufficiently substantial to survive small UI sizes.
```

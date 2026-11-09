# 本地输出与干净构建

`autolabel/` 保存正式程序源码、模型适配器、标注工作台与运行资源，纳入 Git。`artifacts/` 保存本地输出，已由 `.gitignore` 和源码分发清单排除，软件正常启动不依赖其内容。

## 输出目录

| 目录 | 用途与处理方式 |
| --- | --- |
| `artifacts/release/` | 当前重新构建并通过校验的 wheel、源码包与平台安装包；对外分发前确认版本和测试结果。 |
| `artifacts/branding/` | 最新 Logo 尺寸预览与浅色、深色界面截图，可随时重建。 |
| `artifacts/test_runs/current/` | 最近一次完整测试使用的数据，后续运行会更新。 |
| `artifacts/pytest_cache/` | pytest 缓存，可重新生成。 |
| `artifacts/build/` | 干净构建的临时工作区，正常结束自动清理。 |
| `artifacts/maintenance/` | 历史诊断脚本和日志。包含针对本机数据集的操作，不能作为通用安装步骤执行。 |
| `artifacts/copyright_docs/` | 本地软著文档与渲染结果，清理前确认原件已归档。 |

用户数据验证导出、软著资料和仍有价值的历史维护记录应单独保留。遇到 Windows 文件权限限制时，记录未清理路径，不自动接管所有权或改变权限。

## 构建与检查

```bash
python tools/build_distribution.py --output-dir artifacts/release --sdist
python tools/check_distribution.py artifacts/release/autolabel-0.1.0-py3-none-any.whl --sdist artifacts/release/autolabel-0.1.0.tar.gz
```

`build_distribution.py` 仅复制构建所需的源码、资源、文档和工具到临时目录，不复用根目录的 `build/`、包元数据或本地测试环境。构建完成后先与当前源码核对，再复制至输出目录。

包检查同时拒绝缺失文件、内容过期文件和源码中已删除的多余应用文件，避免旧 Logo 或缓存混入安装包。源码包也排除本地软著材料。Windows/Linux 构建脚本和 CI 使用相同的干净构建入口。

```powershell
# Windows 安装包；Python 参数可指定 Python 3.11 的完整路径
.\packaging\windows\build.ps1 -Python python
```

```bash
python packaging/linux/build.py
```

这些命令生成分发文件，不安装依赖、不验证 Linux 桌面或真实 GPU 推理。正式发布新版本时须同步更新 `pyproject.toml` 等版本声明，避免用相同版本号分发不同内容。

## 可复用工具

```bash
python tools/export_brand_icons.py
python tools/verify_brand_ui.py
python tools/review_readme.py
```

PNG 与平台图标从当前 SVG 矢量主图导出；界面预览使用独立临时配置，不覆盖用户设置；README 工具检查本地链接、页内导航和折叠区，并生成 `artifacts/readme-preview.html`。Qt 资源重建步骤见 [品牌说明](branding.md)。

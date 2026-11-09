# Third-party notices

AutoLabel includes and modifies the X-AnyLabeling 4.2.0 annotation workbench, originally obtained from the user's existing workspace backup. Its source now resides in `autolabel/workbench/`; the Python namespace and application branding have been changed for the AutoLabel distribution. These changes do not change the origin or ownership of the upstream work.

- Upstream project: [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)
- Upstream documentation: [xanylabeling.com](https://xanylabeling.com)
- Bundled workbench license: GNU General Public License, version 3; see [LICENSE](LICENSE).

Original copyright, author and license notices within bundled source files are retained. Other embedded implementations may carry additional notices in their source files. Python dependencies and model weights are provided by their respective authors and retain their own licenses; model selection does not grant additional redistribution rights.

The `autolabel/` application modules outside `workbench/` provide the video dataset workflow, model lifecycle management, annotation bridge and desktop integration. The integrated distribution must retain the applicable GPL license and source obligations; changing module names or branding does not remove them.

Some model presets still use upstream download URLs. Those URLs intentionally retain the upstream repository name so existing models remain downloadable. Historical configuration filenames are recognized only as migration inputs; newly written configuration uses `.autolabelrc`.

The modifications in this working tree include namespace migration, AutoLabel branding, local YOLO integration, annotation validation and rollback support. Consult the source and repository documentation for the actual behavior; no claim is made that the entire bundled workbench or third-party model implementations were independently created for AutoLabel.

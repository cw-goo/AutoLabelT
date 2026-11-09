import os
import re
import time

import cv2
import torch
from ultralytics import YOLO

def is_frame_qualified(valid_boxes, small_boxes):
    """
    质量评估函数：综合目标数量与小目标占比进行过滤
    :return: (bool 是否保留, str 原因)
    """
    n_valid = len(valid_boxes)
    n_small = len(small_boxes)
    n_total = n_valid + n_small

    # 1. 没有任何合格人体 (>= min_pixel_height)
    if n_valid == 0:
        return False, "无合格目标"

    # 2. 小目标占比过高（超过 60%）且合格大目标只有 1 个 -> 认定为远景杂乱帧
    if n_total >= 3 and (n_small / n_total) > 0.60 and n_valid <= 1:
        return False, "小目标占比过高"

    # 3. 画面仅有 1 个人，且高度处于边缘临界 (90~110px) -> 特征偏弱淘汰
    if n_total == 1 and n_valid == 1 and valid_boxes[0][5] < 110:
        return False, "单人边缘弱特征"

    return True, "合格"


def extract_and_label_dataset_unlimited(
    video_path: str,
    output_dir: str,
    conf_threshold: float = 0.35,
    min_pixel_height: int = 90,      # 合格人体高度门槛 (像素)
    # --------- 增大的跳帧步长参数（以25fps计算） ---------
    idle_jump: int = 150,            # 空镜/淘汰帧跳跃步长 (约 6.0 秒)
    step_single: int = 50,           # 单人走动跳跃步长 (约 2.0 秒)
    step_small_group: int = 25,      # 2~3人群体跳跃步长 (约 1.0 秒)
    step_large_group: int = 12,      # >=4人密集群体跳跃步长 (约 0.5 秒)
    # ----------------------------------------------------
    model_name: str = "staffv2.pt",
    imgsz: int = 1280,
    device: str | None = None,
):
    """
    全视频遍历（不设上限），采用大步长自适应跳帧采样并自动标注
    """
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 提取摄像头编号或使用视频文件名作为前缀
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cam_id = re.search(r"ch\d+", video_name, re.IGNORECASE)
    prefix = cam_id.group(0).lower() if cam_id else video_name
    print(f"[*] 摄像头前缀: {prefix} (源视频: {video_name})")

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[*] 加载检测模型: {model_name} (device={device})...")
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[*] 视频分辨率: {frame_width}x{frame_height}, 总帧数: {total_frames}, FPS: {fps:.1f}")
    print(f"[*] 过滤门槛: 有效人体高度 >= {min_pixel_height}px | 自动剔除小目标聚集/弱单人")
    print(f"[*] 步长配置: 空镜 {idle_jump} 帧 | 单人 {step_single} 帧 | 小群体 {step_small_group} 帧 | 大群体 {step_large_group} 帧")
    print(f"[*] 模式: 全片完整遍历，不设保存数量上限\n")

    current_frame_idx = 0
    next_step = step_single  # 初始步长
    saved_count = 0
    scanned_count = 0
    discarded_count = 0
    t_start = time.time()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while current_frame_idx < total_frames:
        # 大步长直接 seek；步长相对较小时顺序 grab 跳过
        if next_step > 35:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
        else:
            for _ in range(max(0, next_step - 1)):
                if not cap.grab():
                    break
            ret, frame = cap.read()

        if not ret or frame is None:
            break

        scanned_count += 1

        # 执行推理
        results = model.predict(source=frame, imgsz=imgsz, conf=conf_threshold,
                                classes=[0], device=device, verbose=False)[0]

        valid_boxes = []  # 存达标大目标: (cls_id, bx, by, bw, bh, box_pixel_h)
        small_boxes = []  # 存不达标微小目标

        if results.boxes is not None and len(results.boxes) > 0:
            xywhn = results.boxes.xywhn.cpu().numpy()
            for box in xywhn:
                bx, by, bw, bh = box
                box_pixel_h = bh * frame_height
                if box_pixel_h >= min_pixel_height:
                    valid_boxes.append((0, bx, by, bw, bh, box_pixel_h))
                else:
                    small_boxes.append((0, bx, by, bw, bh, box_pixel_h))

        # 帧质量裁决
        keep_frame, reason = is_frame_qualified(valid_boxes, small_boxes)

        n_valid = len(valid_boxes)
        if not keep_frame:
            discarded_count += 1
            next_step = idle_jump  # 空镜/淘汰帧采用大步长快速越过
        else:
            # 优质帧落盘
            file_base = f"{prefix}_f{current_frame_idx:06d}"
            img_path = os.path.join(images_dir, f"{file_base}.jpg")
            cv2.imwrite(img_path, frame)

            txt_path = os.path.join(labels_dir, f"{file_base}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                # 合格目标 + 同框 >= 40px 的背景人一同标注，避免产生假负例
                all_targets = valid_boxes + [b for b in small_boxes if b[5] >= 40]
                for cls_id, bx, by, bw, bh, _ in all_targets:
                    f.write(f"{cls_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")

            saved_count += 1

            # 根据人数动态决定下一跳间隔（放大后的步长）
            if n_valid == 1:
                next_step = step_single
            elif n_valid in [2, 3]:
                next_step = step_small_group
            else:
                next_step = step_large_group

        current_frame_idx += next_step

        if scanned_count % 8 == 0 or current_frame_idx >= total_frames:
            print_progress(current_frame_idx, total_frames, saved_count,
                           discarded_count, time.time() - t_start, next_step)

    print()
    cap.release()
    print_stats(saved_count, scanned_count, discarded_count, output_dir, time.time() - t_start)


def print_progress(current, total, saved, discarded, elapsed, step, width=32):
    """单行刷新终端进度条"""
    current_bounded = min(current, total)
    ratio = min(1.0, current_bounded / total) if total else 1.0
    filled = int(width * ratio)
    bar = "█" * filled + "─" * (width - filled)
    speed = current_bounded / elapsed if elapsed > 0 else 0.0
    eta = (total - current_bounded) / speed if speed > 0 else 0
    print(f"\r    [{bar}] {current_bounded}/{total} ({ratio*100:4.1f}%) | "
          f"已保存 {saved} 帧 | 淘汰 {discarded} | 下一步长 {step} | "
          f"{elapsed:.0f}s | ETA {eta:.0f}s   ", end="", flush=True)


def print_stats(saved_count, scanned_count, discarded_count, output_dir, elapsed=0.0):
    print(f"\n[✔] 全视频扫描与标注完成！")
    print(f"    - 最终提取样本数: {saved_count} 帧 (无上限限制)")
    print(f"    - 检验采样节点: {scanned_count} | 淘汰低质帧: {discarded_count}")
    print(f"    - 总耗时: {elapsed:.1f}s ({scanned_count/max(elapsed, 1e-9):.1f} 帧/s)")
    print(f"    - 数据集输出路径: {output_dir}")


if __name__ == "__main__":
    VIDEO_PATH = "videos/sample_ch2.mp4"
    OUTPUT_DIR = "./datasets/sample_ch2"

    extract_and_label_dataset_unlimited(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        conf_threshold=0.35,
        min_pixel_height=100,          # 人体框高度过滤门槛
        idle_jump=500,                # 空镜大步长（跳150帧，约6秒）
        step_single=300,               # 单人大步长（跳50帧，约2秒）
        step_small_group=160,          # 2~3人步长（跳25帧，约1秒）
        step_large_group=100,          # 密集人群步长（跳12帧，约0.5秒）
        model_name="staffv2.pt",
        imgsz=1280
    )

import cv2
import os
import time
from yolo_detector import YoloDetector
from tracker import Tracker
from segment_anything import sam_model_registry, SamPredictor
import torch
import numpy as np

MODEL_PATH = "/hpc2hdd/home/xingmu/MedSAM/yolo11l.pt"
IMAGE_ROOT = "/hpc2hdd/home/xingmu/MedSAM/dataset"  # 替换为你的数据集根目录
OUTPUT_DIR = "output_videos"
output_directory = 'results/plane'
os.makedirs(output_directory, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_checkpoint = "/hpc2hdd/home/xingmu/MedSAM/sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)


def compute_iou(boxA, boxB):
    # box = [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou


def process_image_sequence(image_dir, output_path, detector, tracker, fps=24):
    # 获取图像序列，按文件名排序
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    if not image_files:
        print(f"No images found in {image_dir}")
        return

    # 读取第一帧确定宽高
    first_frame = cv2.imread(os.path.join(image_dir, image_files[0]))
    height, width = first_frame.shape[:2]

    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    target_tracking_ids = [None, None]
    target_tracking_id = None  # 目标物体的tracking_id
   # target_box_estimate = np.array([720, 400, 1500, 800])  # <-- 你大致估计的物体位置（x1, y1, x2, y2）
    target_box_estimates = [
    np.array([200, 300, 320, 500]),   # 第一个目标
    np.array([720, 500, 920, 600])   # 第二个目标（示例）
    ] 
    for idx, image_file in enumerate(image_files):
        frame = cv2.imread(os.path.join(image_dir, image_file))
        if frame is None:
            continue

        start_time = time.perf_counter()
        detections = detector.detect(frame)
        tracking_ids, boxes = tracker.track(detections, frame)
        print(boxes)
        # 在第一帧中寻找目标 tracking_id
        if idx == 0:
            for i, target_est in enumerate(target_box_estimates):
                max_iou = 0
                for tid, box in zip(tracking_ids, boxes):
                    iou = compute_iou(box[0:4], target_est)
                    if iou > max_iou:
                        max_iou = iou
                        target_tracking_ids[i] = tid
                print(f"[INFO] Selected target {i} tracking_id: {target_tracking_ids[i]} (IOU: {max_iou:.3f})")
        for tracking_id, bounding_box in zip(tracking_ids, boxes):
            # print(tracking_id)
            # print(type(tracking_id))
            #if tracking_id == target_tracking_id:
            if tracking_id in target_tracking_ids:
                predictor.set_image(frame)
                box_cood = bounding_box[0:4]
                masks, scores, logits = predictor.predict(point_coords=None, point_labels=None, box=box_cood,
                                                          multimask_output=False, )
                mask = masks[0]
                color = (0, 255, 0) if tracking_id == target_tracking_ids[0] else (0, 0, 255)
            # colored_mask = np.zeros_like(frame, dtype=np.uint8)
                frame[mask] = color
            # cv2.rectangle(frame, (int(bounding_box[0]), int(bounding_box[1])),
            # (int(bounding_box[2]), int(bounding_box[3])), (0, 0, 255), 2)
            # cv2.putText(frame, f"{str(tracking_id)}", (int(bounding_box[0]), int(bounding_box[1]) - 10),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                mask = mask * 255
        output_path = os.path.join(output_directory, f'output_image_{idx}.png')
        cv2.imwrite(output_path, frame)
        end_time = time.perf_counter()
        fps_info = 1 / (end_time - start_time)
        print(f"[{image_dir}] Processing {image_file}, FPS: {fps_info:.2f}")

        out.write(frame)


    out.release()
    print(f"Saved video to {output_path}")

def main():
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.2)
    tracker = Tracker()
    #video_path = '/hpc2hdd/home/xingmu/MedSAM/DAVIS/JPEGImages/1080p/car-roundabout'
    video_path='/hpc2hdd/home/xingmu/MedSAM/computer_vision/yolov10_detection_and_tracking'
    # 遍历每个“视频子目录”
    #for subdir in sorted(os.listdir(IMAGE_ROOT)):
        #subdir_path = os.path.join(IMAGE_ROOT, subdir)
        #if not os.path.isdir(subdir_path):
            #continue
    output_path = os.path.join(OUTPUT_DIR, "plane.mp4")
    process_image_sequence(video_path, output_path, detector, tracker)

if __name__ == "__main__":
    main()


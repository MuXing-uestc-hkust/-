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
output_directory = 'results/person'
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
def process_video(video_file, output_path, detector, tracker):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Failed to open video: {video_file}")
        return

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width)
    print(height)
    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    target_tracking_ids = [None, None]
    target_box_estimates = [
        np.array([471, 412, 527, 489]),
        np.array([728, 518, 782, 596])
    ]

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.perf_counter()
        detections = detector.detect(frame)
        print(detections)
        tracking_ids, boxes = tracker.track(detections, frame)
        print(tracking_ids)
        print(boxes)
        if frame_idx == 0:
            for i, target_est in enumerate(target_box_estimates):
                max_iou = 0
                for tid, box in zip(tracking_ids, boxes):
                    iou = compute_iou(box[0:4], target_est)
                    if iou > max_iou:
                        max_iou = iou
                        target_tracking_ids[i] = tid
                print(f"[INFO] Selected target {i} tracking_id: {target_tracking_ids[i]} (IOU: {max_iou:.3f})")

        for tracking_id, bounding_box in zip(tracking_ids, boxes):
            if tracking_id in target_tracking_ids:
                predictor.set_image(frame)
                box_cood = bounding_box[0:4]
                masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=box_cood,
                                                multimask_output=False)
                mask = masks[0]
                color = (0, 255, 0) if tracking_id == target_tracking_ids[0] else (0, 0, 255)
                frame[mask] = color

        end_time = time.perf_counter()
        fps_info = 1 / (end_time - start_time)
        print(f"[{video_file}] Frame {frame_idx}, FPS: {fps_info:.2f}")
        output_path = os.path.join(output_directory, f'output_image_{frame_idx}.png')
        cv2.imwrite(output_path, frame)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved processed video to {output_path}")


def main():
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.2)
    tracker = Tracker()
    #video_path = '/hpc2hdd/home/xingmu/MedSAM/DAVIS/JPEGImages/1080p/car-roundabout'
    video_path='/hpc2hdd/home/xingmu/MedSAM/computer_vision/yolov10_detection_and_tracking/assets/football.mp4'
    # 遍历每个“视频子目录”
    #video_file = '/hpc2hdd/home/xingmu/MedSAM/your_input_video.mp4'
    #for subdir in sorted(os.listdir(IMAGE_ROOT)):
        #subdir_path = os.path.join(IMAGE_ROOT, subdir)
        #if not os.path.isdir(subdir_path):
            #continue
    output_path = os.path.join(OUTPUT_DIR, "plane.mp4")
    process_video(video_path, output_path, detector, tracker)

if __name__ == "__main__":
    main()


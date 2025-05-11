import cv2
import time
from yolo_detector import YoloDetector
from tracker import Tracker

MODEL_PATH = "/hpc2hdd/home/xingmu/MedSAM/yolo11l.pt"
VIDEO_PATH = "assets/football.mp4"
OUTPUT_PATH = "output.mp4"  # 输出视频路径

def main():
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.5)
    tracker = Tracker()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()

    # 获取原视频的宽、高、帧率信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)

    # 定义视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_input, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.perf_counter()
        detections = detector.detect(frame)
        tracking_ids, boxes = tracker.track(detections, frame)

        for tracking_id, bounding_box in zip(tracking_ids, boxes):
            cv2.rectangle(frame, (int(bounding_box[0]), int(bounding_box[1])), 
                          (int(bounding_box[2]), int(bounding_box[3])), (0, 0, 255), 2)
            cv2.putText(frame, f"{str(tracking_id)}", (int(bounding_box[0]), int(bounding_box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        print(f"Current fps: {fps:.2f}")

        # 写入到输出视频
        out.write(frame)

        # 如果你想实时预览，也可以取消下面的注释
        # cv2.imshow("Tracking", frame)
        # if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        #     break

    cap.release()
    out.release()
    #cv2.destroyAllWindows()
    print(f"Video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

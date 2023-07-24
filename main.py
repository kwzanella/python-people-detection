import cv2
from ultralytics import YOLO

# set supported camera resolution here
# must be multiple of 32 (320x320, 416x416, 448x448, 608x608, etc)
FRAME_WIDTH = 416
FRAME_HEIGHT = 416

def process_frame(model, frame):
    # Run YOLOv8 inference on the frame
    results = model.predict(source=frame, imgsz=FRAME_WIDTH, classes=0, device="cpu", verbose=False)
    annotated_frame = results[0].plot()
    
    return annotated_frame


if __name__ == "__main__":
    model = YOLO(model="models/yolov8n.onnx", task="detect")

    camera = cv2.VideoCapture(0)  # use default camera (index 0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    try:
        while camera.isOpened():
            is_read, frame = camera.read()

            if is_read:
                cv2.imshow("Live People Detection", process_frame(model, frame))
                # press "q" to exit window
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
    except KeyboardInterrupt:
        print("Interrupted by user")
        
    finally:
        camera.release()
        cv2.destroyAllWindows()

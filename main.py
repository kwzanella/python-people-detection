import cv2
from ultralytics import YOLO

# set supported camera resolution here
FRAME_WIDTH = 400
FRAME_HEIGHT = 400

# number of frames to skip between detections
SKIP_FRAMES = 5

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

    frame_count = 0
    try:
        while True:
            is_read, frame = camera.read()

            if is_read:
                frame_count += 1
                if frame_count % SKIP_FRAMES == 0:  # only process and show frame after "SKIP_FRAMES" amount
                    cv2.imshow("Live People Detection", process_frame(model, frame))
                    frame_count = 0

                # press "q" to exit window
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
            
    except KeyboardInterrupt:
        print("Interrupted by user")
        
    finally:
        camera.release()
        cv2.destroyAllWindows()

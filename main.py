import cv2
from multiprocessing import Process, Queue
from queue import Empty
from ultralytics import YOLO

"""
OpenCV's thread safety issues primarily concern the GUI functions, not the video capture functions.
Therefore, it shouldn't be a problem to keep "VideoCapture" methods inside a different process.

Ultralytics's YOLOv8 uses the COCO dataset. That is why "classes=0" is used, because it represents "people" in the dataset
https://cocodataset.org/#home

Producer/Consumer Architecture:
https://www.ni.com/en/support/documentation/supplemental/21/producer-consumer-architecture-in-labview0.html
"""

# Set supported camera resolution here
# Must be multiple of 32 (320x320, 416x416, 448x448, 608x608, etc)
RESOLUTION = (416, 416)

# Define based in available memory and video delay (bigger size = more delay)
QUEUE_SIZE = 20


# Generates frames from device camera
def producer(queue: Queue, camera: cv2.VideoCapture) -> None:
    try:
        while camera.isOpened():
            is_read, frame = camera.read()
            if is_read:
                queue.put(frame)

    finally:
        camera.release()
        
        
# Uses frames as input to YOLO model predict
def consumer(queue_in: Queue, queue_out: Queue, model: YOLO) -> None:
    try:
        while True:
            results = model.predict(source=queue_in.get(), imgsz=RESOLUTION[0], classes=0, device="cpu", verbose=False)
            queue_out.put(results[0].plot())

    finally:
        cv2.destroyAllWindows()


# Creates new ONNX model based on PyTorch (pt) model
# If pt model is not found, it will automatically download it
def export_new_model() -> None:
    model = YOLO(model="models/yolov8n.pt", task="detect")
    model.export(format="onnx", imgsz=RESOLUTION[0], classes=0, dynamic=False, simplify=True)


if __name__ == "__main__":
    #export_new_model()
    model = YOLO(model="models/yolov8n.onnx", task="detect")

    camera = cv2.VideoCapture(0)  # use default camera (index 0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

    queue_in = Queue(maxsize=QUEUE_SIZE)
    queue_out = Queue(maxsize=QUEUE_SIZE)

    producer_process = Process(target=producer, args=(queue_in, camera))
    consumer_process = Process(target=consumer, args=(queue_in, queue_out, model))

    producer_process.start()
    consumer_process.start()

    cv2.namedWindow("Person Detection", cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            try:
                result = queue_out.get(timeout=1)  # wait for up to one second for an item to become available
                cv2.imshow("Person Detection", result)
                cv2.waitKey(1)  # cv2 doesn't work without this
            except Empty:
                continue

    except KeyboardInterrupt:
        print("Interrupted by user, shutting down...")

    finally:
        producer_process.terminate()
        consumer_process.terminate()
        producer_process.join()
        consumer_process.join()

        cv2.destroyAllWindows()
        camera.release()

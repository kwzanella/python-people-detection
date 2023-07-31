import cv2
from multiprocessing import Process, Queue
from ultralytics import YOLO


# There is a delay because the producer is faster than the consumer.
# This can be resolved by lowering resolution or lowering queue max size.

# Producer/Consumer Architecture:
# - https://www.ni.com/en/support/documentation/supplemental/21/producer-consumer-architecture-in-labview0.html

# Set supported camera resolution here
# Must be multiple of 32 (320x320, 416x416, 448x448, 608x608, etc)
RESOLUTION = (416, 416)

# Define based in available memory and video delay (bigger size = more delay)
QUEUE_SIZE = 20


# Generates frames from device camera
def producer(queue, camera):
    try:
        while camera.isOpened():
            is_read, frame = camera.read()
            if is_read:
                queue.put(frame)
            #print(f"Queue Size = {queue.qsize()}")

    finally:
        camera.release()


# Uses frames as input to YOLO model and show on window
def consumer(queue, model):
    try:
        while True:
            results = model.predict(source=queue.get(), imgsz=RESOLUTION[0], classes=0, device="cpu", verbose=False)
            cv2.imshow("Live People Detection", results[0].plot())
            cv2.waitKey(1)  # cv2 doesn't work without this

    finally:
        cv2.destroyAllWindows()


# Creates new ONNX model based on PyTorch (pt) model
# If pt model is not found, it will automatically download it
def export_new_model():
    model = YOLO(model="models/yolov8n.pt", task="detect")
    model.export(format="onnx", imgsz=RESOLUTION[0], classes=0, dynamic=False, simplify=True)


if __name__ == "__main__":
    #export_new_model()
    model = YOLO(model="models/yolov8n.onnx", task="detect")

    camera = cv2.VideoCapture(0)  # use default camera (index 0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

    queue = Queue(maxsize=QUEUE_SIZE)

    producer_process = Process(target=producer, args=(queue, camera))
    consumer_process = Process(target=consumer, args=(queue, model))

    producer_process.start()
    consumer_process.start()

    try:
        producer_process.join()
        consumer_process.join()
        
    except KeyboardInterrupt:
        print("Interrupted by user, shutting down...")
        
    finally:
        # calling join after terminate is important for resouce cleanup
        producer_process.terminate()
        consumer_process.terminate()
        
        producer_process.join()
        consumer_process.join()
        
        cv2.destroyAllWindows()
        camera.release()

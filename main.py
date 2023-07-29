import cv2
from multiprocessing import Process, Queue
from ultralytics import YOLO

# set supported camera resolution here
# must be multiple of 32 (320x320, 416x416, 448x448, 608x608, etc)
FRAME_WIDTH = 416
FRAME_HEIGHT = 416

"""
Producer/Consumer Architecture:
- https://www.ni.com/en/support/documentation/supplemental/21/producer-consumer-architecture-in-labview0.html
"""

def producer(queue, camera):
    try:
        while camera.isOpened():
            is_read, frame = camera.read()
            if is_read:
                queue.put(frame)
    finally:
        camera.release()


def consumer(queue, model):
    try:
        while True:
            results = model.predict(source=queue.get(), imgsz=FRAME_WIDTH, classes=0, device="cpu", verbose=False)
            cv2.imshow("Live People Detection", results[0].plot())

            if cv2.waitKey(1) & 0xFF == ord("q"):  # press "q" to exit window
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model = YOLO(model="models/yolov8n.onnx", task="detect")

    camera = cv2.VideoCapture(0)  # use default camera (index 0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    queue = Queue()

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

import cv2


# set supported camera resolution here
FRAME_WIDTH = 600
FRAME_HEIGHT = 600

# number of frames to skip between detections
SKIP_FRAMES = 5


def process_frame(frame):
    # detects people in frame
    boxes, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # draws bounding boxes around people
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (0, 0, 255), 2)
        
    return frame


if __name__ == "__main__":
    # initialize HOG (Histogram of Oriented Gradients)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    camera = cv2.VideoCapture(0)  # use default camera (index 0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    frame_count = 0
    try:
        while True:
            is_read, frame = camera.read()
            
            if is_read:
                frame_count += 0
                # only process frame after "SKIP_FRAMES" amount
                if frame_count % SKIP_FRAMES == 0:
                    cv2.imshow("People Detection", process_frame(frame))

            # press "q" to exit window
            if cv2.waitKey(1) == ord('q'):
                break
    
    except cv2.error as e:
        print(f"OpenCV error: {e}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
        
    finally:
        camera.release()
        cv2.destroyAllWindows()

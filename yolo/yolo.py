"""
@software{yolov8_ultralytics,
  author       = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title        = {Ultralytics YOLOv8},
  version      = {8.0.0},
  year         = {2023},
  url          = {https://github.com/ultralytics/ultralytics},
  orcid        = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license      = {AGPL-3.0}
}
"""

from ultralytics import YOLO


if __name__ == "__main__":
    # Load the model
    model = YOLO('yolov8n.pt')  # load a pretrained model

    # Perform inference
    results = model('../test.jpg')

    # Print the results
    results.print()
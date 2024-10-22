import torch
from collections import Counter
import cv2
from ultralytics.yolov5.utils.augmentations import letterbox
from ultralytics.yolov5.utils.general import non_max_suppression, scale_coords
from ultralytics.yolov5.utils.torch_utils import select_device, time_sync
from ultralytics.yolov5.utils.plots import Annotator, colors
from ultralytics.yolov5.models.common import DetectMultiBackend
from ultralytics.yolov5.utils.datasets import LoadStreams, LoadImages
from ultralytics.yolov5.utils.general import check_img_size, check_imshow, increment_path, strip_optimizer
from ultralytics.yolov5.utils.torch_utils import select_device, time_sync
from pathlib import Path
import numpy as np

# Constants for managing the buffer of OCR results
ocr_results = []
max_len = 10  # Adjust the size of the OCR history buffer if needed

def getOCR(image, xyxy):
    """
    Function to extract OCR from a specified region in an image.
    Input:
        image: The image from which to extract text.
        xyxy: Bounding box coordinates for cropping the region of interest.
    Output:
        OCR result (string).
    """
    # Crop the image to the bounding box
    x1, y1, x2, y2 = map(int, xyxy)
    cropped_img = image[y1:y2, x1:x2]
    
    # Apply OCR using Tesseract
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    ocr_result = pytesseract.image_to_string(gray)
    
    return ocr_result.strip()

class YOLOv5Inference:
    def __init__(self, weights, source, img_size=640, conf_thres=0.25, iou_thres=0.45, device=''):
        self.weights = weights
        self.source = source
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = select_device(device)
        self.model = DetectMultiBackend(self.weights, device=self.device)
        self.stride = self.model.stride
        self.img_size = check_img_size(self.img_size, s=self.stride)  # check image size
        self.names = self.model.names

        # Dataloader
        self.dataset = LoadStreams(self.source, img_size=self.img_size, stride=self.stride) if self.source.isnumeric() else LoadImages(self.source, img_size=self.img_size, stride=self.stride)
        
    def detect(self):
        # Run inference
        for path, img, im0s, vid_cap in self.dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = self.model(img)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

            for i, det in enumerate(pred):  # detections per image
                im0 = im0s[i].copy() if isinstance(im0s, list) else im0s

                if len(det):
                    # Rescale boxes from img_size to original size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Annotator for drawing bounding boxes
                    self.annotator = Annotator(im0, line_width=2, example=str(self.names))

                    # Process detections
                    for *xyxy, conf, cls in reversed(det):
                        ocr = getOCR(im0, xyxy)  # OCR result
                        ocr_results.append(ocr)

                        # Maintain a buffer of the last 'max_len' OCR results
                        if len(ocr_results) > max_len:
                            ocr_results.pop(0)

                        # Find the most common OCR result in the buffer
                        most_common_ocr = Counter(ocr_results).most_common(1)[0][0]

                        # Draw bounding box and label with most common OCR result
                        label = f'{most_common_ocr}'
                        self.annotator.box_label(xyxy, label, color=colors(cls, True))

            cv2.imshow(str(path), im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                break

# Example usage
if __name__ == "__main__":
    yolo = YOLOv5Inference(weights='best.pt', source='0')  # '0' for webcam or video path
    yolo.detect()

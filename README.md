# yolov8-yolov11_guide
# YOLOv11 프로젝트

## YOLOv8 vs YOLOv11 비교

| Feature               | YOLOv8                          | YOLOv11                        |
|-----------------------|---------------------------------|--------------------------------|
| **Model Architecture** | CSPDarknet53 + PANet            | Transformer-based architecture with custom modifications |
| **Pretrained Weights** | Available on official repo      | Available (often uses more diverse datasets for training) |
| **Detection Speed**    | Faster than previous YOLO versions | Improved detection speed with better small object detection |
| **Accuracy**           | High mAP on large objects       | Enhanced accuracy, especially for small and occluded objects |
| **Key Improvements**   | Better augmentation techniques and flexible training | Integration of Transformer networks for better feature extraction |
| **Framework Support**  | PyTorch, TensorFlow, ONNX      | PyTorch, TensorFlow, ONNX      |
| **Training Techniques**| AutoAugment, Hyperparameter tuning | Vision Transformers (ViTs), Mixup, CutMix |
| **Supported Backbones**| YOLOv4, YOLOv5, EfficientNet   | YOLOv8, Custom Backbones       |
| **Model Size**         | Relatively smaller and faster  | Larger model size, but more accurate and robust |

## YOLO에서 사용되는 용어 정리

| Term                   | Description |
|------------------------|-------------|
| **YOLO (You Only Look Once)** | An object detection algorithm that predicts class probabilities and bounding box coordinates in a single pass. |
| **mAP (mean Average Precision)** | A metric to evaluate the accuracy of object detection models, averaging precision across all classes. |
| **Anchor Boxes**        | Predefined bounding boxes used during training to match ground truth boxes of different sizes. |
| **Non-Maximum Suppression (NMS)** | A technique used to eliminate redundant bounding boxes that overlap, keeping only the one with the highest score. |
| **Confidence Score**    | The probability that a bounding box contains an object of a certain class. |
| **IoU (Intersection over Union)** | A metric that calculates the overlap between predicted and ground truth bounding boxes. |
| **Backbone**            | The feature extraction network of YOLO. Common backbones are Darknet, ResNet, and EfficientNet. |
| **YOLO Layers**         | The layers of the network where bounding boxes and class predictions are made. |
| **Loss Function**       | The function that measures the error between the predicted and actual values during training. YOLO uses a custom loss function that combines localization, confidence, and classification losses. |
| **Objectness Score**    | A score indicating whether an anchor box contains an object or not. |
| **Bounding Box**        | A rectangle that indicates the location of an object in an image. Defined by its center coordinates (x, y), width (w), and height (h). |
| **Class Prediction**    | The process of predicting the class label (e.g., car, person) for a detected object. |

## 프로젝트 설정 및 학습

1. **환경 설정**:
   - 먼저, YOLOv11을 실행하기 위한 환경을 설정합니다.
     ```bash
     git clone https://github.com/ultralytics/yolov11.git
     cd yolov11
     pip install -r requirements.txt
     ```

2. **데이터셋 준비**:
   - 데이터셋을 준비하고, `data.yaml` 파일을 수정하여 경로와 클래스 정보를 입력합니다.

3. **학습 시작**:
   - 모델을 학습시키기 위해, YOLOv11 학습 명령어를 사용합니다.
     ```bash
     python train.py --data data.yaml --cfg yolov11.yaml --weights '' --batch-size 16 --epochs 50
     ```

## 결론

- **YOLOv8**는 빠른 속도와 좋은 정확도를 자랑하며, 다양한 데이터셋에서 높은 성능을 보입니다.
- **YOLOv11**은 Transformer 기반의 아키텍처를 도입하여 더욱 향상된 정확도를 제공하며, 특히 작은 객체 탐지에서 두각을 나타냅니다.

# YOLOv11 성능 비교

| 모델         | 크기 (픽셀) | mAP@val 50-95 | 속도 CPU ONNX (ms) | 속도 T4 TensorRT10 (ms) | 파라미터 (M) | FLOPs (B) |
|--------------|-------------|--------------|-------------------|------------------------|--------------|-----------|
| **YOLOv11n** | 640         | 39.5         | 56.1 ± 0.8        | 1.5 ± 0.0              | 2.6          | 6.5       |
| **YOLOv11s** | 640         | 47.0         | 90.0 ± 1.2        | 2.5 ± 0.0              | 9.4          | 21.5      |
| **YOLOv11m** | 640         | 51.5         | 183.2 ± 2.0       | 4.7 ± 0.1              | 20.1         | 68.0      |
| **YOLOv11l** | 640         | 53.4         | 238.6 ± 1.7       | 6.2 ± 0.1              | 25.3         | 86.9      |
| **YOLOv11x** | 640         | 54.7         | 462.8 ± 6.7       | 11.3 ± 0.2             | 56.9         | 194.9     |

*위 표는 COCO에서 훈련된 모델들의 성능을 비교한 표입니다.*

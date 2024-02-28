# Customized YOLO based on Ultralytics
## Outline
This project is YOLOv8 project for mobile system.
We will develop and study for lightweight YOLO model.

## How to install
```
git clone https://github.com/az0422/customizedYOLO.git
cd customizedYOLO
pip install .
```

## How to use
Same than Ultralytics YOLOv8

## Included New Models
### Configuration
 - yolov8-mobile.yaml
 - yolov8-mobile-fast.yaml
 - yolov8-mobile-tiny.yaml
 - yolov8nd.yaml

### Pre-trained Weights
 - yolov8-mobile.pt
 - yolov8-mobile-fast.pt
 - yolov8-mobile-tiny.pt
 - yolov8ndn.pt
 - yolov8nds.pt
 - yolov8ndm.pt
 - yolov8ndl.pt
 - yolov8ndx.pt

Train parameters: `optimizer=SGD lr0=0.01 batch=32 epochs=300 data=coco.yaml`

### Performance of models
#### System Environment
 - CPU: Intel Xeon Silver 4216 x2
 - RAM: 192GB DDR4 3200MHz
 - GPU: RTX A5000 x3

#### Performance
| Model | Parameters | GFLOPs | mAP50-95 | Speed<br>GPU|
|-------|------------|--------|----------|-----------------------|
| YOLOv8-mobile | 16M | 34.3 | 43.8% | 11.8ms |
| YOLOv8-mobile-tiny | 8.8M | 19.3 | 41% | 10.2ms |
| YOLOv8-mobile-nano | 4.2M | 10.7 | 36.5% | 6.4ms |
||
| YOLOv8ndn | 3.1M | 9.9 | 34.2% | 7.2ms |
| YOLOv8nds | 9.8M | 25.8 | 40.8% | 7.8ms |
| YOLOv8ndm | 22.8M | 66.6 | 45.7% | 9.3ms |
| YOLOv8ndl | 38.8M | 142.8 | 48.4% | 10.7ms |
| YOLOv8ndx | 60.2M | 221.0 | 49.7% | 12.3ms |
||
| YOLOv8n | 3.15M | 8.7 | 37.1% | 8.4ms |
| YOLOv8s | 11.2M | 28.6 | 44.7% | 9.1ms |
| YOLOv8m | 25.9M | 78.9 | 50.1% | 11.3ms |
| YOLOv8l | 43.7M | 165.2 | 52.9% | 13.4ms |
| YOLOv8x | 68.2M | 257.8 | 54.0% | 13.7ms

data: coco.yaml (batch 1 for inference)

checkpoint: best weights until 300 epochs

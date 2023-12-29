# Customized YOLO based on Ultralytics
## Outline
This project is YOLOv8 project for mobile system.
We will develop and study for lightweight YOLO model.

## How to install
git clone https://github.com/az0422/customizedYOLO.git

cd customizedYOLO

pip install .

## How to use
Same than Ultralytics YOLOv8

## Included New Models
### Configuration
 - YOLOv8-mobile.yaml
 - YOLOv8-mobile-fast.yaml
 - YOLOv8-mobile-tiny.yaml

### Pre-trained Weights
 - YOLOv8-mobile.pt
 - YOLOv8-mobile-fast.pt
 - YOLOv8-mobile-tiny.pt

### Performance of models
#### System Environment
 - CPU: Intel Xeon Silver 4216 x2
 - RAM: 192GB DDR4 3200MHz
 - GPU: RTX A5000 x3

#### Performance
| Model | Parameters | GFLOPs | mAP50-95 | Recall | Precision | Speed<br>GPU|
|-------|------------|--------|----------|--------|-----------|-----------------------|
| YOLOv8-mobile | 16M | 34.3 | 43.8% | 55.3% | 67.3% | 11.8ms |
| YOLOv8-mobile-tiny | 8.8M | 19.3 | 41% | 52.4% | 52.4% | 10.2ms |
| YOLOv8-mobile-nano | 4.2M | 10.7 | 36.5% | 48.8% | 62.6% | 6.4ms |
||
| YOLOv8n | 3.15M | 8.7 | 37.1% | 47.5% | 64.2% | 9.4ms |
| YOLOv8s | 11.2M | 28.6 | 44.7% | 56.1% | 68.3% | 9.5ms |

data: coco.yaml (batch 1 for inference)

checkpoint: best weights until 300 epochs

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
| Model | Parameters | GFLOPS | mAP50-95 | recall | precision | inference speed (GPU) |
|-------|------------|--------|----------|--------|-----------|-----------------------|
| YOLOv8-mobile | 6.09M | 13.2 | 42.0% | 53.6% | 67.4% | 15.8ms |
| YOLOv8-mobile-fast | 5.76M | 12.3 | 40.1% | 51.5% | 66.9% | 10.8ms |
| YOLOv8-mobile-tiny | 1.37M | 2.5 | 26.4% | 38.4% | 53.9% | 6.9ms |
|-------|------------|--------|----------|--------|-----------|-----------------------|
| YOLOv8n | 3.15M | 8.7 | 37.1% | 47.5% | 64.2% | 8.5ms |
| YOLOv8s | 11.2M | 28.6 | 44.7% | 56.1% | 68.3% | 8.5ms |

data: coco.yaml (batch 1 for inference)

checkpoint: best weights ultil 300 epochs

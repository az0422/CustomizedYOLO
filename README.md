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
 - YOLOv8-mobile
 - YOLOv8-mobile-fast
 - YOLOv8-mobile-tiny

### Pre-trained Weights
 - YOLOv8-mobile.pt # has an error
 - YOLOv8-mobile-fast.pt # has an error
 - YOLOv8-mobile-tiny.pt

### Performance of models
#### System Environment
 - CPU: Intel Xeon Silver 4216 x2
 - RAM: 192GB DDR4 3200MHz
 - GPU: RTX A5000 x3

#### Performance
| Model | Parameters | GFLOPS | mAP50-95 | recall | precision | inference speed |
|-------|------------|--------|----------|--------|-----------|-----------------|
| YOLOv8-mobile | - | - | - | - | - | - | - |
| YOLOv8-mobile-fast | - | - | - | - | - | - | - |
| YOLOv8-mobile-tiny | 1.37M | 2.5 | 26.4% | 38.4% | 53.9% | 6.9ms |

data: coco.yaml (batch 1)

checkpoint: epochs 300

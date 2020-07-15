# Image Super Resolution Demo

## Work Flow

1. 將原圖縮小至 30x30 再放大至 90x90 得到輸入圖片(bicubic)

2. 原圖直接縮小至 90x90 得到 ground truth 圖片(90x90 解析度圖片)

3. 將 bicubic 圖片丟入模型並得到輸出圖片

## Environment

- Ubuntu 18.04
- Python 3.6

### Dependencies

- numpy
- pillow (6.1.0)
- scipy (1.2.1)
- pytorch (1.4.0)

## Usage

```
usage: demo.py [-h] [--cuda] [--model MODEL] [--image IMAGE] [--gpus GPUS]
               
optional arguments:
  -h, --help            Show this help message and exit
  --cuda                Use cuda
  --model               Model path. Default=model/model_epoch_50.pth
  --image               Image name. Default=butterfly_GT
  --gpus GPUS           gpu ids (default: 0)
```

**Example**

```
$ python demo.py --cuda --image demo_pic/butterfly.bmp
```

**Output Files**

- origin.bmp - 原圖縮小至 (30, 30) 的圖片

- gt.bmp - 原圖縮小至 (90, 90) 的圖片

- input.bmp - origin.bmp 再利用 bicubic 放大到 (90, 90) 所產生的圖片

- output.bmp - 利用 input.bmp 輸入模型預測出來的圖片，大小為 (90, 90)

## Training


**Prepare Dataset**

Use generate_train.m

**Train Model**

```
usage: train.py [-h] [--cuda] [--gpus GPUS]
               
optional arguments:
  -h, --help            Show this help message and exit
  --cuda                Use cuda
  --gpus GPUS           gpu ids (default: 0)
```
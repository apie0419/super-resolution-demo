import argparse, os, math, torch
from torch.autograd import Variable
from scipy.ndimage import imread
from scipy.misc import imresize
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="PyTorch VDSR Demo")

parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--image", default="", type=str, help="image name")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

opt = parser.parse_args()
cuda = opt.cuda

model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

if not os.path.exists("Output"):
    os.makedirs("Output", exist_ok=True)

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

def colorize(y, ycbcr): 
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

im_ycbcr = imread(opt.image, mode="YCbCr")

im_orgin_ycbcr = imresize(im_ycbcr, (30, 30), interp="bicubic")
im_bicubic_ycbcr = imresize(im_orgin_ycbcr, (90, 90), interp="bicubic")
im_gt_ycbcr = imresize(im_ycbcr, (90, 90), interp="bicubic")

im_input = im_bicubic_ycbcr[:,:,0].astype(float)
bicubic_psnr = PSNR(im_gt_ycbcr[:,:,0].astype(float), im_input, shave_border=3)
print ("Bicubic PSNR: {}".format(bicubic_psnr))

im_input /= 255.

im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

if cuda:
    model = model.cuda()
    im_input = im_input.cuda()
else:
    model = model.cpu()

out = model(im_input)

out = out.cpu()

im_h_y = out.data[0].numpy().astype(np.float32)

im_h_y = im_h_y * 255.
im_h_y[im_h_y < 0] = 0
im_h_y[im_h_y > 255.] = 255.

im_h = colorize(im_h_y[0,:,:], im_bicubic_ycbcr)
im_origin = Image.fromarray(im_orgin_ycbcr, "YCbCr").convert("RGB")
im_gt = Image.fromarray(im_gt_ycbcr, "YCbCr").convert("RGB")
im_bicubic = Image.fromarray(im_bicubic_ycbcr, "YCbCr").convert("RGB")

predict_psnr = PSNR(im_gt_ycbcr[:,:,0].astype(float), im_h_y[0,:,:], shave_border=3)
print ("Predict PSNR: {}".format(predict_psnr))

im_origin.save("Output/origin.bmp")
im_bicubic.save("Output/input.bmp")
im_gt.save("Output/gt.bmp")
im_h.save("Output/output.bmp")
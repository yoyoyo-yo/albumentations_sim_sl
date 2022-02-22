import streamlit as st
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import albumentations as A
import math

#---
# upload image
#---

st.title('Albumentations fast simulator')

'Albumentations is wonderful library for data augmentation. https://github.com/albumentations-team/albumentations '
'This is albumentations result simulation.'
'[caution] Result is no warranty'


st.header('Upload image')
uploaded_file = st.file_uploader('', type=['jpg', 'png'])

if uploaded_file is None:
    img = cv2.imread('images/imori.jpg')[..., ::-1]
    st.image(img, caption='imori.jpg')
else:
    st.image(uploaded_file, caption='Uploaded Image :' + uploaded_file.name)

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = img[..., ::-1]


#---
# parse codes
#---

st.header('Your augmentation')
st.markdown('argument format is definitely A=B')
input_code = st.text_area('', 
    '''A.Resize(height=156, width=156),
A.RandomResizedCrop(height=128, width=128, scale = (0.9,1.0), ratio = (0.9,1.0)),
A.ShiftScaleRotate(p=0.2, shift_limit=0.2, scale_limit=0.2, rotate_limit=45),
A.Normalize(p=1),
A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=2, p=0.1),
A.CoarseDropout(p=0.1, max_holes=16),
A.HorizontalFlip(p=0.2),
A.VerticalFlip(p=0.2),
A.Cutout(p=0.1),
A.MotionBlur(p=0.1),
A.OneOf([
    A.MotionBlur(p=0.1),
    A.GaussianBlur(p=1),
], p=0.1),''')


def parse(text):
    stack = []
    stack_text = ''
    nest_text = ''
    nest_count = 0
    parsed_result = []

    for c in text:

        if any([c == x for x in [' ', '\n']]):
            continue

        if nest_count == 0:
            stack_text += c
        elif nest_count > 0:
            nest_text += c

        if c == '(':
            stack.append('(')

        elif c == ')':
            stack.pop()

            if len(stack) == 0:
                if stack_text.startswith(','):
                    stack_text = stack_text[1:]

                if nest_text != '':
                    nest_text = parse(nest_text)
                else:
                    nest_text = []

                parsed_result.append([stack_text, nest_text])
                stack_text = ''
                nest_text = ''
                nest_count = 0

        elif c == '[':
            stack.append('[')
            nest_count += 1

            if stack_text.endswith('['):
                stack_text = stack_text[:-1]

        elif c == ']':
            stack.pop()
            nest_count -= 1
            
            if nest_count == 0 and nest_text.endswith(']'):
                nest_text = nest_text[:-1]

    return parsed_result


ALB_DIC = {#'AdvancedBlur':A.AdvancedBlur, 
        'Blur':A.Blur, 'CLAHE':A.CLAHE,
        'ChannelDropout':A.ChannelDropout, 'ChannelShuffle':A.ChannelShuffle, 'ColorJitter':A.ColorJitter, 
        'Downscale':A.Downscale, 'Emboss':A.Emboss, 'Equalize':A.Equalize, 'FDA':A.FDA, 'FancyPCA':A.FancyPCA, 
        'FromFloat':A.FromFloat, 'GaussNoise':A.GaussNoise, 'GaussianBlur':A.GaussianBlur, 'GlassBlur':A.GlassBlur,
        'HistogramMatching':A.HistogramMatching, 'HueSaturationValue':A.HueSaturationValue,
        'ISONoise':A.ISONoise, 'ImageCompression':A.ImageCompression, 'InvertImg':A.InvertImg, 
        'MedianBlur':A.MedianBlur, 'MotionBlur':A.MotionBlur, 'MultiplicativeNoise':A.MultiplicativeNoise,
        'Normalize':A.Normalize, 'PixelDistributionAdaptation':A.PixelDistributionAdaptation,
        'Posterize':A.Posterize, 'RGBShift':A.RGBShift, 'RandomBrightnessContrast':A.RandomBrightnessContrast,
        'RandomFog':A.RandomFog, 'RandomGamma':A.RandomGamma, 'RandomRain':A.RandomRain, 'RandomShadow':A.RandomShadow,
        'RandomSnow':A.RandomSnow, 'RandomSunFlare':A.RandomSunFlare, 'RandomToneCurve':A.RandomToneCurve,
        #'RingingOvershoot':A.RingingOvershoot, 
        'Sharpen':A.Sharpen, 'Solarize':A.Solarize, 'Superpixels':A.Superpixels,
        'TemplateTransform':A.TemplateTransform, 'ToFloat':A.ToFloat, 'ToGray':A.ToGray, 'ToSepia':A.ToSepia, 
        #'UnsharpMask':A.UnsharpMask
        'Affine':A.Affine, 'CenterCrop':A.CenterCrop, 'CoarseDropout':A.CoarseDropout, 'Crop':A.Crop,
        'CropAndPad':A.CropAndPad, 'CropNonEmptyMaskIfExists':A.CropNonEmptyMaskIfExists, 
        'ElasticTransform':A.ElasticTransform, 'Flip':A.Flip, 'GridDistortion':A.GridDistortion,
        'GridDropout':A.GridDropout, 'HorizontalFlip':A.HorizontalFlip, 'Lambda':A.Lambda, 'LongestMaxSize':A.LongestMaxSize,
        'MaskDropout':A.MaskDropout, 'NoOp':A.NoOp, 'OpticalDistortion':A.OpticalDistortion, 'PadIfNeeded':A.PadIfNeeded,
        'Perspective':A.Perspective, 'PiecewiseAffine':A.PiecewiseAffine, #'PixelDropout':A.PixelDropout,
        'RandomCrop':A.RandomCrop, 'RandomCropNearBBox':A.RandomCropNearBBox, 'RandomGridShuffle':A.RandomGridShuffle,
        'RandomResizedCrop':A.RandomResizedCrop, 'RandomRotate90':A.RandomRotate90, 'RandomScale':A.RandomScale,
        'RandomSizedBBoxSafeCrop':A.RandomSizedBBoxSafeCrop, 'RandomSizedCrop':A.RandomSizedCrop, 'Resize':A.Resize,
        'Rotate':A.Rotate, 'SafeRotate':A.SafeRotate, 'ShiftScaleRotate':A.ShiftScaleRotate,
        'SmallestMaxSize':A.SmallestMaxSize, 'Transpose':A.Transpose, 'VerticalFlip':A.VerticalFlip,
        'Cutout':A.Cutout, 'OneOf':A.OneOf
        }


def is_num(s):
    try:
        float(s.replace(',', '').replace(' ', '').replace('_', ''))
    except ValueError:
        return False
    else:
        return True


def parse_arg(text):
    args = []
    _text = ''
    stack = []

    for c in text:
        _text += c
        
        if c == '(' or c == '[':
            stack.append(c)

        if c == ')' or c == ']':
            stack.pop()

        if len(stack) == 0 and c == ',':

            if _text.endswith(','):
                _text = _text[:-1]

            if _text != '':
                args.append(_text)

            _text = ''
            stack = []

    if len(stack) == 0:
        if _text.endswith(','):
            _text = _text[:-1]

        if _text != '':
            args.append(_text)

    return args


def convert_arg(key, text):
    val = text

    if is_num(text):
        val = float(text)

        if val - val // 1 == 0:
            val = int(val)

        # if key == 'width' or key == 'height':
        #     val = int(val)

    if (text[0] == '(' and text[-1] == ')') or (text[0] == '[' and text[-1] == ']'):
        text = text[1:-1]
        val = tuple([float(x) if is_num(x) else x for x in text.split(',')])

    return val


def str2alb(text, flist=None):
    if text.startswith('A.'):
        text = text[2:]

    fname = None

    for k in ALB_DIC.keys():
        if text.startswith(k):
            fname = k
        
    if fname is None:
        st.text('cannot find function >> ' + text + ', so this was skipped')
        return None

    # if fname == 'Normalize':
    #     st.text('A.Normalize will be skipped')
    #     return None

    text = text.replace(fname, '')
    if text.startswith('('):
        text = text[1:]
    if text.endswith(')'):
        text= text[:-1]

    args = parse_arg(text)

    params = {}

    for arg in args:
        if arg == '':
            continue
  
        k, v = arg.split('=')
        v = convert_arg(k, v)
        params[k] = v

    if flist is None:
        obj = ALB_DIC[fname](**params)
    else:
        obj = ALB_DIC[fname](flist, **params)

    return obj




def str2transforms(result):
    trans = []

    for func, nest in result:
        if len(nest) == 0:
            func_obj = str2alb(func)
        else:
            flist = str2transforms(nest)
            func_obj = str2alb(func, flist=flist)
        
        if func_obj is not None:
            trans.append(func_obj)

    return trans


def normalize_img(img):
    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)

    img -= img.min()
    img /= img.max()
    img *= 255
    img = img.astype(np.uint8)
    return img



parsed_result = parse(input_code)
transforms_ele = str2transforms(parsed_result)
transforms = transforms_train = A.Compose(transforms_ele)

st.header('Your code was parsed below')
with st.expander('Result'):
    transforms_ele

st.header('Select sample number')
max_img_n = st.slider('', value=40, min_value=1, max_value=200)

COL_N = 5
count = 0

st.header('Simulation result')
st.text('output image is forcely normalized')
for r in range(math.ceil(max_img_n / COL_N)):
    cols = st.columns(COL_N)
    for c in range(COL_N):
        with cols[c]:
            res_img = transforms(image=img.astype(np.float32))['image']
            res_img = normalize_img(res_img)
            st.image(res_img, clamp=True)
            
        count += 1

        if count == max_img_n:
            break

    if count == max_img_n:
        break



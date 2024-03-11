import torch

from torch.utils.data import Dataset, DataLoader

import torch.quantization as quantization
from torch.quantization import quantize_dynamic
from torch.nn.functional import softmax, interpolate

from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver , default_fake_quant , default_weight_fake_quant

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image , normalize, resize
from torchvision.io.image import read_image

from torchcam.methods import CAM, GradCAM, GradCAMpp 
from torchcam.utils import overlay_mask


from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from time import time 
import datetime

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


import logging

now = datetime.datetime.now()
logging.basicConfig(filename=f'quant_{now}.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)

logging.info(f'Starting program at {now}')



def load_image(image_path):
    # Load the input image
    input_image = Image.open(image_path)

    return input_image



# get images from the validation set
image_dir = './ILSVRC/Data/CLS-LOC/val'
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
image_files = image_files[:10000]
image_files = sorted(image_files)


# Old transform function
def transform_image(image):
    # Preprocess the input image
    input_tensor = preprocess(image)

    # Add a batch dimension
    input_batch = input_tensor.unsqueeze(0)

    return input_batch

# Preprocess function for the input images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


models_ = {
    'vgg16': models.vgg16, 
    'resnet50': models.resnet50,
    'densenet121': models.densenet121,
    'mobilenet_v2': models.mobilenet_v2,
    'efficientnet_b0': models.efficientnet_b0,
    'squeeze': models.squeezenet1_1,
}

def load_model(model_name):
    model = models_[model_name](pretrained=True)
    model.train()
    return model



default_fake_quant_int16 = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=-32768, quant_max=32767,
                                            dtype=torch.int16, qscheme=torch.per_tensor_affine, reduce_range=True)

default_weight_fake_quant_int16 = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=-32768, quant_max=32767,
                                                   dtype=torch.int16, qscheme=torch.per_tensor_symmetric, reduce_range=False)


default_fake_quant_int8 = default_fake_quant
default_weight_fake_quant_int8  = default_weight_fake_quant


custom_qconfig_int16 = torch.quantization.QConfig(
   
    activation= default_fake_quant_int16,
    weight= default_weight_fake_quant_int16,

)

custom_qconfig_int8 = torch.quantization.QConfig(  

    activation= default_fake_quant_int8,
    weight= default_weight_fake_quant_int8,

)

quant_configs = [
    {
        'dtype': torch.int16,
        'config' : custom_qconfig_int16
    },
    {
        'dtype': torch.int8,
        'config' : custom_qconfig_int8
    },

]

def quantize_model(model: nn.Module, quant_configs: list, name: ):
    quantized_models = [{'name': name,'dtype':torch.float32,'model':model},]
    for quant_config in quant_configs:
        dtype = quant_config['dtype']
        model.qconfig = quant_config['config']
        quantized_model =  torch.quantization.prepare_qat(model, inplace=False) 
        quantized_models.append({'name': name, 'dtype':dtype,'model':quantized_model})
    logging.info(f"quntized the {name} model to f32  int16  int8")
    return quantized_models

def generate_heatmap(model: nn.Module, image:torch.Tensor, target_layer:nn.Module) -> np.ndarray:

    """
    Generate heatmap from 
    """

    # Initialize GradCAM++ extractor
    cam_extractor = GradCAMpp(model, target_layer=target_layer)
    # Read and preprocess the image
    img = image
    # # Perform model inference
    out = model(img)
    # Generate GradCAM++
    cams = cam_extractor(out.squeeze(0).argmax().item(), out)
    # Overlay CAM on the original image
    heatmap = cams[0].squeeze(0).cpu().numpy()

    return heatmap


def color_heatmap( heatmap: np.ndarray) -> np.ndarray :
    """
    Transform heatmap from grey scale to jet. 
    """
    cmap = cm.get_cmap('jet')
    colored_heatmap = cmap(heatmap)[:, :, :3] 
    return colored_heatmap

def overlay_heatmap(image :np.ndarray ,heatmap :np.ndarray, alpha=0.4 : float)-> np.ndarray :
    """
    Overlay heatmap 
    """
    overlayed_img = Image.fromarray((alpha * np.asarray(image) + (1 - alpha) * heatmap *255).astype(np.uint8))
    return overlayed_img

#  Old resize function
def resize_heatmap(heatmap: np.ndarray, image_size: tuple)-> np.ndarray :
    """
    Resize Image 
    """
    heatmap = Image.fromarray(heatmap).resize(image_size, Image.BICUBIC)
    heatmap = np.array(heatmap)
    heatmap = (heatmap - heatmap.min() )/ (heatmap.max() - heatmap.min())
    return heatmap

# Batch resize function
# def resize_heatmap(heatmap, image_size):
#     heatmap = Image.fromarray(heatmap).resize(image_size[::-1], Image.BICUBIC)
#     heatmap = np.array(heatmap)
#     heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
#     return heatmap


def save_image(image: np.ndarray, output_path : str):
    """
    Save image 
    """
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax) 

def save_map(image : np.ndarray, heatmap : np.ndarray, model_name : str, dtype: np.dtype, output_folder : str):
    """
    Save heatmap 

    """
    # Create a folder to save the results
    save_folder = os.path.join(output_folder, model_name, dtype)
    os.makedirs(save_folder, exist_ok=True)
    # get the image name
    image_name = os.path.basename(image).split('.')[0]
    image_name = image_name + '.png'
    # Attempt to save the converted image
    try:
        plt.imsave(os.path.join(output_folder, model_name, dtype, image_name), heatmap, 
                   cmap='gray', pil_kwargs={'compress_level':0})
    except Exception as e:
        logging.error(f'{image_name} => {e}')
        print(f"Error occurred while saving the image: {image_name} \n", e)
    

def normalize_heatmap(heatmap : [np.ndarray, torch.Tensor]):

    """
    Normalize the heamap 
    """
    
    heatmap = (heatmap - heatmap.mean()) / heatmap.std() 
    return heatmap

def get_target_layer(model:nn.Module) -> nn.Module:
    """
    
    params : 
        model : one of the supported models

    return :  the feature extractor of the network
    """
    if model['name'] == 'vgg16':
        target_layer = model['model'].features[-2]
    
    elif model['name'] == 'resnet50':
        target_layer = model['model'].layer4[-1].conv3
    
    elif model['name'] == 'densenet121':
        target_layer = model['model'].features[-2]
    
    elif model['name'] == 'mobilenet_v2':
        target_layer = model['model'].features[-1]
    
    elif model['name'] == 'efficientnet_b0':
        target_layer = model['model'].features[-1]

    elif model['name'] == 'squeeze':
        target_layer = model['model'].features[-1]

    return target_layer




# single image loop
if __name__ == '__main__':

    results_folder = './results_heatmaps_10000'

    print("=============================================")
    for name,model in models_.items():
        logging.info(f"Starting inference for {name}")
        print(f"{name}: " )
        model = load_model(name)
        quantized_models = quantize_model(model, quant_configs=quant_configs, name=name)
        print("=============================================")
        for quantized_model in quantized_models:
            print(quantized_model['name'],str(quantized_model['dtype']).split('.')[-1])
            logging.info(f"Starting inference for model {quantized_model['name']} _ {str(quantized_model['dtype']).split('.')[-1]} ...")
            pbar = tqdm( enumerate(image_files), colour="green", total = len(image_files))
            m = quantized_model['model']      
            target_layer = get_target_layer(quantized_model)
            for i, image_file in pbar:
                image = load_image(image_file)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                input_batch = transform_image(image)
                input_batch.requires_grad = True
                heatmap = generate_heatmap(m, input_batch,target_layer=target_layer)
                heatmap = resize_heatmap(heatmap, image.size)
                save_map(image_file, heatmap, name, str(quantized_model['dtype']).split('.')[-1], results_folder)
                if i+1 % 1000 == 0: 
                    logging.info(f"Finished inference for {i:06d} images in {quantized_model['name']} _ {str(quantized_model['dtype']).split('.')[-1]}. ")
            logging.info(f"Finished inference for model {quantized_model['name']} _ {str(quantized_model['dtype']).split('.')[-1]}")
        
        logging.info(f"Finished inference for {name}")   
        print("=============================================")
    

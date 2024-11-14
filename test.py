import os

import tqdm
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
from torchvision.utils import save_image

from utils.loss import mean_iou, compute_loss

import warnings
warnings.filterwarnings(action='ignore')

##########################################################################################################################################

def test(CFG, model, test_loader):

    model.eval()
    test_miou = []
    test_loss = []
    gif_image = []

    pixel_std = model.pixel_std
    pixel_mean = model.pixel_mean

    resize = T.Resize((CFG['Resize'], CFG['Resize']))
    gif_path = f"./result/{CFG['Today_Date']}_{CFG['Current_Time']}.gif"
    result_dir = f"./result/{CFG['Today_Date']}_{CFG['Current_Time']}"
    os.mkdir(result_dir)

    with torch.no_grad():
        
        for input, input_name, target in tqdm(test_loader):

            input.to(CFG['Device'])
            target.to(CFG['Device'], dtype=torch.float32)

            test_encode_feature = model.image_encoder(input)
            test_sparse_embeddings, test_dense_embeddings = model.prompt_encoder(points = None, 
                                                                                 boxes = None, 
                                                                                 masks = None
                                                                                 )
            pred, pred_iou = model.mask_decoder(image_embeddings = test_encode_feature,
                                                image_pe = model.prompt_encoder.get_dense_pe(),
                                                sparse_prompt_embeddings = test_sparse_embeddings,
                                                dense_prompt_embeddings = test_dense_embeddings,
                                                multimask_output = False
                                                )
            
            true_iou = mean_iou(pred, target, eps=1e-6)
            loss = compute_loss(pred, target, pred_iou, true_iou)

            test_miou += true_iou.to_list()
            test_loss.append(loss.item())

            input = resize(input)
            input = ((input * pixel_std) + pixel_mean) / 255

            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)

            pred_zero = torch.zeros_like(pred)
            pred_result = torch.cat([pred_zero, pred * 0.5, pred_zero], dim=1)
            input += pred_result

            target_zero = torch.zeros_like(target)
            target_result = torch.cat([target * 0.5, target_zero, target_zero], dim=1)
            input += target_result

            result = torch.cat([pred_result, input, target_result], dim=3)
            result_path = f"{result_dir}/{input_name}"
            save_image(result, result_path)

            gif_image.append(Image.open(result_path))

        _test_miou = np.mean(test_miou)
        _test_loss = np.mean(test_loss)

        print(f"Train_miou: [{_test_miou:.4f}]")   
        print(f"Train_loss: [{_test_loss:.4f}]") 

        gif_image[0].save(gif_path, 
                          save_all=True, 
                          append_images=gif_image[1:], 
                          duration=1000, 
                          loop=0
                          )

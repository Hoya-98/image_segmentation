import sys
import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from segment_anything import sam_model_registry

from utils.cus_dataset import cus_Dataset
from utils.options import Tee
from train import set_seed, train
from test import test

##########################################################################################################################################

def main(CFG):

    log_path = f"./log/{CFG['Today_Date']}_{CFG['Current_Time']}.txt"
    logfile = open(log_path, 'a')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, logfile)
    print('Log 기록을 위한 Memo')  
    print(CFG)
    set_seed(CFG)
    
    model = sam_model_registry[CFG['Model_Size']](checkpoint=CFG['Checkpoint'])

    data = pd.read_csv(CFG['Meta_Path'])
    train_data, tmp_data = train_test_split(data, test_size=0.2, random_state=CFG['Random_Seed'])
    valid_data, test_data = train_test_split(tmp_data, test_size=0.5, random_state=CFG['Random_Seed'])
    train_dataset = cus_Dataset(CFG, model, train_data)
    valid_dataset = cus_Dataset(CFG, model, valid_data)
    test_dataset = cus_Dataset(CFG, model, test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=CFG['Batch_Size'], shuffle=True, num_workers=CFG['Num_Workers'])
    valid_dataloader = DataLoader(valid_dataset, batch_size=CFG['Batch_Size'], shuffle=False, num_workers=CFG['Num_Workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=CFG['Batch_Size'], shuffle=False, num_workers=CFG['Num_Workers'])

    CFG['Role'] = 'Train'
    optimizer = optim.AdamW(model.mask_decoder.parameters(),
                           lr = CFG['Learning_Rate'], 
                           betas = CFG['Betas'], 
                           weight_decay = CFG['Weight_Decay']
                           )
    scheduler = lr_scheduler.MultiStepLR(optimizer, 
                                         milestones=CFG['Milestone'], 
                                         gamma=CFG['Gamma']
                                         )
    model.to(CFG['Device'])
    train(CFG, model, optimizer, scheduler, train_dataloader, valid_dataloader)

    CFG['Role'] = 'Test'
    weight = f"./model/{CFG['Today_Date']}_{CFG['Current_Time']}.pth"
    model.load_state_dict(torch.load(weight, weights_only=True))
    model.to(CFG['Device'])
    test(CFG, model, test_dataloader)

    sys.stdout = original_stdout
    logfile.close()

##########################################################################################################################################

if __name__ == '__main__':

    CFG = {
        'Model_Name' : 'sam',
        'Model_Size' : 'vit_b',
        'Checkpoint' : './checkpoint/sam_vit_b_01ec64.pth',

        'Device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'Today_Date' : datetime.date.today(),
        'Current_Time' : datetime.datetime.now().strftime('%H%M'),

        'Meta_Path' : '../skinex_burn/data/segmentation/test/meta.csv',
        'Image_Dir' : '../skinex_burn/data/segmentation/test/image',
        'Mask_Dir' : '../skinex_burn/data/segmentation/test/mask',

        'Random_Seed' : 42,
        'Epochs' : 50,
        'Batch_Size' : 32,
        'Learning_Rate': 1e-5,
        'Num_Workers' : 8,

        'Resize' : 256,
        'Betas' : [0.9, 0.999],
        'Gamma' : 0.1,
        'Milestone' : [60000, 86666],
        'Weight_Decay' : 0.1
    }

    main(CFG)
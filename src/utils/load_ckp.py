'''
    Utility functions for loading weights.
'''

import gdown
import torch

def load_ckp(checkpoint_fpath, model, optimizer=None, scheduler=None, scaler=None, device='cpu'):
    '''
        Loads entire checkpoint from file.
    '''

    checkpoint = torch.load(checkpoint_fpath, map_location=device)

    model.load_state_dict(checkpoint['model1_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint['epoch']

def download_weights(checkpoint_fpath):
    '''
        Downloads weights from Google Drive.
    '''
    
    gdown.download('https://drive.google.com/uc?id=1lEufQVOETFEIhPdFDYaez31uroq_5Lby', checkpoint_fpath, quiet=False)
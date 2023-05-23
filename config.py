import torch
from pathlib import Path
import data
class CFG:
    """Config class that keeps universal variables accessed from different places"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parent_dir = Path(__file__).parent.absolute()
    models_dir = parent_dir.joinpath('models')
    data_dir = parent_dir.joinpath('data')
    scraped_data_dir = data_dir.joinpath('scraped_data')

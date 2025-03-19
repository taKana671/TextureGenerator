import pathlib
from datetime import datetime


def make_dir(parent, dir_name):
    now = datetime.now()
    # path = f'{dir_name}_{now.strftime("%Y%m%d%H%M%S")}'
    path = parent / f'{dir_name}_{now.strftime("%Y%m%d%H%M%S")}'
    path.mkdir()
    return path
    

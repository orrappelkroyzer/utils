from datetime import datetime
import logging
import json
from pathlib import Path
from typing import Union

LOG_LEVEL = logging.INFO
def get_logger(s):
    logging.basicConfig(format='%(asctime)s|%(levelname)s|%(name)s (%(lineno)d): %(message)s',
                        datefmt="%d/%m/%y %H:%M:%S")
    logger = logging.getLogger(s)
    logger.setLevel(LOG_LEVEL)
    return logger
 
def load_config(config_path: Union[Path, str] = Path("config.json"), output_dir_suffix: str = None, add_date: bool = True) -> dict:
    with open(config_path) as f:
        config = json.load(f)
    output_dir_today = Path(config['output_dir'])
    if output_dir_suffix is not None:
        output_dir_today /= output_dir_suffix
    if add_date:
        output_dir_today /= datetime.now().date().isoformat()
    config['output_dir'] = output_dir_today
    output_dir_today.mkdir(parents=True, exist_ok=True)
    if 'db_dir' in config:
        config['db_dir'] = Path(config['db_dir'])
    return config
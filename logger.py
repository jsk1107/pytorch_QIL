import logging
import os
import json
import logging.config


def get_logger(save_dir, log_config='./logging.json', level=logging.INFO):

    if not os.path.exists(log_config):
        logging.basicConfig(level=level)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    with open(log_config, 'rt', encoding='utf-8') as f:
        cfg = json.load(f)

        for _, handler in cfg['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = f'{save_dir}/{handler["filename"]}'
        logging.config.dictConfig(cfg)
        logger = logging.getLogger()

    return logger
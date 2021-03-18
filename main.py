import numpy as np
import tensorflow as tf

from config import get_config
from util import prepare_dirs_and_logger, save_config

from trainer import Trainer
from trainer3 import Trainer3

def main(config):
    prepare_dirs_and_logger(config)
    tf.compat.v1.set_random_seed(config.random_seed)
    

    
    if 'nn' in config.arch:
        from data_nn import BatchManager
    else:
        from data import BatchManager
    batch_manager = BatchManager(config)


        
    if config.is_3d:
        trainer = Trainer3(config, batch_manager)
    else:
        trainer = Trainer(config, batch_manager)
    
    print("---------------------------------")
    print("|                               |")
    print("|                               |")
    print("|       prepare trainer         |")
    print("|           is done             |")
    print("|                               |")
    print("|                               |")
    print("---------------------------------")
    

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

    

if __name__ == "__main__":
    config, unparsed = get_config()

    print("------------------------------------------------------------------")
    print("|                                                                |")
    print("|                                                                |")
    print("|         tf avail!!                                             |")
    print("| tensorflow version : " + tf.__version__ + "|" )
    #tf.test.is_gpu_available()
    print("|                                                                |")
    print("|                                                                |")
    print("|                                                                |")
    print("------------------------------------------------------------------")

    main(config)

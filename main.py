import tensorflow as tf

from data_loader.fruits360dataset import Fruits360Dataset
from models.resnet50 import resNet50
from trainers.mpl_trainer import MLPTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args



def main():
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    data = Fruits360Dataset(config)

    print(f"{data.train_set}")
 
    model = resNet50

    logger = Logger(config)

    trainer = MLPTrainer(model, data, config, logger)
    
    trainer.train_model()

    # Test model

    # Pront results


if __name__ == '__main__':
    main()

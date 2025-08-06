# Description: This script is used to train a model using PyTorch.

import yaml
import argparse
import os
import json
from model_trainer_torch import ModelTrainer

if __name__ == '__main__':
    # Example usage
    # config = {
        # "img_height": 224,
        # "img_width": 224,
        # "batch_size": 32,
        # "epochs": 5,
        # "patience": 5,
        # "dataset_path": "/scratch/tmp/b_kari02/TinyEnergyBench/data/hymenoptera_data",
        # "output_path": "/scratch/tmp/b_kari02/TinyEnergyBench/results/torch",
        # "model_name": "transfer_learning_model_torch_mbnv2",
        # "model_type": "mobilenet"
    # }
    
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file.")
    parser.add_argument("--working_dir", type=str, default="./", help="Path to local data directory.")
    args=parser.parse_args()

    # Read the configuration file
    try:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
            dataset_subpath=config["dataset_path"] 
            output_subpath=config["output_path"] 
            dataset_path=os.path.join(args.working_dir,dataset_subpath)
            output_path=os.path.join(args.working_dir,output_subpath)
            config["dataset_path"]=dataset_path
            config["output_path"]=output_path
            if not "requires_grad" in config:
                config["requires_grad"]=False
        print("Configuration:", json.dumps(config, sort_keys=True, indent=4))
    except Exception as e:
        # If the configuration file is not found or invalid, print an error message and exit
        print("Error reading the config file.")
        print(e)
        exit(1)
    
    trainer = ModelTrainer(config)
    trainer.load_data()
    trainer.preprocess_data()
    trainer.create_model()
    trainer.train_model()
    trainer.evaluate_model(best_model=True)
    trainer.plot_history()
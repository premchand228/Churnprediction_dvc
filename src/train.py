import logging

from tensorflow.python.keras.api._v1.keras import callbacks
from src.all_utils import read_yaml,create_directory
from src.utils.callbacks import get_callbacks
from src.model import model
import argparse
import pandas as pd
import os
import tensorflow as tf
import joblib


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

def train(config_path,params_path):
    config=read_yaml(config_path)
    params=read_yaml(params_path)
    artifacts=config["artifacts"]
    artifacts_dir=artifacts["ARTIFACTS_DIR"]
    train_file_path=os.path.join(artifacts_dir,artifacts["TRAIN_DIR"],artifacts["TRAIN_FILE_NAME"])
    test_file_path=os.path.join(artifacts_dir,artifacts["TRAIN_DIR"],artifacts["TRAIN_FILE_NAME"])
    model_file_dir=os.path.join(artifacts_dir,artifacts["MODEL_DIR"])
    logging.info("Creating directory")
    create_directory([model_file_dir])
    model_save_path=os.path.join(model_file_dir,artifacts["MODEL_FILE_NAME"])    
    tran_set=pd.read_csv(train_file_path)
    logging.info(tran_set.shape)
    X_train=tran_set.iloc[:,:-1]
    Y_train=tran_set.iloc[:,-1]
    logging.info("train and test files have been read sucessfully")
    logging.info(f" shape of X_train is {X_train.shape}")
    logging.info(f" shape of Y_train is {Y_train.shape}")
    logging.info(f"training started")
    callbacks_path=os.path.join(artifacts_dir,artifacts["CALL_BACK_DIR"])
    callbacks=get_callbacks(callbacks_path)
    model(X_train,Y_train,model_save_path,callbacks)
    logging.info(f"Training is finished")


if __name__=="__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")

    parsed_args = args.parse_args()

    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()



    try:
        logging.info("getting data for train")
        train("config/config.yaml","params.yaml")
        logging.info("training is complete")
    except Exception as e:
        logging.exception(e)
        raise e    
        



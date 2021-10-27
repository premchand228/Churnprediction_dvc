import numpy as np 
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt
import logging
import os
import time
from  src.all_utils import read_yaml,create_directory,save_local_df
import argparse
from  sklearn.preprocessing import MinMaxScaler
#from  sklearn.model_selection import train_test_split

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")
def feature_selection(df):
    df1=df.iloc[:,6:]
    return df1
def handle_catagorical_values(df):
    df['Churn?']=df['Churn?'].astype('category')
    df['Churn?']=df['Churn?'].map({"True.":"1","False.":"0"})
    return df
def scaling(df):
    scaler=MinMaxScaler()
    scaled_data=scaler.fit_transform(df)
    scaled_num_df= pd.DataFrame(data=scaled_data, columns=df.columns,index=df.index)
    #scaled_num_df.head()
    return scaled_num_df

def preprocessing(df):
    df1=handle_catagorical_values(df)
    df2=feature_selection(df1)
    df3=scaling(df2)
    return df3       

def load_fashion_data(config_path,params_path):
    config=read_yaml(config_path)
    params=read_yaml(params_path)
    artifacts=config["artifacts"]
    artifacts_dir=artifacts["ARTIFACTS_DIR"]
    train_dir=os.path.join(artifacts_dir,artifacts["TRAIN_DIR"])
    test_dir=os.path.join(artifacts_dir,artifacts["TEST_DIR"])
    valid_dir=os.path.join(artifacts_dir,artifacts["VALID_DIR"])
    create_directory([train_dir,test_dir,valid_dir])
    train_file_path=os.path.join(train_dir,artifacts["TRAIN_FILE_NAME"])
    test_file_path=os.path.join(test_dir,artifacts["TEST_FILE_NAME"])
    valid_file_path=os.path.join(valid_dir,artifacts["VALID_FILE_NAME"])  
    data_path=os.path.join(artifacts_dir,artifacts["DATA_DIR"],artifacts["DATA_FILE_NAME"])
    print(data_path)
    df=pd.read_csv(data_path)
    len1=len(df)
    train_set =df[0:round(len1*0.8)]
    test_set=df[round(len1*0.8):]
    logging.info(f"getting the mnist data {tf.__version__}")    
    fashion=tf.keras.datasets.mnist
    #logging.info(f" shape of x_train is {X_train.shape} ")
    #logging.info(f" shape of y_train is {Y_train.shape} ")
    train_set=preprocessing(train_set)
    test_set=preprocessing(test_set)
    save_local_df(train_set,train_file_path)
    save_local_df(test_set,test_file_path)
    #logging.info(fashion.size)
    logging.info(valid_file_path)

    #return (X_train,Y_train),(X_valid,Y_valid),(X_test,Y_test)

if __name__=="__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")

    parsed_args = args.parse_args()

    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()



    try:
        logging.info("getting info from mnist dataset")
        load_fashion_data("config/config.yaml","params.yaml")
        logging.info("data is sucessfully loaded")
    except Exception as e:
        logging.exception(e)
        raise e    
        
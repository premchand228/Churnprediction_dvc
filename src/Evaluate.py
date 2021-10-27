import tensorflow as tf 
from src.all_utils import create_directory, read_yaml
from  sklearn.metrics import classification_report
import os
import pandas as pd
import argparse
import logging

def evaulate(config_path, params_path):
    config=read_yaml(config_path)
    artifacts=config["artifacts"]
    artfifacts_dir=artifacts["ARTIFACTS_DIR"]
    model_dir=artifacts["MODEL_DIR"]
    model_file_name=artifacts["MODEL_FILE_NAME"]
    test_dir=artifacts["TEST_DIR"]
    test_file_name=artifacts["TEST_FILE_NAME"]
    model_file_name=os.path.join(artfifacts_dir,model_dir,model_file_name)
    loaded_ann_model=tf.keras.models.load_model(model_file_name)
    test_file_path=os.path.join(artfifacts_dir,test_dir,test_file_name)
    test_set=pd.read_csv(test_file_path)
    true_value=test_set['Churn?']
    predictions=loaded_ann_model.predict(test_set.drop("Churn?",axis=1))
    predictions = (predictions > 0.5)
    report_dir=os.path.join(artfifacts_dir,artifacts["REPORTS_DIR"])
    create_directory([report_dir])
    report_file_path=os.path.join(report_dir,artifacts["REPORT_FILE_NAME"])
    with open(report_file_path, "w") as f:
        f.write(classification_report(true_value,predictions))

if __name__=="__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")

    parsed_args = args.parse_args()

    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()



    try:
        logging.info("getting data for Evaluation")
        evaulate("config/config.yaml","params.yaml")
        logging.info("Evaluation is complete")
    except Exception as e:
        logging.exception(e)
        raise e    
        

   
 
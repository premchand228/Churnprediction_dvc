import tensorflow as tf
from tensorflow.python.keras.api._v1.keras import callbacks
from tensorflow.python.keras.layers.core import Activation, Dense 
from src.utils.callbacks import get_callbacks

def model(X_train,Y_train,model_file_path,callbacks_in):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=14,input_dim=len(X_train.columns),activation="relu"))
    model.add(tf.keras.layers.Dense(units=300,activation="relu"))
    model.add(tf.keras.layers.Dense(units=150,activation="relu"))
    #model.add(tf.keras.layers.Dense(units=75,activation="relu"))
    model.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
    model.compile("Adam",loss=tf.keras.losses.binary_crossentropy,metrics=["accuracy"])
    model.fit(X_train,Y_train,epochs=170,batch_size=32,callbacks=callbacks_in)
    model.save(model_file_path)
import tensorflow as tf
from tensorflow.python.keras.layers.core import Activation, Dense 

def model(X_train,Y_train,model_file_path):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=14,input_dim=len(X_train.columns),activation="relu"))
    model.add(tf.keras.layers.Dense(units=300,activation="relu"))
    model.add(tf.keras.layers.Dense(units=150,activation="relu"))
    model.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
    model.compile("Adam",loss=tf.keras.losses.binary_crossentropy,metrics=["accuracy"])
    model.fit(X_train,Y_train,epochs=200,batch_size=32)
    model.save(model_file_path)
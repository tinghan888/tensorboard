from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import myfun

# cmd  tensorboard --logdir=logs/

df = pd.read_excel("清洗並且數字化Exasens.xlsx",0)

plt.style.use('seaborn-poster')

print("====下載資料==============")
col = ["Imagery_part_min","Imagery_part_avg","Real_part_min","Real_part_avg","Gender","Age","Smoking"]
col_target=["Diagnosis"]


print("====讀取資料==標準化============")

train_x, test_x, train_y, test_y,scaler=myfun.ML_read_dataframe_標準化("清洗並且數字化Exasens.xlsx", col, col_target)
# print("外型大小",train_x.shape,test_x.shape,train_y.shape,test_y.shape)
# print("前面幾筆:",train_x)

# 特徵值跟欄位完全看你的Excel的內容來判斷,這個很重要 切不好都會error
category=4
dim=7


train_y2=tf.keras.utils.to_categorical(train_y,num_classes=(category))
test_y2=tf.keras.utils.to_categorical(test_y,num_classes=(category))

# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=10,
    activation=tf.nn.relu,
    input_dim=dim))
model.add(tf.keras.layers.Dense(units=10,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss = tf.keras.losses.categorical_crossentropy,
    metrics = ['accuracy'])
tensorboard = TensorBoard(log_dir="logs")
history=model.fit(train_x, train_y2,
    epochs=200,batch_size=128,
    callbacks=[tensorboard],
    verbose=1)



#測試
score = model.evaluate(test_x, test_y2, batch_size=128)
print("score:",score)

predict = model.predict(test_x)
print("Ans:",predict)
print("Ans:",np.argmax(predict,axis=-1))

"""
predict2 = model.predict_classes(x_test)
print("predict_classes:",predict2)
print("y_test",y_test[:])
"""


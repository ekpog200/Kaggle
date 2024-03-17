import pandas as pd
import keras
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, Callback
from keras.metrics import mean_absolute_error



data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# print(data_train.shape)
# print(data_test.shape)

#Let's make X and y from data_train
train_y = data_train['label'].astype('float32')
train_x = data_train.drop(['label'], axis=1).astype('int32')

test_x = data_test.astype('float32')
# print(train_x.head())
# print(train_y.head())

train_x = train_x.values.reshape(-1,28,28,1)  # axis - calculated automatically, 28x28 with 1 channel
train_x = train_x / 255.0 # The data is stored from 0 to 255. Converting to 0 from 1
test_x = test_x.values.reshape(-1,28,28,1)
test_x = test_x / 255.0

# print(train_x.shape)
# print(test_x.shape)

print(train_y)
# this will take the value 10 because there are 10 unique values
train_y = to_categorical(train_y, num_classes= len(np.unique(train_y)))
print(train_y)

model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.5),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.5),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
print(model.summary())



Optimizer = Adam(
    learning_rate=0.0005,
    beta_1 = 0.9,
    beta_2= 0.999,
    epsilon=1e-07,
    name='Adam'
)
# let's make my own Callback
class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('accuracy') > 0.9995):
            self.model.stop_training = True

m_callbacks = [
    MyCallback(),
]
model.compile(optimizer=Optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x,train_y, batch_size = 50, epochs = 50, callbacks=m_callbacks)

results = model.predict(test_x)
#print(results)
results = np.argmax(results, axis = 1)
results = pd.Series(results, name='Label')
#print(results)
submission = pd.concat([pd.Series(range(1,28001), name="ImageId"), results], axis=1)
submission.to_csv('submiss.csv', index=False)

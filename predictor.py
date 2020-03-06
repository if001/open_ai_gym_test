import keras
from keras.layers import Input, Dense, Dropout, Concatenate
from keras.models import Model
from keras.optimizers import RMSprop, Adam
import numpy as np


class Predictor():
    def __init__(self):
        self.model: Model = self.create_model()

    def create_model(self):
        input_layer = Input(shape=(3,))
        layer1 = Dense(10, activation='relu')(input_layer)
        layer1 = Dropout(0.3)(layer1)

        layer2 = Dense(5, activation='relu')(layer1)
        layer2 = Dropout(0.3)(layer2)

        m = Concatenate()([layer1, layer2])
        layer3 = Dense(10, activation='relu')(m)
        layer3 = Dropout(0.3)(layer3)
        output_layer = Dense(2, activation='sigmoid')(layer3)
        model = Model(input_layer, output_layer)

        # loss = 'binary_crossentropy'
        loss = 'mse'

        model.compile(loss=loss,
                      optimizer=Adam(),
                      metrics=['accuracy'])
        return model

    def train(self, x_train, y_train):
        self.history = self.model.fit(x_train, y_train,
                                      batch_size=200,
                                      epochs=1,
                                      verbose=1)
        return self.history

    def predict(self, x):
        return self.model.predict_on_batch(x)


def main():
    p = Predictor()
    x_train = np.array([
        [1, 2, 3]
    ])
    y_train = np.array([
        [2, 3]
    ])

    p.model.summary()
    p.train(x_train, y_train)

    x_predict = np.array([
        [1, 2, 3]
    ])

    predict = p.predict(x_predict)
    print(predict)


if __name__ == "__main__":
    main()

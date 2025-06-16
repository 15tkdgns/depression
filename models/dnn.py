from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam

def build_dnn(input_dim, arch_layers, optimizer='Adam', dropout=0.2):
    optimizer_options = {
        'Adam': Adam,
        'RMSprop': RMSprop,
        'Nadam': Nadam,
    }

    model = Sequential()
    model.add(Dense(arch_layers[0], activation='relu', input_dim=input_dim, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    for units in arch_layers[1:]:
        model.add(Dense(units, activation='relu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    model.add(Dense(3, activation='softmax', dtype='float32'))

    opt_instance = optimizer_options[optimizer](learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=opt_instance, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

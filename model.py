from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Add, MaxPooling1D, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model

def residual_block(x, filters, kernel_size=5, strides=1):
    
    shortcut = x

    y = Conv1D(filters, kernel_size, strides=strides, padding='same')(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)

    y = Conv1D(filters, kernel_size, padding='same')(y)
    y = BatchNormalization()(y)

    if x.shape[-1] != filters:
        shortcut = Conv1D(filters, kernel_size=1, strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    y = Add()([shortcut, y])
    y = ReLU()(y)
    return y

def create_advanced_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    x = residual_block(x, filters=64)
    x = residual_block(x, filters=64)
    
    x = residual_block(x, filters=128, strides=2)
    x = residual_block(x, filters=128)
    
    x = residual_block(x, filters=256, strides=2)
    x = residual_block(x, filters=256)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
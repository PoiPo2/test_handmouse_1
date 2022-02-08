from datetime import datetime
from load_config import load_config
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

import logging.config
import numpy as np
import matplotlib.pyplot as plt
import os

logging.config.fileConfig('logging.conf')
logger = logging.getLogger()
logger.info('Try to initialize variables.')
config = load_config()
logger.info('Success to initialize variables.')


def generate_model(param):
    if param == 'model_basic':
        network = Sequential([LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
                              Dense(32, activation='relu'),
                              Dense(len(actions), activation='softmax')])
        network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        network.summary()
        network_hist = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                                   callbacks=[ModelCheckpoint(f'./models/{param}.h5', monitor='val_acc', verbose=1,
                                                              save_best_only=True, mode='auto'),
                                              ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1,
                                                                mode='auto')])
        return network, network_hist
    elif param == 'model_1':
        network = Sequential([LSTM(128, activation='relu', input_shape=x_train.shape[1:3], return_sequences=True),
                              LSTM(64, activation='relu', return_sequences=True),
                              LSTM(32, activation='relu', return_sequences=False),
                              Dense(16, activation='relu'),
                              Dense(len(actions), activation='softmax')])
        network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        network.summary()

        network_hist = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                                   callbacks=[ModelCheckpoint(f'./models/{param}.h5', monitor='val_acc', verbose=1,
                                                              save_best_only=True, mode='auto'),
                                              ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1,
                                                                mode='auto')])
        return network, network_hist
    elif param == 'model_2':
        network = Sequential([LSTM(64, activation='relu', input_shape=x_train.shape[1:3], return_sequences=True),
                              LSTM(128, activation='relu', return_sequences=True),
                              LSTM(64, activation='relu', return_sequences=True),
                              LSTM(32, activation='relu', return_sequences=False),
                              Dense(16, activation='relu'),
                              Dense(len(actions), activation='softmax')])
        network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        network.summary()

        network_hist = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                                   callbacks=[ModelCheckpoint(f'./models/{param}.h5', monitor='val_acc', verbose=1,
                                                              save_best_only=True, mode='auto'),
                                              ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1,
                                                                mode='auto')])
        return network, network_hist
    elif param == 'model_3':
        network = Sequential([LSTM(128, activation='tanh', input_shape=x_train.shape[1:3], return_sequences=True),
                              LSTM(64, activation='tanh', return_sequences=False),
                              Dense(32, activation='relu'),
                              Dense(len(actions), activation='softmax')])
        network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        network.summary()
        network_hist = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                                   callbacks=[ModelCheckpoint(f'./models/{param}.h5', monitor='val_acc', verbose=1,
                                                              save_best_only=True, mode='auto'),
                                              ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=20, verbose=1,
                                                                mode='auto')])
        return network, network_hist
    elif param == 'model_4':
        network = Sequential([LSTM(64, activation='tanh', input_shape=x_train.shape[1:3]),
                              Dense(32, activation='relu'),
                              Dense(len(actions), activation='softmax')])
        network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        network.summary()
        network_hist = network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs,
                                   callbacks=[ModelCheckpoint(f'./models/{param}.h5', monitor='val_acc', verbose=1,
                                                              save_best_only=True, mode='auto'),
                                              ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=20, verbose=1,
                                                                mode='auto')], verbose=1)
        return network, network_hist
    else:
        return False, False


if __name__ == '__main__':
    # 작업을 수행할 디렉토리 경로를 가져옵니다.
    working_directory = config['working_directory']
    # 학습 하려는 액션의 정보를 가져옵니다.
    actions = config['actions']
    # 시퀀스 데이터의 길이 정보를 가져옵니다.
    sequence_length = config['sequence_length']
    # 모델 학습량(epochs) 정보를 가져옵니다.
    epochs = config['epochs']
    logger.info(f'working directory: {working_directory}')
    logger.info(f'actions: {actions}')
    logger.info(f'sequence_length: {sequence_length}')
    logger.info(f'epochs: {epochs}')
    # 작업 경로를 변경합니다.
    try:
        os.chdir(working_directory)
    except FileNotFoundError:
        logger.exception(f'"{working_directory}" 경로를 찾을 수 없습니다.. config.ini 설정을 확인하세요.')
        exit()
    except Exception as E:
        logger.exception(f'Unknown error occurred.. ({E})')
        exit()

    x_train, x_test, y_train, y_test = np.load('./models/train_metadata_test.npy', allow_pickle=True)
    logger.info(f'x_train, y_train: {x_train.shape}, {y_train.shape}')
    logger.info(f'x_test, y_test: {x_test.shape}, {y_test.shape}')
    model, history = generate_model(param='model_4')

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['acc'], 'black', label='train accuracy', linestyle='--', linewidth=0.7)
    plt.plot(history.history['val_acc'], 'green', label='validation accuracy', linewidth=2.5)
    plt.xlabel('Epochs')
    plt.ylabel('Ratio')
    plt.legend(loc='upper left')
    plt.show()

    # fig, loss_ax = plt.subplots(figsize=(16, 10))
    # acc_ax = loss_ax.twinx()
    #
    # loss_ax.plot(history.history['loss'], 'y', label='train loss')
    # loss_ax.plot(history.history['val_loss'], 'r', label='validation loss')
    # loss_ax.set_xlabel('epoch')
    # loss_ax.set_ylabel('loss')
    # loss_ax.legend(loc='upper left')
    #
    # acc_ax.plot(history.history['acc'], 'b', label='train accuracy')
    # acc_ax.plot(history.history['val_acc'], 'g', label='validation accuracy')
    # acc_ax.set_ylabel('accuracy')
    # acc_ax.legend(loc='upper left')
    #
    # plt.show()

from datetime import datetime
from load_config import load_config
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import logging.config
import numpy as np
import os

logging.config.fileConfig('logging.conf')
logger = logging.getLogger()
logger.info('Try to initialize variables.')
config = load_config()
logger.info('Success to initialize variables.')


def generate_metadata(seq_data):
    logger.info(f'seq_data.shape: {seq_data.shape}')
    # 학습에 사용될 훈련 데이터셋 및 라벨을 생성합니다.
    x_data = seq_data[:, :, :-1]
    labels = seq_data[:, 0, -1]
    logger.info(f'x_data.shape: {x_data.shape}')
    logger.info(f'labels.shape: {labels.shape}')
    y_data = to_categorical(labels, num_classes=len(actions))
    logger.info(f'y_data.shape: {y_data.shape}')

    # 훈련 데이터셋 메타데이터를 생성합니다.
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=500)
    train_metadata = [x_train, x_test, y_train, y_test]
    logger.info(f'x_train, y_train: {x_train.shape}, {y_train.shape}')
    logger.info(f'x_test, y_test: {x_test.shape}, {y_test.shape}')
    filename = 'train_metadata_test_sampling' if random_sampling else 'train_metadata_test'
    np.save(f'./models/{filename}', train_metadata)
    logger.info(f'Saving "{filename}.npy" was successful.')


if __name__ == '__main__':
    # 작업을 수행할 디렉토리 경로를 가져옵니다.
    working_directory = config['working_directory']
    # 시퀀스 데이터의 길이 정보를 가져옵니다.
    sequence_length = config['sequence_length']
    # 모델 예측 및 표현하고자 하는 액션의 정보를 가져옵니다.
    actions = config['actions']
    # 랜덤 샘플링과 관련된 설정 값을 가져옵니다.
    random_sampling = True if config['random_sampling'].upper() == 'TRUE' else False
    # 작업 경로를 변경합니다.
    try:
        os.chdir(working_directory)
    except FileNotFoundError:
        logger.exception(f'"{working_directory}" 경로를 찾을 수 없습니다.. config.ini 설정을 확인하세요.')
        exit()
    except Exception as E:
        logger.exception(f'Unknown error occurred.. ({E})')
        exit()
    logger.info(f'working directory: {working_directory}')
    logger.info(f'sequence_length: {sequence_length}')
    logger.info(f'actions: {actions}')
    logger.info(f'random_sampling: {random_sampling}')

    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')

    # dataset 폴더에 있는 'raw_*_timestamp.npy' 파일들을 가져옵니다.
    raw_file_list = []
    for filename in os.listdir('./dataset/'):
        if 'raw' in filename:
            raw_file_list.append(filename)
            logger.debug(f'Append "{filename}" to list(raw_file_list).')
    logger.info(f'raw_file_list: {raw_file_list}')

    # raw_file_list 에서 라벨을 추출하고, 중복 라벨을 제거합니다.
    label_list = []
    for filename in raw_file_list:
        label = filename.split('_')[1]
        label_list.append(label)
        logger.debug(f'Append "{label}" to list(label_list).')
    label_list = list(dict.fromkeys(label_list))
    logger.info(f'label_list: {label_list}')

    # 라벨 데이터를 딕셔너리 타입으로 변경합니다.
    label_dict = {}
    for element in label_list:
        label_dict[element] = []
    logger.debug(f'Create dict(label_dict) variable. {label_dict}')

    # 딕셔너리에 라벨명과 같은 파일을 추가합니다.
    for filename in os.listdir('./dataset/'):
        if 'raw' in filename and 'composited' not in filename:
            label = filename.split('_')[1]
            label_dict[label].append(filename)
            logger.debug(f'Append "{filename}" to dict(label_dict[{label}]).')
    logger.info(f'label_dict: {label_dict}')

    # 각 라벨별로 raw_*_timestamp.npy 파일을 가져옵니다.
    for keyword in label_dict.keys():
        raw_data = np.concatenate([np.load(f'./dataset/{label_dict[keyword][0]}')], axis=0)
        # 파일을 여러개 가져온 경우, 가져온 갯수만큼 이어서 붙입니다.
        for value in label_dict[keyword][1:]:
            raw_data = np.concatenate([raw_data, np.load(f'./dataset/{value}')], axis=0)
        logger.info(f'{keyword}.shape: {raw_data.shape}')

        # sequence_length 길이를 가진 시퀀스 데이터를 생성합니다.
        sequence_data = []
        for sequence in range(len(raw_data) - sequence_length):
            sequence_data.append(raw_data[sequence:sequence + sequence_length])
        # 생성된 시퀀스 데이터(sequence_data)를 넘파이 형태로 변환합니다.
        sequence_data = np.array(sequence_data)
        logger.debug(f'sequence_data.shape: {sequence_data.shape}')

        # 생성된 시퀀스 데이터를 저장합니다.
        file_name = f'sequence({sequence_length})_{keyword}'
        np.save(os.path.join('dataset', file_name), sequence_data)
        logger.info(f'"{file_name}.npy" was saved.\t\tshape: {sequence_data.shape}')

    # 모든 시퀀스 데이터를 가져옵니다.
    sequence_data = []
    sequence_filename = []
    for filename in os.listdir('./dataset/'):
        if f'sequence({sequence_length})' in filename:
            file = np.load(f'dataset/{filename}')
            logger.info(f'Load "{filename}.npy"')
            sequence_data.append(file)
            logger.debug(f'Append "{file.shape}" to list(sequence_data).')
            sequence_filename.append(filename)
            logger.debug(f'Append "{filename}" to list(sequence_filename).')

    # 균등한 학습 데이터셋을 구성하기 위하여, 시퀀스 데이터 중 가장 적은 시퀀스를 가진 데이터셋을 구합니다.
    sequence_length_list = []
    for index in range(len(sequence_data)):
        sequence_length_list.append(sequence_data[index].shape[0])
        logger.debug(f'Append "{sequence_data[index].shape[0]}" to list(sequence_length_list).')
    logger.info(f'Minimum length of sequence data: {min(sequence_length_list)}')

    if random_sampling:
        # 랜덤 샘플링을 하기 위해 각 시퀀스 데이터별로 최소 갯수만큼 인덱스를 무작위로 추출합니다. (비복원 추출)
        extract_index = []
        for index in range(len(sequence_data)):
            extractor = np.random.choice(len(sequence_data[index]), min(sequence_length_list), replace=False)
            extract_index.append(extractor)
            logger.debug(f'Append "{extractor[:5]}..." to list(extract_index).')

        # 비복원으로 랜덤 추출한 인덱스를 이용하여 시퀀스 데이터를 재구성합니다.
        renew_sequence_data = []
        for index in range(len(sequence_data)):
            renew_sequence_flatten_data = sequence_data[index][extract_index[index][0]]
            for extracted_index in extract_index[index][1:]:
                renew_sequence_flatten_data = np.concatenate([renew_sequence_flatten_data, sequence_data[index][extracted_index]], axis=0)
            label_name = sequence_filename[index].split('_')[1].split('.')[0]
            logger.debug(f'renew_{label_name}.shape: {renew_sequence_flatten_data.shape}')

            # 재구성된 데이터를 정의된 시퀀스 길이에 맞춰 구성합니다.
            sequence_storage = []
            for sequence in range(0, len(renew_sequence_flatten_data), sequence_length):
                sequence_storage.append(renew_sequence_flatten_data[sequence:sequence + sequence_length])
            sequence_storage = np.array(sequence_storage)
            renew_sequence_data.append(sequence_storage)
            logger.info(f'sequence_storage({label_name}).shape: {sequence_storage.shape}')

        sequence_data = renew_sequence_data[0]
        print(f'sequence_data.shape: {sequence_data.shape}')
        for index in range(1, len(renew_sequence_data)):
            sequence_data = np.concatenate([sequence_data, renew_sequence_data[index]], axis=0)
        print(f'sequence_data.shape: {sequence_data.shape}')

        generate_metadata(seq_data=sequence_data)
    else:
        data = sequence_data[0][:min(sequence_length_list)]
        for index in range(1, len(sequence_data)):
            data = np.concatenate([data, sequence_data[index][:min(sequence_length_list)]], axis=0)

        generate_metadata(seq_data=data)

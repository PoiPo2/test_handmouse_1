from datetime import datetime
from load_config import load_config
import cv2
import logging.config
import mediapipe as mp
import numpy as np
import os
import time

logging.config.fileConfig('logging.conf')
logger = logging.getLogger()
logger.info('Try to initialize variables.')
config = load_config()
logger.info('Success to initialize variables.')


if __name__ == '__main__':
    # Configuration to Mediapipe-packages
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 작업을 수행할 디렉토리 경로를 가져옵니다.
    working_directory = config['working_directory']
    # 수집하고자 하는 액션 리스트 정보를 가져옵니다.
    train_actions = [config['train_actions']] if type(config['train_actions']) == str else config['train_actions']
    # 데이터 수집 시간 값을 가져옵니다.
    seconds_for_collection = config['seconds_for_collection']
    # 다음 데이터 수집을 위한 준비 시간 값을 가져옵니다.
    seconds_for_wait = config['seconds_for_wait']
    # 설정하고자 하는 카메라의 해상도 값을 가져옵니다.
    cam_width, cam_height = config['cam_width'], config['cam_height']
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
    logger.info(f'train_actions: {train_actions}')
    logger.info(f'seconds_for_collection: {seconds_for_collection} seconds.')
    logger.info(f'seconds_for_wait: {seconds_for_wait} seconds.')
    logger.info(f'Video resolution: {cam_width, cam_height}')

    cam = cv2.VideoCapture(0)
    logger.info(f'Video capture object creation success.')
    os.makedirs('dataset', exist_ok=True)
    while cam.isOpened():
        for index, action in enumerate(train_actions):
            data = []
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, dsize=(cam_width, cam_height), interpolation=cv2.INTER_AREA)
            notice = f'Start "{action}" data collection after {seconds_for_wait} seconds.'
            cv2.putText(frame, notice, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), thickness=2)
            cv2.imshow('dataset maker', frame)
            cv2.moveWindow('dataset maker', 10, 10)
            logger.info(f'Prepare for data collection with a "{action}".')
            if cv2.waitKey(seconds_for_wait * 1000) == ord('q'):
                logger.info(f'User has stopped recording action "{action}".')
                continue

            start_time = time.time()
            logger.info(f'Start collecting "{action}" data.')
            try:
                while time.time() - start_time < seconds_for_collection:
                    ret, frame = cam.read()
                    frame = cv2.flip(frame, 1)
                    frame = cv2.resize(frame, dsize=(cam_width, cam_height), interpolation=cv2.INTER_AREA)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    if result.multi_hand_landmarks is not None:
                        for res in result.multi_hand_landmarks:
                            joint = np.zeros((21, 4))
                            for idx, lm in enumerate(res.landmark):
                                joint[idx] = [lm.x, lm.y, lm.z, lm.visibility]

                            print(f'No.8 point {round(joint[8][0], 3), round(joint[8][1], 3), round(joint[8][2], 3)}')
                            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 19, 20], :3]
                            v = v2 - v1
                            # Normalization
                            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                            # Get angle using arc-cosine of dot products
                            angle = np.arccos(np.einsum('nt, nt->n',
                                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                            angle = np.degrees(angle)

                            # Labeling
                            angle_label = np.array([angle], dtype=np.float32)
                            angle_label = np.append(angle_label, index)
                            flat_data = np.concatenate([joint.flatten(), angle_label])
                            data.append(flat_data)

                            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

                    timer = seconds_for_collection - int(time.time() - start_time)
                    cv2.putText(frame, f'Recoding "{action.upper()}" action.', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 100, 100), thickness=2)
                    if timer >= int(seconds_for_collection * 0.5):
                        cv2.putText(frame, str(timer), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)
                    elif int(seconds_for_collection * 0.2) <= timer < int(seconds_for_collection * 0.5):
                        cv2.putText(frame, str(timer), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), thickness=2)
                    else:
                        cv2.putText(frame, str(timer), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), thickness=2)
                    cv2.imshow('dataset maker', frame)
                    cv2.moveWindow('dataset maker', 10, 10)
                    if cv2.waitKey(1) == ord('q'):
                        raise StopIteration
            except StopIteration:
                logger.info(f'Stop "{action}" data collection.')
                continue
            else:
                logger.info(f'Successfully completed "{action}" data collection.')
                data = np.array(data)
                logger.info(f'{action}.shape: {data.shape}')
                file_name = f'raw_{action}_{datetime.now().strftime("%y%m%d_%H%M%S")}'
                np.save(os.path.join('dataset', file_name), data)
                logger.info(f'"{file_name}.npy" was saved.')
        break
    try:
        cam.release()
    except Exception as E:
        logger.exception(f'Unknown error occurred while adding data.. ({E})')
    else:
        logger.info(f'Stop using the video device.')
    finally:
        cv2.destroyAllWindows()

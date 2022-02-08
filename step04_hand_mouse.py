from load_config import load_config
from tensorflow.keras.models import load_model
from threading import Thread
import cv2
import logging.config
import mediapipe as mp
import numpy as np
import os
import pyautogui
import time
from datetime import datetime

logging.config.fileConfig('logging.conf')
logger = logging.getLogger()
logger.info('Try to initialize variables.')
config = load_config()
logger.info('Success to initialize variables.')


def initialize_thread():
    # 모션을 인식하여 해당 모션에 대한 분류 결과값을 리턴하는 스레드 실행
    while True:
        try:
            thread_predict_motion = Thread(target=predict_motion, name='predict_motion')
            thread_predict_motion.daemon = True
            thread_predict_motion.start()
        except Exception as Err:
            logger.exception(f'Cannot launched "{thread_predict_motion.name}" thread.. retry after 3 seconds.')
            time.sleep(3)
        else:
            logger.info(f'"{thread_predict_motion.name}" thread start.')
            break

    # Stop / Release
    while True:
        try:
            thread_manage_stop_and_release = Thread(target=manage_stop_and_release, name='manage_stop_and_release')
            thread_manage_stop_and_release.daemon = True
            thread_manage_stop_and_release.start()
        except Exception as Err:
            logger.exception(f'Cannot launched "{thread_predict_motion.name}" thread.. retry after 3 seconds.')
            time.sleep(3)
        else:
            logger.info(f'"{thread_manage_stop_and_release.name}" thread start.')
            break


def execute_action(param):
    global MOVE_FLAG, MANAGE_SNR_FLAG
    if param == actions[0]:
        if MOVE_FLAG:
            # 프레임 안에서 8번 랜드마크의 x, y 좌표를 획득합니다.
            frame_pointer = {'x': int(res.landmark[8].x * frame.shape[1]), 'y': int(res.landmark[8].y * frame.shape[0])}
            # 프레임 해상도와 모니터 해상도가 다르기 때문에 실제 이동을 위한 보정값을 산출합니다.
            move_factor = {'x': screen_width / cam_width, 'y': screen_height / cam_height}
            # 프레임 안에서의 x, y 좌표에 비례하여 실제 마우스 커서를 이동합니다.
            real_pointer = {'x': int(frame_pointer['x'] * move_factor['x']),
                            'y': int(frame_pointer['y'] * move_factor['y'])}
            try:
                pyautogui.moveTo(real_pointer['x'], real_pointer['y'], cursor_smoothing)
            except Exception as Err:
                logger.exception(f'Unknown error occurred.. ({Err})')
            else:
                logger.info(f'Move the cursor to {real_pointer["x"], real_pointer["y"]}')
        else:
            pass
            # logger.info(f'Locked move the cursor.')
    elif param == actions[1]:
        if not MOVE_FLAG:
            # pyautogui.click(button='left')
            logger.info(f'Clicked {pyautogui.position()[0], pyautogui.position()[1]} point. [{pred_val}]')
            time.sleep(0.5)
        else:
            # 프레임 안에서 8번 랜드마크의 x, y 좌표를 획득합니다.
            frame_pointer = {'x': int(res.landmark[8].x * frame.shape[1]), 'y': int(res.landmark[8].y * frame.shape[0])}
            # 프레임 해상도와 모니터 해상도가 다르기 때문에 실제 이동을 위한 보정값을 산출합니다.
            move_factor = {'x': screen_width / cam_width, 'y': screen_height / cam_height}
            # 프레임 안에서의 x, y 좌표에 비례하여 실제 마우스 커서를 이동합니다.
            real_pointer = {'x': int(frame_pointer['x'] * move_factor['x']),
                            'y': int(frame_pointer['y'] * move_factor['y'])}
            try:
                pyautogui.moveTo(real_pointer['x'], real_pointer['y'], cursor_smoothing)
            except Exception as Err:
                logger.exception(f'Unknown error occurred.. ({Err})')
            else:
                logger.info(f'Move the cursor to {real_pointer["x"], real_pointer["y"]} - Replaces the click action. [{pred_val}]')
            # logger.info(f'Locked click action.')
    elif param == actions[2]:
        MANAGE_SNR_FLAG = True
        # if MOVE_FLAG:
        #     MOVE_FLAG = False
        #     logger.info(f'Stop!!!')
        #     # time.sleep(1)
        # else:
        #     logger.info(f'Release.')
    # elif param == actions[3]:
    #     print('release')
    # elif param == actions[3]:
    #     if not MOVE_FLAG:
    #         MOVE_FLAG = True
    #         logger.info(f'Release.')
    #         time.sleep(1)
    # elif param == actions[4]:
    #     pyautogui.click(button='left', clicks=2, interval=0.25)
    #     logger.info(f'Double clicked.')
    #     time.sleep(1)
    else:
        filename = f'{datetime.now().strftime("%y%m%d_%H%M%S%f")}.png'
        cv2.imwrite(f'./samples/{filename}', frame)
        pass


def manage_stop_and_release():
    global MANAGE_SNR_FLAG, MOVE_FLAG
    while True:
        if MANAGE_SNR_FLAG:
            MOVE_FLAG = not MOVE_FLAG
            logger.info(f'MOVE_FLAG: {MOVE_FLAG} [{pred_val}]')
            time.sleep(1)
            MANAGE_SNR_FLAG = False
        time.sleep(0.1)


def predict_motion():
    global PREDICT_FLAG, this_action, pred_val
    while True:
        if PREDICT_FLAG:
            y_predict = model.predict(input_data).squeeze()
            i_predict = int(np.argmax(y_predict))
            pred_val = []
            for val in y_predict:
                pred_val.append(round(float(val), 4))
            confidence = y_predict[i_predict]
            # print(pred_val)
            # logger.info(f'Confidence({actions[i_predict]}): {round(float(confidence), 4):.4f} {pred_val}')
            if confidence < 0.9:
                continue
            else:
                action = actions[i_predict]
                action_sequence.append(action)
                if len(action_sequence) < 3:
                    continue

                this_action = ''
                this_action = action
                # if action_sequence[-1] == action_sequence[-2] == action_sequence[-3]:
                #     this_action = action
                execute_action(param=this_action)
                # cv2.putText(frame, f'{this_action.upper()}', org=(int(res.landmark[8].x * frame.shape[1]),
                #                                                   int(res.landmark[8].y * frame.shape[0] + 20)),
                #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                PREDICT_FLAG = False
        time.sleep(0.01)


if __name__ == '__main__':
    # Configuration to Mediapipe-packages
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 작업을 수행할 디렉토리 경로를 가져옵니다.
    working_directory = config['working_directory']
    # 설정하고자 하는 카메라의 해상도 값을 가져옵니다.
    cam_width, cam_height = config['cam_width'], config['cam_height']
    # 시퀀스 데이터의 길이 정보를 가져옵니다.
    sequence_length = config['sequence_length']
    # 모델 예측 및 표현하고자 하는 액션의 정보를 가져옵니다.
    actions = config['actions']
    # 마우스 커서의 이동과 관련된 설정 값을 가져옵니다.
    cursor_smoothing = config['cursor_smoothing'] / 1000
    cursor_smoothing = 0 if cursor_smoothing < 0 or cursor_smoothing > 1000 else cursor_smoothing
    # Click 이벤트의 지연 시간 값을 가져옵니다.
    delay_click = config['delay_click'] / 1000
    # 인공지능 모델을 가져옵니다.
    model = load_model('./models/model_4.h5')
    # 사용자의 실제 디스플레이 해상도 값을 획득합니다.
    screen_width, screen_height = pyautogui.size()
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
    logger.info(f'cursor_smoothing: {cursor_smoothing}')
    logger.info(f'delay_click: {delay_click:.2f} seconds')
    logger.info(f'Video resolution: {cam_width, cam_height}')
    logger.info(f'Monitor resolution: {screen_width, screen_height}')

    # 멀티 스레드 실행, 스레드에 사용되는 변수 초기화
    PREDICT_FLAG = False
    MOVE_FLAG = True
    MANAGE_SNR_FLAG = False
    pred_val = None
    initialize_thread()

    cap = cv2.VideoCapture(0)

    sequence = []
    action_sequence = []
    check_time = 0
    this_action = ''
    while cap.isOpened():
        # FPS
        start_time = time.time()
        ret, frame = cap.read()
        # 영상의 출력을 좌우 방향으로 반전 시킵니다.
        frame = cv2.flip(frame, 1)
        # 영상의 출력 사이즈를 조절합니다. (임시)
        frame = cv2.resize(frame, dsize=(cam_width, cam_height), interpolation=cv2.INTER_AREA)
        # 영상의 채널 설정을 BGR 채널에서 RGB 채널로 변환합니다.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                joint_position = np.zeros((21, 2))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                    # 각 세그먼트의 좌표를 기록하되, 해상도 범위를 벗어난 경우 최대, 최소 값으로 보정합니다.
                    loc_x, loc_y = int(lm.x * cam_width), int(lm.y * cam_height)
                    if loc_x < 0:
                        loc_x = 0
                    elif loc_x > cam_width:
                        loc_x = int(cam_width)

                    if loc_y < 0:
                        loc_y = 0
                    elif loc_y > cam_height:
                        loc_y = int(cam_height)
                    joint_position[j] = [int(loc_x), int(loc_y)]
                # 프레임에 세그먼트 위치를 표시할 사각형(rectangle)을 표현하기 위해, 최대 최소 좌표를 구합니다.
                min_loc_x = cam_width
                max_loc_x = 0
                min_loc_y = cam_height
                max_loc_y = 0
                for index in range(len(joint_position)):
                    if min_loc_x >= joint_position[index][0]:
                        min_loc_x = joint_position[index][0]
                    if max_loc_x <= joint_position[index][0]:
                        max_loc_x = joint_position[index][0]
                    if min_loc_y >= joint_position[index][1]:
                        min_loc_y = joint_position[index][1]
                    if max_loc_y <= joint_position[index][1]:
                        max_loc_y = joint_position[index][1]

                # 사각형 그리기
                # print(f'{(int(min_loc_x), int(min_loc_y))}, {(int(max_loc_x), int(max_loc_y))}')
                # cv2.rectangle(frame, (0, 0), (150, 150), color=(0, 255, 255), thickness=2)
                cv2.rectangle(frame, (int(min_loc_x), int(min_loc_y)), (int(max_loc_x), int(max_loc_y)),
                              color=(0, 255, 255), thickness=2)
                # print(min_loc_x, max_loc_x, min_loc_y, max_loc_y)
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
                v = v2 - v1  # [20, 3]
                # Normalize
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                # Convert radian to degree
                angle = np.degrees(angle)
                data = np.concatenate([joint.flatten(), angle])
                sequence.append(data)
                mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)
                if len(sequence) < sequence_length:
                    continue

                input_data = np.expand_dims(np.array(sequence[-sequence_length:], dtype=np.float32), axis=0)
                PREDICT_FLAG = True

        fps = 1 / (start_time - check_time)
        check_time = start_time
        fps = str(int(fps))
        cv2.putText(frame, f'FPS: {fps}', org=(20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, f'{this_action}', org=(10, cam_height - 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
        cv2.imshow(f'Frame({cam_width} * {cam_height}), Screen({screen_width} * {screen_height})', frame)
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            break

    cv2.destroyAllWindows()

import cv2
import numpy as np
import os

RAW_IMG_DIR = 'client_data/raw_img'
RESULT_IMG_DIR = 'client_data/result_img'
SAVE_DIR = 'client_data/annotation_img'

data_number = 0
select_list = []
index = 1000

fn = os.listdir(RAW_IMG_DIR)
file_tuples = [(int(f.split(".")[0]), f) for f in fn if f[:-4].isdigit()]
sorted_files = sorted(file_tuples, key=lambda x: x[0])
file_names = []

for file_tuple in sorted_files:
    file_names.append(file_tuple[1])

while(True):
    if len(file_names) == 0:
        print('記録データがありません')
        break
    file_name = file_names[index]
    im_raw = cv2.imread(f'{RAW_IMG_DIR}/{file_name}')
    im_result = cv2.imread(f'{RESULT_IMG_DIR}/{file_name}')
    if not (im_raw is None or im_result is None):    
        img = cv2.hconcat([im_raw, im_result])
        white_image = np.ones((50, img.shape[1], 3), dtype=np.uint8) * 255
        img = cv2.vconcat([white_image, img])
        if index in select_list:
            cv2.putText(img, 'data selected!', (50,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 192, 0), 2, cv2.LINE_AA)
        cv2.imshow('selector', img)

        key = cv2.waitKey(0) & 0xFF

    if key == 27:   #esc
        break
    if key == 13:   #enter
        if index not in select_list:
            select_list.append(index)
        index += 1
        print(f'データを保存対象として選択 ({index}/{len(file_names)})')
    if key == ord('k'):
        break
    elif key == 8:    #backspace
        if index in select_list:
            select_list.remove(index)
            print(f'データを削除 ({index}/{len(file_names)})')
    elif key == 81:    #<-
        if index > 0:
            index -= 1
            print(f'({index}/{len(file_names)})')
        continue
    elif key == 83:   #->
        index += 1
        print(f'({index}/{len(file_names)})')

    if index >= len(file_names):
        key = 81
        print('データとして問題が無ければkキーを押してください')

if key == ord('k'):
    for i in select_list:
        im_raw = cv2.imread(f'{RAW_IMG_DIR}/{file_names[i]}')
        cv2.imwrite(f'{SAVE_DIR}/output_{data_number}.png', im_raw)
        data_number += 1

cv2.destroyAllWindows()
import cv2
import numpy as np
import os
import pandas as pd
import csv

TARGET_CSV = 'bottle_3d.csv'

RAW_IMG_DIR = 'client_data/raw_img'
RESULT_IMG_DIR = 'client_data/result_img'
DATA_CSV_DIR = 'client_data/csv_data'
TRAIN_CSV_DIR = 'training_csv'

CLASS_DIC = {'action_3d.csv' : ['None', 'Walking', 'Standing', 'Sitting', 'Laying'],
             'book_3d.csv' : ['None', 'Reading Book', 'Holding Book'],
             'bottle_3d.csv' : ['None', 'Drinking', 'Holding Bottle'],
             'cellphone_3d.csv' : ['None', 'Calling Phone', 'Holding Phone'],
             'cushion_3d.csv' : ['None', 'Holding Cushion'],
             'food_3d.csv' : ['None', 'Eating', 'Holding Banana'],
             'keyboard_3d.csv' : ['None', 'Working on Computer']}

df = pd.read_csv(f'{DATA_CSV_DIR}/{TARGET_CSV}') 
X = df.drop('data_id', axis=1)
y = df['data_id']

data_number = 0
select = 0
data_dict = {}
class_list = CLASS_DIC[TARGET_CSV]
keys = ''
num = 0
index = 0
for cls in class_list:
    num += 1
    keys += f'{num}:{cls} '

while(True):
    data_number = y[index]
    im_raw = cv2.imread(f'{RAW_IMG_DIR}/{data_number}.png')
    im_result = cv2.imread(f'{RESULT_IMG_DIR}/{data_number}.png')
    img = cv2.hconcat([im_raw, im_result])
    white_image = np.ones((100, img.shape[1], 3), dtype=np.uint8) * 255
    img = cv2.vconcat([white_image, img])
    cv2.putText(img, str(X['class'][index]), (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, keys, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    if index in data_dict:
        cv2.putText(img, data_dict[index], (50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (64, 255, 64), 2, cv2.LINE_AA)
    cv2.imshow('selector', img)

    key = cv2.waitKey(0) & 0xFF

    if key == 27:   #esc
        break
    if key == ord('k'):
        break
    elif key == 8:    #backspace
        if data_number in data_dict:
            del data_dict[data_number]
            print(f'データを削除 ({index}/{len(y)})')
    elif key == 81:    #<-
        if index > 0:
            index -= 1
            print(f'({index}/{len(y)})')
        continue
    elif key == 83:   #->
        index += 1
        print(f'({index}/{len(y)})')
    elif key == ord('1'):
        if len(class_list) < 1:
            continue
        if X['class'][index] != class_list[0]:
            data_dict[index] = class_list[0]
        index += 1
        print(f'{class_list[0]}を選択しました。 ({index}/{len(y)})')
    elif key == ord('2'):
        if len(class_list) < 2:
            continue
        if X['class'][index] != class_list[1]:
            data_dict[index] = class_list[1]
        index += 1
        print(f'{class_list[1]}を選択しました。 ({index}/{len(y)})')
    elif key == ord('3'):
        if len(class_list) < 3:
            continue
        if X['class'][index] != class_list[2]:
            data_dict[index] = class_list[2]
        index += 1
        print(f'{class_list[2]}を選択しました。 ({index}/{len(y)})')
    elif key == ord('4'):
        if len(class_list) < 4:
            continue
        if X['class'][index] != class_list[3]:
            data_dict[index] = class_list[3]
        index += 1
        print(f'{class_list[3]}を選択しました。 ({index}/{len(y)})')
    elif key == ord('5'):
        if len(class_list) < 5:
            continue
        if X['class'][index] != class_list[4]:
            data_dict[index] = class_list[4]
        index += 1
        print(f'{class_list[4]}を選択しました。 ({index}/{len(y)})')

    if index >= len(y):
        index = len(y)-1
        print('データとして問題が無ければkキーを押してください')

if key == ord('k'):
    for i, data in data_dict.items():
        with open(f'{TRAIN_CSV_DIR}/{TARGET_CSV}',mode='a' ,newline='') as f:
            csv_writer =csv.writer(f, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
            X.iat[i,0] = data
            csv_writer.writerow(X.iloc[i,:])

cv2.destroyAllWindows()
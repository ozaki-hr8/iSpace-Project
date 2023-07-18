import socket;
import threading
import json
import time
import csv
import numpy as np
import cv2
import math
from utils.point_handler import PointHandler, Cluster, PointMap
from utils.ip_handler import get_ip, get_port

# 接続待ちするサーバのホスト名とポート番号を指定
HOST = '127.0.1.1'
PORT = get_port()

#カメラ数
CAMERA_AMOUNT = 6

data_dict = {}
person_map = PointMap(map_img=cv2.imread("map.png"))

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def add_location_data(data_name, data):
    global data_dict
    if data_name in data_dict:
        data_dict[data_name].append(data)
    else:
        data_dict[data_name] = [data]
    return data_dict

def main():
    sock.bind((HOST, PORT))
    sock.listen(CAMERA_AMOUNT)

    #マップ表示用のスレッドを立ち上げる
    map_thread = threading.Thread(target=map_view)
    map_thread.start()

    while True:
        try:
            conn, addr = sock.accept()
        except KeyboardInterrupt:
            sock.close()
            exit()
            break
        # アドレス確認
        print("[アクセスを確認] => {}:{}".format(addr[0], addr[1]))
        # スレッド作成
        thread = threading.Thread(target=loop_handler, args=(conn, addr))
        thread.start()

def loop_handler(connection, address):
    global person_handler
    global person_data
    while True:
        try:
            #クライアント側から受信
            res = connection.recv(4096)
        except Exception as e:
            print(e)
            break
        try:
            json_dic = json.loads(res.decode('utf-8'))
            if not json_dic:
                continue
            if json_dic == {}:
                continue
            for k, v in json_dic.items():
                if k != 'timestamp':
                    for val in v:
                        add_location_data(k, val)
            print(data_dict)
        except Exception as e:
            pass

def map_view():
    global person_map
    global data_dict
    #マップの更新間隔(秒)
    RELOAD_INTERVAL = 0.2
    #表示するマップの時間範囲(指定時間から-TIME_RANGE~TIME_RANGE)(秒)
    TIME_RANGE = 0.5
    #マップの表示遅延時間(秒)
    TIME_DELAY = 0.5

    MOVE_X = 1.63
    MOVE_Z = 0.84
    MOVE_THETA = 0

    while True:
        #指定した範囲の時間の座標をすべて取り出す
        map_img = person_map.clear_map_img()
        for k, v in data_dict.items():
            for val in v:
                average_location = [val['x'], val['z']]
                temp_x = -MOVE_X+average_location[0]*math.cos(MOVE_THETA)-average_location[1]*math.sin(MOVE_THETA)
                average_location[1] = -MOVE_Z+average_location[0]*math.sin(MOVE_THETA)+average_location[1]*math.cos(MOVE_THETA)
                average_location[0] = temp_x
                x, y = person_map.get_map_location(average_location)
                map_img = person_map.plot(x, y, k, map_img)
                # if(x>=0 and y>=0 and x<person_map.width and y < person_map.height):
                #     file_exists = False
                #     file_path = 'obj_data.csv'
                #     try:
                #         with open(file_path, 'r'):
                #             file_exists = True
                #     except FileNotFoundError:
                #         pass
                #     with open(file_path, 'a', newline='') as file:
                #         writer = csv.writer(file)
                #         # ファイルが存在しなかった場合はヘッダーを書き込む
                #         if not file_exists:
                #             writer.writerow(['class', 'x', 'y'])  # ヘッダーの内容を適宜変更
                #         # データを書き込む
                #         writer.writerow([time_stamp, x, y, action, ",".join(map(str, interact_list)), ",".join(map(str, target_obj_list))])
        cv2.imshow('server_map', map_img) 
        cv2.waitKey(1)
        time.sleep(RELOAD_INTERVAL)

if __name__ == "__main__":
    main()

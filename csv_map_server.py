import socket;
import threading
import json
import time
import numpy as np
import cv2
from utils.point_handler import PointHandler, Cluster, PointMap

# 接続待ちするサーバのホスト名とポート番号を指定
HOST = "127.0.0.5"
PORT = 55580

#カメラ数
CAMERA_AMOUNT = 5

person_data = None
person_handler = PointHandler('person')
person_cluster = Cluster(eps=0.3, min_samples=5)
person_map = PointMap()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

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
            person_data = person_handler.put_json_data(json_dic=json_dic, data_list=person_data)
        except Exception as e:
            pass

def map_view():
    global person_handler
    global person_cluster
    global person_map
    global person_data
    #マップの更新間隔(秒)
    RELOAD_INTERVAL = 0.2
    #表示するマップの時間範囲(指定時間から-TIME_RANGE~TIME_RANGE)(秒)
    TIME_RANGE = 0.5
    #マップの表示遅延時間(秒)
    TIME_DELAY = 1.0
    while True:
        #指定した範囲の時間の座標をすべて取り出す
        point_list = person_handler.get_data(person_data, delay=TIME_DELAY, before_range=TIME_RANGE)
        map_img = person_map.clear_map_img()
        if point_list is not None:
            average_location = person_cluster.get_normal_average(point_list)
            interact_list, with_obj_list, action = person_handler.analize_data(point_list)
            person_handler.write_data_to_csv('server_data.csv', time.time()-TIME_DELAY, average_location, action, interact_list, with_obj_list)
            map_img = person_map.get_normal_map_img(average_location, map_img)
        cv2.imshow('server_map', map_img) 
        cv2.waitKey(1)
        time.sleep(RELOAD_INTERVAL)

if __name__ == "__main__":
    main()
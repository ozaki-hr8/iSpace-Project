import socket;
import threading
import json
import time
import numpy as np
import cv2
from point_handler import PointHandler, Cluster, PointMap

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
    RELOAD_INTERVAL = 1
    #表示するマップの時間範囲(指定時間から-TIME_RANGE~TIME_RANGE)(秒)
    TIME_RANGE = 1.0
    #マップの表示遅延時間(秒)
    TIME_DELAY = 1.0
    while True:
        #指定した範囲の時間の座標をすべて取り出す
        point_list = person_handler.get_data(person_data, delay=TIME_DELAY, before_range=TIME_RANGE)
        map_img = person_map.clear_map_img()
        if point_list is not None:
            cluster_list = person_cluster.get_cluster(point_list)
            average_list = person_cluster.get_average_list(point_list, cluster_list)
            map_img = person_map.get_map_img(point_list, cluster_list, average_list, map_img)
        cv2.imshow('server_map', map_img) 
        cv2.waitKey(1)
        time.sleep(RELOAD_INTERVAL)

if __name__ == "__main__":
    main()
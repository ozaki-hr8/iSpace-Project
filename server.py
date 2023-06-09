import socket;
import threading
import json
import time
import datetime
import numpy as np
import cv2
import math
from utils.point_handler import PointHandler, Cluster, PointMap
from utils.ip_handler import get_ip, get_port

# 接続待ちするサーバのホスト名とポート番号を指定
HOST = get_ip()
PORT = get_port()

#カメラ数
CAMERA_AMOUNT = 6

person_data = None
person_handler = PointHandler('person')
person_cluster = Cluster(eps=0.3, min_samples=5)
person_map = PointMap(map_img=cv2.imread("map.png"))

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
    TIME_DELAY = 0.5

    MOVE_X = 1.63
    MOVE_Z = 0.84
    MOVE_THETA = 0

    while True:
        #指定した範囲の時間の座標をすべて取り出す
        point_list = person_handler.get_data(person_data, delay=TIME_DELAY, before_range=TIME_RANGE)
        map_img = person_map.clear_map_img()
        if point_list is not None:
            # cluster処理（複数人対応）
            # cluster_list = person_cluster.get_cluster(point_list)
            # average_list, prod_list = person_cluster.get_average_and_prod_list(point_list, cluster_list)
            # map_img = person_map.get_map_img(point_list, cluster_list, average_list, prod_list, map_img)
            # 1人の検知のみ
            interact_list, target_obj_list, action = person_handler.multicam_complement(point_list)
            #average_location = person_cluster.get_normal_average(point_list)
            average_location = person_cluster.get_norm_average(point_list)
            temp_x = -MOVE_X+average_location[0]*math.cos(MOVE_THETA)-average_location[1]*math.sin(MOVE_THETA)
            average_location[1] = -MOVE_Z+average_location[0]*math.sin(MOVE_THETA)+average_location[1]*math.cos(MOVE_THETA)
            average_location[0] = temp_x
            c_list = ''
            for cid, oid in zip(interact_list, target_obj_list):
                if (cid !="none" and cid !="holding"):
                    c_list += f'{cid} with {oid} '
            x, y = person_map.get_map_location(average_location)
            map_img = person_map.get_normal_map_img(x, y, map_img, action, c_list)
            data_time = datetime.datetime.now()-datetime.timedelta(seconds=TIME_DELAY)
            cv2.putText(map_img, str(data_time),
                                   (10,10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            if(x>=0 and y>=0 and x<person_map.width and y < person_map.height):
                person_handler.write_data_to_csv('server_data.csv', data_time, x, y, action, interact_list, target_obj_list)
        cv2.imshow('server_map', map_img) 
        cv2.waitKey(1)
        time.sleep(RELOAD_INTERVAL)

if __name__ == "__main__":
    main()

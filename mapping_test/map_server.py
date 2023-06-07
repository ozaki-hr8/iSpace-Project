import socket;
import threading
import json
import time
import numpy as np
import cv2

# 接続待ちするサーバのホスト名とポート番号を指定
HOST = "127.0.0.6"
PORT = 55580
# マップにプロットする際のX,Y範囲(m)
X_RANGE = 2.0
Y_RANGE = 5.0
#マップ画像のサイズ (縦,横)
MAP_SIZE=(480,640)
#カメラ数
CAMERA_AMOUNT = 5

data_array = np.empty((0, 5), float)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def main():
    id = 0
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
        thread = threading.Thread(target=loop_handler, args=(conn, addr, id))
        thread.start()
        id += 1

def loop_handler(connection, address, id):
    while True:
        try:
            #クライアント側から受信
            res = connection.recv(4096)
            json_dic = json.loads(res.decode('utf-8'))
            #data_arrayに受信データを保存
            add_data(json_dic, id)
        except Exception as e:
            print(e)
            break

#data_arrayに収まるようにJSONデータを加工し、保存する
def add_data(json_dic, id):
    global data_array
    person_loc = get_data(json_dic, 'person')
    time_stamp = get_data(json_dic, 'timestamp')
    if person_loc is not None:
        for loc in person_loc:
            x,z,weight = loc.split(',')
            new_data = np.array([[time_stamp, x, z, weight, id]], float)
            data_array = np.concatenate((data_array, new_data), axis=0)

def get_data(json_dic, data_name):
    if data_name in json_dic:
        return json_dic[data_name]
    return None

def map_view():
    #マップの更新間隔(秒)
    RELOAD_INTERVAL = 0.2
    #表示するマップの時間範囲(指定時間から-TIME_RANGE~TIME_RANGE)(秒)
    TIME_RANGE = 0.1
    #マップの表示遅延時間(秒)
    TIME_DELAY = 1.0
    while True:
        #指定した範囲の時間の座標をすべて取り出す
        points_array = get_points_array(TIME_RANGE, TIME_DELAY)
        #白画像作成
        map_img = cv2.cvtColor((np.zeros(MAP_SIZE, np.uint8)+255), cv2.COLOR_GRAY2BGR)
        #マップに点を打つ
        for data in points_array:
            y = int((data[2]/Y_RANGE)*MAP_SIZE[1])
            x = int(((data[1]/X_RANGE)+1)*(MAP_SIZE[0]/2))
            weight = 255-int(255*data[3])
            if data[4] == 0:
                cv2.circle(map_img, (y,x), 3, (weight, weight, 255), thickness=-1)
            elif data[4] == 1:
                cv2.circle(map_img, (y,x), 3, (weight, 255, weight), thickness=-1)
            else:
                cv2.circle(map_img, (y,x), 3, (255, weight, weight), thickness=-1)
        cv2.imshow('server_map', map_img) 
        cv2.waitKey(1)
        time.sleep(RELOAD_INTERVAL)

#指定した範囲の時間のデータのみを配列として返す
def get_points_array(time_range, time_delay):
    delay_time = time.time() - time_delay
    return data_array[abs(delay_time - data_array[:, 0]) <= time_range]

if __name__ == "__main__":
    main()
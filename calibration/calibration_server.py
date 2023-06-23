#使用方法
#1. prediction_server.pyを起動
#2. メインカメラコンピュータでprediction_client.pyを起動
#3. 推定したいカメラコンピュータでprediction_client.pyを起動
#4. それぞれのコンピュータで交互に同じ物体をクリックする
#5. 出力されたX,Z,THETAをmap_client.pyのプログラムに書き込む

import socket;
import threading
import json
import math
import numpy as np

# 接続待ちするサーバのホスト名とポート番号を指定
HOST = "172.31.178.47"
PORT = 55580
# マップにプロットする際のX,Y範囲(m)
X_RANGE = 2.0
Y_RANGE = 5.0
#マップ画像のサイズ (縦,横)
MAP_SIZE=(480,640)
DATA_AMOUNT = 5

#ベースとなるカメラのIP
source_ip = None
#位置推定を行いたいカメラのIP
target_ip = None

source_data = np.empty((0, 2), float)
target_data = np.empty((0, 2), float)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def main():
    global source_ip
    global target_ip
    sock.bind((HOST, PORT))
    sock.listen(2)

    while True:
        try:
            conn, addr = sock.accept()
        except KeyboardInterrupt:
            sock.close()
            exit()
            break
        # アドレス確認
        if source_ip is None:
            source_ip = addr[0]
        elif target_ip is None:
            target_ip = addr[0]
        print("[アクセスを確認] => {}:{}".format(addr[0], addr[1]))
        # スレッド作成
        thread = threading.Thread(target=loop_handler, args=(conn, addr))
        thread.start()

def loop_handler(connection, address):
    while True:
        try:
            #クライアント側から受信
            res = connection.recv(4096)
            json_dic = json.loads(res.decode('utf-8'))
        except Exception as e:
            print(e)
            break
        #data_arrayに受信データを保存
        add_data(json_dic, address)

#data_arrayに収まるようにJSONデータを加工し、保存する
def add_data(json_dic, address):
    global target_data
    global source_data
    person_loc = get_data(json_dic, 'person')
    if person_loc is not None:
        for loc in person_loc:
            x,z,w = loc.split(',')
            new_data = np.array([[x,z]], float)
            if target_ip == address[0]:
                target_data = np.concatenate((target_data, new_data), axis=0)
            else:
                source_data = np.concatenate((source_data, new_data), axis=0)
            print(target_data, source_data)
            if target_data.shape[0] >= DATA_AMOUNT and source_data.shape[0] >= DATA_AMOUNT:
                theta = calc_theta()    #メインカメラに対するターゲットカメラの角度を推定
                dx,dz = calc_xz(theta)       #メインカメラからのターゲットカメラの相対座標を推定
                print('THETA = '+str(theta))
                print('X = '+str(dx))
                print('Z = '+str(dz))
                source_data = np.empty((0, 2), float)
                target_data = np.empty((0, 2), float)

#得た点の座標からカメラの角度を推定する
def calc_theta():
    theta_array = []
    target_diff = np.diff(target_data, axis=0)
    source_diff = np.diff(source_data, axis=0)
    for i in range(DATA_AMOUNT-1):
      #2点を結ぶ線分の角度をそれぞれ求める
      theta_t = math.atan2(target_diff[i][1], target_diff[i][0])
      theta_s = math.atan2(source_diff[i][1], source_diff[i][0])

      theta_diff = ((theta_s - theta_t)+2*math.pi) % (2*math.pi)
      
      theta_array.append(theta_diff) #角度の差をリストに保存

    theta_array = np.array(theta_array)
    print(theta_array)

    return np.mean(theta_array)     #角度の差の平均を返す

#推定したカメラ角度を用いて、カメラ座標を推定する
def calc_xz(theta):
    diff_array = []
    for i in range(DATA_AMOUNT):
        #メインカメラと同じ角度になるよう、座標を回転させる。
        x = target_data[i][0]*math.cos(theta)-math.sin(theta)*target_data[i][1]
        z = target_data[i][0]*math.sin(theta)+math.cos(theta)*target_data[i][1]
        #回転後の座標の差を求め、リストに保存
        diff = np.array([x,z])-source_data[i]
        diff_array.append(diff)
    return np.mean(diff_array, axis=0)  #平均座標を返す

def get_data(json_dic, data_name):
    if data_name in json_dic:
        return json_dic[data_name]
    return None

if __name__ == "__main__":
    main()


#使用方法
#1. prediction_server.pyを起動
#2. メインカメラコンピュータでprediction_client.pyを起動
#3. 推定したいカメラコンピュータでprediction_client.pyを起動
#4. それぞれのコンピュータで交互に同じ物体をクリックする
#5. 出力されたX,Z,THETAをmap_client.pyのプログラムに書き込む

import socket
import threading
import math
import cv2
import numpy as np
import time
from calibration.calib_ip_handler import get_ip, get_port

# 接続待ちするサーバのホスト名とポート番号を指定
HOST = get_ip()
PORT = get_port()
# マップにプロットする際のX,Y範囲(m)
X_RANGE = 5.0
Y_RANGE = 10.0
#マップ画像のサイズ (縦,横)
MAP_SIZE=(640,640)

client_list = []

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def main():
    state = 0
    sock.bind((HOST, PORT))
    sock.listen(10)

    while True:
        if state == 0:
            display = cv2.cvtColor((np.zeros((640, 640), np.uint8)+255), cv2.COLOR_GRAY2BGR)
            cv2.putText(display, f'Connect: {len(client_list)}',
                                    (220, 320),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA
                                            )
            key = cv2.waitKey(1)
            cv2.imshow('server', display)
            if key == ord('k'):
                state = 1
                display = cv2.cvtColor((np.zeros((640, 640), np.uint8)+255), cv2.COLOR_GRAY2BGR)
                cv2.putText(display, '5 seconds',
                                    (220, 320),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA
                                            )
                cv2.imshow('server', display)
                time.sleep(5)
            try:
                conn, addr = sock.accept()
            except KeyboardInterrupt:
                sock.close()
                exit()
                break
            # アドレス確認
            print("[アクセスを確認] => {}:{}".format(addr[0], addr[1]))
            client = Client(conn, addr)
            client_list.append(client)
        elif state == 1:
            display = cv2.cvtColor((np.zeros((640, 640), np.uint8)+255), cv2.COLOR_GRAY2BGR)
            cv2.putText(display, f'START',
                                    (220, 320),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA
                                            )
            cv2.waitKey(1)
            cv2.imshow('server', display)

if __name__ == "__main__":
    main()

class Client:
    def __init__(self, connection, address):
        self.connection = connection
        self.address = address
        self.data_list = []
        thread = threading.Thread(target=self.loop, args=self)
        thread.start()
    
    def loop(self):
        while(True):
            try:
                #クライアント側から受信
                res = self.connection.recv(4096)
            except Exception as e:
                print(e)
                break
            try:
                location_data = res.decode('utf-8')
                if location_data:
                    if ',' in location_data:
                        self.data_list.append(list(map(float, location_data.split(','))))
                    else:
                        self.data_list.append('None')
            except Exception as e:        
                pass
    
    def send_signal(self):
        self.connection.send('START'.encode(encoding='utf-8'))

    def get_matrix(self, world_coods):
        X = np.empty((0,2),float)
        W = np.array([])
        for i in range(len(self.data_list)):
            if self.data_list[i] == 'None' or world_coods[i] == 'None':
                continue
            X = np.append(X, np.array([[self.data_list[i][0], self.data_list[i][1]]]), axis=0)
            W = np.append(W, np.array([world_coods[i][0], world_coods[i][1]]))
        XX = []
        for i in range(len(X)):
            XX.append([X[i][0], X[i][1], 1, 0, 0, 0])
            XX.append([0, 0, 0, X[i][0], X[i][1], 1])
        X = np.array(XX)
        matrix = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(W)
        return matrix

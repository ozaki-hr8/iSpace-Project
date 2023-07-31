#使用方法
#1. prediction_server.pyを起動
#2. メインカメラコンピュータでprediction_client.pyを起動
#3. 推定したいカメラコンピュータでprediction_client.pyを起動
#4. それぞれのコンピュータで交互に同じ物体をクリックする
#5. 出力されたX,Z,THETAをmap_client.pyのプログラムに書き込む

import socket
import threading
import math
import json
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
state = 0

END_COUNT = 300

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def loop_handler():
    while True:
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

def main():
    global state
    # sock.bind((HOST, PORT))
    # sock.listen(10)
    # thread = threading.Thread(target=loop_handler)
    # thread.start()
    count = 0

    while True:
        display = cv2.cvtColor((np.zeros((640, 640), np.uint8)+255), cv2.COLOR_GRAY2BGR)
        if state == 0:
            cv2.putText(display, f'Connect: {len(client_list)}',
                                    (220, 320),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA
                                            )
        elif state == 1:
            cv2.putText(display, '5 seconds',
                                    (220, 320),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA
                                            )
            cv2.imshow('server', display)
            state = 2
        elif state == 2:
            cv2.putText(display, 'START',
                                    (220, 320),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA
                                            )
            time.sleep(5)
            state = 3
        elif state == 3:
            for client in client_list:
                client.send_signal()
            count += 1
            time.sleep(0.5)
            cv2.putText(display, f'count: {count}',
                                    (220, 320),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA
                                            )
            if count >= END_COUNT:
                state = 4
        elif state == 4:
            time.sleep(1)
            base_client = client_list[0]
            for i in range(len(client_list-1)):
                clt = client_list[i+1]
                dx, dz, dt = clt.get_matrix(base_client.data_list)
                diff_x, diff_z, diff_theta = clt.get_diff(dx, dz, dt)
                bdiff_x, bdiff_z, bdiff_theta = clt.get_diff(clt.bx, clt.bz, clt.bt)
                print(f'result: {diff_x},{diff_z},{diff_theta}')
                print(f'before_diff: {bdiff_x},{bdiff_z},{bdiff_theta}')
                print(f'current_diff: {diff_x},{diff_z},{diff_theta}')
            state = 5
        elif state == 5:
            sock.close()
            cv2.destroyAllWindows()
            break


        key = cv2.waitKey(1) & 0xFF
        cv2.imshow('server', display)

        if key == ord('k') and state == 0:
            state = 1
        elif key == 27:
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()

class Client:
    def __init__(self, connection, address):
        self.connection = connection
        self.address = address
        self.data_list = []
        thread = threading.Thread(target=self.loop, args=self)
        thread.start()
        self.connection.send('B'.encode(encoding='utf-8'))
    
    def loop(self):
        while(True):
            try:
                #クライアント側から受信
                res = self.connection.recv(4096)
            except Exception as e:
                print(e)
                break
            try:
                json_data = json.loads(res.decode('utf-8'))
                if json_data:
                    if 'bx' in json_data and 'bz' in json_data and 'bt' in json_data:
                        self.bx = json_data['bx']
                        self.bz = json_data['bz']
                        self.bt = json_data['bt']
                    elif 'cx' in json_data and 'cz' in json_data:
                        cx = json_data['cx']
                        cz = json_data['cz']
                        if cz == 0:
                            self.data_list.append('NoData')
                        else:
                            self.data_list.append([cx, cz])
            except Exception as e:        
                pass
    
    def send_signal(self):
        self.connection.send('C'.encode(encoding='utf-8'))

    def n(self, theta):
        return (theta+2*math.pi) % (2*math.pi)

    def get_matrix(self, world_coods):
        self.X = np.empty((0,2),float)
        self.W = np.array([])
        for i in range(len(self.data_list)):
            if self.data_list[i] == 'NoData' or world_coods[i] == 'NoData':
                continue
            self.X = np.append(self.X, np.array([[self.data_list[i][0], self.data_list[i][1]]]), axis=0)
            self.W = np.append(self.W, np.array([world_coods[i][0], world_coods[i][1]]))
        self.XX = []
        for i in range(len(self.X)):
            self.XX.append([self.X[i][0], self.X[i][1], 1, 0, 0, 0])
            self.XX.append([0, 0, 0, self.X[i][0], self.X[i][1], 1])
        self.X = np.array(self.XX)
        matrix = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.W)
        dx = matrix[2]
        dz = matrix[5]
        dt = (self.n(math.acos(matrix[0]))+self.n(math.asin(-matrix[1]))+self.n(math.asin(matrix[3]))+self.n(math.acos(matrix[4])))/4
        return dx, dz, dt
    
    def get_diff(self, dx, dz, dt):
        diff_x = 0
        diff_z = 0
        diff_dist = 0
        size = self.X.shape[0]
        for i in range(self.X.shape[0]):
            px, pz = self.X[i]
            temp_x = -dx+px*math.cos(dt)-pz*math.sin(dt)
            pz = -dz+px*math.sin(dt)+pz*math.cos(dt)
            px = temp_x
            wx, wz = self.W[i]
            diff_x += abs(wx-px)
            diff_z += abs(wz -pz)
            diff_dist += abs(math.sqrt((wx-px)**2+(wz-pz)**2))
        return diff_x/size, diff_z/size, diff_dist/size


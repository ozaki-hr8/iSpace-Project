import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import json
import threading

from calib_ip_handler import get_ip, get_port

#カメラ位置推定サーバーのIP,ポート
SERVER_IP = get_ip()
SERVER_PORT = get_port()

#realsense設定
WIDTH = 640
HEIGHT = 480
FPS = 30

#クリックした座標の平均を求めるのに使用するデータ数
AVERAGE_AMOUNT = 50

click_x = -1
click_y = -1
data_array = np.empty((0, 3), float)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def main():
    #マップ生成サーバーへ接続
    sock.connect((SERVER_IP, SERVER_PORT))
    #RealSenseの初期設定を行う
    align, config, pipeline, profile = realsense_setting()
    #RealSenseの内部パラメータを取得
    color_intr = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()
    #距離情報にかけるフィルターを取得する
    decimate, spatial, depth_to_disparity, disparity_to_depth = get_filters()

    try:
        while True:
            # フレーム取得
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue
            # RGB画像
            color_image = np.asanyarray(color_frame.get_data())

            #距離情報にフィルターをかける
            filter_frame = decimate.process(depth_frame)
            filter_frame = depth_to_disparity.process(filter_frame)
            filter_frame = spatial.process(filter_frame)
            filter_frame = disparity_to_depth.process(filter_frame)
            depth_frame = filter_frame.as_depth_frame()

            #ピクセル座標をカメラを中心とした3D座標に変換する
            clicked_location = get_average_position(color_intr, depth_frame)
            if clicked_location is not None:
                #マップ生成サーバーに座標を送信
                send_location(clicked_location)
                map_view(clicked_location)

            cv2.imshow('RealSense', color_image)
            cv2.setMouseCallback('RealSense', click_event)

            if cv2.waitKey(1) & 0xff == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

#realsenseの初期設定を行う
def realsense_setting():
    align = rs.align(rs.stream.color)
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    return (align, config, pipeline, profile)

#pixel_x, pixel_yの地点のカメラを起点とした三次元座標を取得する
def get_3d_location(color_intr, depth_frame, pixel_x, pixel_y):
    distance = depth_frame.get_distance(pixel_x,pixel_y)
    if distance == 0:
        return None
    return rs.rs2_deproject_pixel_to_point(color_intr , [pixel_x,pixel_y], distance)

#距離情報にかけるフィルターを設定、取得する
def get_filters():
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 1)
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 1)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    # disparity
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    return (decimate, spatial, depth_to_disparity, disparity_to_depth)

#三次元の座標データを加工し、マップ作成サーバーへ送信する
def send_location(location_3d):
    global sock
    data = {
               'person' : ['{},{},1.0'.format(location_3d[0], location_3d[2])],
            }
    print(location_3d)
    sock.send(json.dumps(data).encode("UTF-8"))

def map_view(location_3d):
    X_RANGE = 2.0
    Y_RANGE = 5.0
    MAP_SIZE = (480,640)
    white_img=np.zeros(MAP_SIZE,np.uint8)
    white_img = white_img + 255
    y = int((location_3d[2]/Y_RANGE)*MAP_SIZE[1])
    x = int(((location_3d[0]/X_RANGE)+1)*(MAP_SIZE[0]/2))
    cv2.circle(white_img, (y,x), 3, (0, 0, 255), thickness=-1)
    cv2.imshow('map', white_img)

def click_event(event, x, y, flags, params):
    global click_x
    global click_y
    global data_array
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x = x
        click_y = y
        data_array = np.empty((0, 3), float)

def get_average_position(color_intr, depth_frame):
    global click_x
    global click_y
    global data_array
    if click_x == -1 or click_y == -1:
        return None
    if data_array.shape[0] < AVERAGE_AMOUNT:
        location = get_3d_location(color_intr, depth_frame, click_x, click_y)
        print(location)
        if location is not None:
            data_array = np.concatenate((data_array, np.array([location])), axis=0)
            print(data_array)
    elif data_array.shape[0] == AVERAGE_AMOUNT:
        average = calc_average()
        click_x = -1
        return average
    return None

#標準偏差を用いて外れ値を排除した座標平均を求める
def calc_average():
    global data_array
    THRESHOLD = 3  # 外れ値の閾値（例えば、3倍の標準偏差）

    std_dev = np.std(data_array, axis=0)  # 列ごとの標準偏差を計算
    mask = np.abs(data_array - np.mean(data_array, axis=0)) < THRESHOLD * std_dev
    # 外れ値を除外したデータを作成
    filtered_data = data_array[np.all(mask, axis=1)]
    # 平均を計算
    average = np.mean(filtered_data, axis=0)
    print(average)
    
    return average

if __name__ == "__main__":
    main()

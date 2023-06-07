#定数を入力した上でRealSenseと接続したNUC上で実行

import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import time
import json
import math

#座標変換用定数。predict_camera_locationで取得する(メインカメラについては、X,Y,THETA=0)
X = 3.682768768475109
Z = -2.6057855270816472
THETA = 4.749273148314835

#マップ生成サーバーのIP,ポート
SERVER_IP = "172.31.177.251"
SERVER_PORT = 55580

#realsense設定
WIDTH = 640
HEIGHT = 480
FPS = 30

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

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
            pixel_loc = trace_pixel(color_image)
            if pixel_loc is not None:
                cv2.circle(color_image, pixel_loc, 3, (0, 0, 255), thickness=-1)
                location_3d = get_3d_location(color_intr, depth_frame, pixel_loc[0], pixel_loc[1])
                if location_3d is not None:
                    #マップ生成サーバーに座標を送信
                    send_location(location_3d)
                    map_view(location_3d)
                    print(location_3d)

            cv2.imshow('RealSense', color_image)

            #距離画像のカラーマップを作成(単なる視覚化)
            #depth_color_frame = rs.colorizer().colorize(depth_frame)
            #depth_image = np.asanyarray(depth_color_frame.get_data())
            #cv2.imshow('depth_image', depth_image)

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

#オレンジの点の中央を座標として取得
def trace_pixel(color_img):
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([0, 150, 10])
    upper_color = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    output = cv2.medianBlur(mask, ksize=23)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(output)
    if retval != 0:
        for coordinate in stats[1:]:
            if coordinate[4] > 200:
                return (int((coordinate[0]*2+coordinate[2])/2), int((coordinate[1]*2+coordinate[3])/2))
    return None

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
    x,z = calc_location(location_3d)
    print(x,z)
    data = {
               'person' : ['{},{},1.0'.format(x, z)],
               'timestamp' : time.time()
            }
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

def calc_location(location_3d):
    if X == 0 and Z == 0:
        return location_3d[0],location_3d[2]
    x = -X+location_3d[0]*math.cos(THETA)-location_3d[2]*math.sin(THETA)
    z = -Z+location_3d[0]*math.sin(THETA)+location_3d[2]*math.cos(THETA)
    return (x,z)
    
if __name__ == "__main__":
    main()

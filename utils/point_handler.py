import numpy as np
import time

id_dict = {
    0 : ['none', 'none'],
    1 : ['holding', 'bottle'],
    2 : ['drinking', 'bottle'],
    3 : ['holding', 'phone'],
    4 : ['calling', 'phone'],
    5 : ['holding', 'book'],
    6 : ['reading', 'book'],
    7 : ['working', 'computer'],
    8 : ['holding', 'cushion'],
    9 : ['holding', 'food'],
    10 : ['eating', 'food'],

    100 : ['standing', 'none'],
    101 : ['sitting', 'none'],
    102 : ['laying', 'none'],
    103 : ['walking', 'none']
}

name_dict = {
    'None' : 0,
    'Holding Bottle' : 1,
    'Drinking' : 2,
    'Holding Phone' : 3,
    'Calling on Phone' : 4,
    'Holding Book' : 5,
    'Reading Book' : 6,
    'Working on Computer' : 7,
    'Holding Cushion' : 8,
    'Holding Food' : 9,
    'Eating' : 10,

    'Standing' : 100,
    'Sitting' : 101,
    'Laying' : 102,
    'Walking' : 103
}

def id_to_name(id):
  global id_dict
  if id in id_dict:
    return id_dict[id]
  return None

def name_to_id(name):
  global name_dict
  if name in name_dict:
    return name_dict[name]
  return None

def get_interact_weight(distance):
    if distance < 1:
        return 0.92
    elif distance < 2:
        return 0.95
    elif distance < 3:
        return 0.92
    elif distance < 4:
        return 0.81
    elif distance < 5:
        return 0.55
    elif distance < 6:
        return 0.61
    else:
        return 0.3

def get_interact_weight_for_list(distance_list):
    for i in range(distance_list.shape[0]):
        distance_list[i] = get_interact_weight(distance_list[i])
    return distance_list


def get_location_weight(distance):
    rms = -2.5E-4 + 1.1904762E-4 * distance + 0.003761904762*(distance^2)
    return 1 - rms/distance

import pandas as pd
import csv

#クライアント側でポイントデータを扱う際に使用
class PointHandler:
    def __init__(self, class_name, none_threshold=0.9):
        self.class_name = class_name
        self.threshold = none_threshold
        self.file_exists = False

    def put_json_data(self, json_dic, data_list=None):
        if not isinstance(json_dic, dict):
            return data_list
        if json_dic == {}:
            return data_list
        if 'timestamp' in json_dic:
            time_stamp = json_dic['timestamp']
            if self.class_name in json_dic:
                for target in json_dic[self.class_name]:
                    if not isinstance(target, dict):
                        return None
                    if not all(key in target for key in ('x', 'z', 'd', 'probs')):
                        return None
                    point_data = [time_stamp, target['x'], target['z'], target['d']]
                    probs = target['probs']
                    flattened_list = [item for sublist in probs for item in sublist]
                    point_data = np.array([[time_stamp, target['x'], target['z'], target['d']]+flattened_list], dtype=float)
                    if data_list is None:
                        data_list = point_data
                    else:
                        data_list = np.concatenate((data_list, point_data), axis=0)
        return data_list

    def get_data(self, data_list, delay=1.0, before_range=5.0):
        if data_list is None:
            return None
        current_time = time.time() - delay
        indices = np.where((current_time - data_list[:,0] > 0) & (current_time - data_list[:,0] < before_range))
        if indices[0].shape[0] == 0:
            return None
        point_list = data_list[indices]
        return point_list

    def analize_data(self, data_list):
        interact_list = []
        with_obj_list = []
        action = None
        data_amount = max(0, int((data_list[0].shape[0]-4)/2))
        for i in range(data_amount):
            #しきい値の割合(0.9)以上がNoneだった場合、リストに追加しない(None処理)
            if np.where(data_list[:,4+i*2] == 0)[0].shape[0]/data_list.shape[0] > self.threshold:
                continue
            column_list = []
            column_total = 0
            #列にあるidをすべて取得
            unique_ids = np.unique(data_list[:,4+i*2])
            for id in unique_ids:
                if id == 0:
                    continue
                id_data = data_list[data_list[:,4+i*2] == id]
                id_total = np.sum(id_data[:, 5+i*2]*get_interact_weight_for_list(id_data[:,3]))
                column_total += id_total
                column_list.append([id, id_total])
            max_id = 0
            max_prob = 0
            for data in column_list:
                prob = data[1]/id_total
                if max_prob < prob:
                    max_prob = prob
                    max_id = data[0]
            class_data = id_to_name(max_id)
            #インタラクションデータの場合は分けて格納
            if class_data is None:
                if data_amount-1 == i:
                    action = 'NO CLASS'
                    continue
                interact_list.append('NO CLASS')
                with_obj_list.append('NO OBJ')
            else:
                if data_amount-1 == i:
                    action = class_data[0]
                    continue
                interact_list.append(class_data[0])
                with_obj_list.append(class_data[1])

        if len(interact_list) == 0:
            interact_list.append('none')
        if len(with_obj_list) == 0:
            with_obj_list.append('none')
        return interact_list, with_obj_list, action


    # def to_csv(self, path, time_stamp, average_loc, action, interact_list, with_obj_list):

    #     df = pd.DataFrame({
    #         'time' : time_stamp,
    #         'x' : average_loc[0],
    #         'z' : average_loc[1],
    #         'action' : action,
    #         'interact' : ",".join(map(str, interact_list)),
    #         'with_obj' : ",".join(map(str, with_obj_list))
    #     })
    #     df.set_index('time')
    #     df.to_csv(path, encoding='shift_jis')

    def write_data_to_csv(self, file_path, time_stamp, average_loc, action, interact_list, with_obj_list):
        # CSVファイルが存在しない場合に新しいファイルを作成する
        if not self.file_exists:
            try:
                with open(file_path, 'r'):
                    self.file_exists = True
            except FileNotFoundError:
                pass
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            # ファイルが存在しなかった場合はヘッダーを書き込む
            if not self.file_exists:
                writer.writerow(['time', 'x', 'z', 'action', 'interactions', 'target_obj'])  # ヘッダーの内容を適宜変更
            # データを書き込む
            writer.writerow([time_stamp, average_loc[0], average_loc[1], action, ",".join(map(str, interact_list)), ",".join(map(str, with_obj_list))])

import math

class Point:
    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.distance = z
        self.prod_list = []

    def add_probability(self, prob_name, probability):
        id = name_to_id(prob_name)
        if(id is None):
            return
        self.prod_list.append([id, probability])

    def get_json(self):
        return {
            'x' : self.x,
            'z' : self.z,
            'd' : self.distance,
            'probs' : self.prod_list
        }

    def convert_location(self, X=0, Z=0, THETA=0, PITCH=0):
        self.z *= math.cos(PITCH)
        if X == 0 and Z == 0:
            return
        temp_x = -X+self.x*math.cos(THETA)-self.z*math.sin(THETA)
        self.z = -Z+self.x*math.sin(THETA)+self.z*math.cos(THETA)
        self.x = temp_x

from sklearn.cluster import DBSCAN

class Cluster:
    def __init__(self, eps=0.3, min_samples=5):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    def get_cluster(self, point_list):
        if point_list is None:
            return
        if point_list.shape[0] == 0:
            return
        return self.dbscan.fit_predict(point_list[:,1:3])

    def get_average_and_prod_list(self, point_list, cluster_list):
        if point_list is None or cluster_list is None:
            return None
        average_list = []
        prod_list = []
        for i in range(-1, max(cluster_list)+1):
            cluster = point_list[cluster_list==i]
            if cluster.shape[0] > 0:
                x_mean = np.mean(cluster[:,1])
                z_mean = np.mean(cluster[:,2])
                average_list.append([x_mean,z_mean])
                print(cluster[0].shape[0])
                for i in range(max(0, int((cluster[0].shape[0]-4)/2))):
                    column_list = []
                    unique_ids = np.unique(cluster[:,4+i*2])
                    for id in unique_ids:
                        id_data = cluster[cluster[:,4+i*2] == id]
                        average = np.mean(id_data[:, 5+i*2]*get_interact_weight_for_list(id_data[:,3]))
                        column_list.append([id, average])
                prod_list.append(column_list)
        return average_list, prod_list

    def get_normal_average(self, point_list):
        if point_list is None:
            return None
        return np.mean(point_list[:,1:3], axis=0)
    
    def get_norm_average(self, point_list):
        if point_list is None:
            return None
        distance_list = 10 - point_list[:,3]
        distance_list[np.where(distance_list < 0)] = 0
        norm = distance_list / np.sum(distance_list)
        x = np.sum(point_list[:,1]*norm)
        z = np.sum(point_list[:,2]*norm)
        return np.array([x,z])

import cv2

class PointMap:
    def __init__(self, width=640, height=480, x_range=5.78, z_range=4.875, map_img=None):
        self.width = width
        self.height = height
        self.x_range = x_range
        self.z_range = z_range
        self.map_img = map_img
        if map_img is not None:
            self.width = map_img.shape[1]
            self.height = map_img.shape[0]

    def get_map_img(self, point_list, cluster_list, average_list, prob_list, map_img):
        PALE_COLORs = [(255,255,128),(255,128,255),(128,255,255),(255,192,192),(192,255,192),(192,192,255)]
        DARK_COLORs = [(255,0,0), (0,255,0), (0,0,255), (128,128,0), (128,0,128), (0,128,128)]

        if point_list.shape[0] == 0 or cluster_list is None or average_list is None:
            return map_img

        for point, cluster_id in zip(point_list, cluster_list):
            color = (0,0,0)
            if cluster_id != -1:
                color = PALE_COLORs[min(cluster_id,5)]
            center = (int((point[2]/self.z_range)*self.width),int(((point[1]/self.x_range)+1)*(self.height/2)))
            cv2.circle(map_img, center, 3, color, thickness=-1)
        avg_id = 0
        for point in average_list:
            color = DARK_COLORs[min(avg_id,5)]
            center = (int((point[1]/self.z_range)*self.width),int(((point[0]/self.x_range)+1)*(self.height/2)))
            cv2.circle(map_img, center, 3, color, thickness=-1)
            cv2.putText(map_img, str(prob_list[avg_id]), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            avg_id += 1
        return map_img

    def get_normal_map_img(self, average_loc, map_img, action_class, interaction_class):
        if average_loc is None:
            return map_img
        #   center = (self.width-int(((average_loc[0]/self.x_range)+1)*(self.width/2)),int((average_loc[1]/self.z_range)*self.height))
        y = int((average_loc[1]/self.z_range)*self.height)
        x = -int((average_loc[0]/self.x_range)*self.width)
        center = (x, y)
        print(center)
        cv2.circle(map_img, center, 3, (0,0,255), thickness=-1)
        cv2.putText(map_img, str(action_class),
                                   (center[0]+30,center[1]+10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
                                        )
        cv2.putText(map_img, str(interaction_class),
                                   (center[0]+30,center[1]-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
                                        )
        return map_img

    def clear_map_img(self):
        if self.map_img is not None:
            return self.map_img.copy()
        return cv2.cvtColor((np.zeros((self.height, self.width), np.uint8)+255), cv2.COLOR_GRAY2BGR)
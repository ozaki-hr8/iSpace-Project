import numpy as np
import time

id_dict = {
    0 : 'None',
    1 : 'Holding Bottle',
    2 : 'Drinking Bottle'
}

name_dict = {value: key for key, value in id_dict.items()}

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

#クライアント側でポイントデータを扱う際に使用
class PointHandler:
    def __init__(self, class_name):
        self.class_name = class_name
        
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
        self.x = -X+self.x*math.cos(THETA)-self.z*math.sin(THETA)
        self.z = -Z+self.x*math.sin(THETA)+self.z*math.cos(THETA)

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
                        average = np.mean(id_data[:, 4+i*2]*get_interact_weight_for_list(id_data[:,3]))
                        column_list.append([id, average])
                prod_list.append(column_list)
        return average_list, prod_list
    
import cv2

class PointMap:
    def __init__(self, width=640, height=480, x_range=5.0, z_range=10.0):
        self.width = width
        self.height = height
        self.x_range = x_range
        self.z_range = z_range

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
    
    def clear_map_img(self):
        return cv2.cvtColor((np.zeros((self.height, self.width), np.uint8)+255), cv2.COLOR_GRAY2BGR)

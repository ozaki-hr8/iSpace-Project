import csv
import os

interact_data_list = ['client_data/csv_data/book_3d.csv',
                      'client_data/csv_data/bottle_3d.csv',
                      'client_data/csv_data/cellphone_3d.csv',
                      'client_data/csv_data/cushion_3d.csv',
                      'client_data/csv_data/food_3d.csv',
                      'client_data/csv_data/keyboard_3d.csv']

action_csv = 'client_data/csv_data/action_3d.csv'

if not os.path.exists('client_data'):
    os.makedirs('client_data')
if not os.path.exists('client_data/csv_data'):
    os.makedirs('client_data/csv_data')
if not os.path.exists('client_data/raw_img'):
    os.makedirs('client_data/raw_img')
if not os.path.exists('client_data/result_img'):
    os.makedirs('client_data/result_img')


num_coords = 33

landmarks = ['data_id', 'probs', 'class']
for val in range(1, num_coords+1):
    landmarks += ['skeleton_x{}'.format(val), 'skeleton_y{}'.format(val), 'skeleton_z{}'.format(val), 'skeleton_v{}'.format(val)]
            
landmarks+=['human_x','human_y','human_z']
landmarks+=['obj_x','obj_y','obj_z']
landmarks+=['obj_w','obj_h']
            
for val in range(1, num_coords+1):
    landmarks += ['distance_so{}'.format(val)]

for val in range(1, num_coords+1):
    landmarks += ['distance_s_x{}'.format(val), 'distance_s_y{}'.format(val), 'distance_s_z{}'.format(val), 'distance_s_v{}'.format(val)]
for csv_dir in interact_data_list:
    with open(csv_dir, mode='w', newline='') as f:
        csv_writer =csv.writer(f, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)
            
landmarks = ['data_id', 'probs','class']
for val in range(1, num_coords+1):
    landmarks += ['skeleton_x{}'.format(val), 'skeleton_y{}'.format(val), 'skeleton_z{}'.format(val), 'skeleton_v{}'.format(val)]

landmarks+=['human_x','human_y','human_z']

for val in range(1, num_coords+1):
    landmarks += ['distance_s_x{}'.format(val), 'distance_s_y{}'.format(val), 'distance_s_z{}'.format(val), 'distance_s_v{}'.format(val)]

with open(action_csv, mode='w', newline='') as f:
    csv_writer =csv.writer(f, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

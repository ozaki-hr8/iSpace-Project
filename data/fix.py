import os
import cv2

dir_path = 'ispace/valid/images' # 対象のディレクトリのパス

for filename in os.listdir(dir_path):
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
        file_path = os.path.join(dir_path, filename)
        try:
            with open(file_path, 'rb') as f:
                f.seek(-2, os.SEEK_END)
                if f.read() == b'\xff\xd9':
                    print(f"{filename}: OK")
                else:
                    # 画像が破損している場合は修正
                    img = cv2.imread(file_path)
                    cv2.imwrite(file_path, img)
                    print(f"{filename}: FIXED")
        except Exception as e:
            print(f"{filename}: ERROR - {e}")

# mpYolo
Method of using mediapipe in conjunction with yolov5 on RealSense D435 / Some application for intelligent Space

【ソースファイル詳細】

**物体認識結果の検索、通知**

detect_rst.py　search.py （index.html)

**骨格情報と物体認識の紐付け**

detect_mp.py

**mpの認識情報の統合及びインスタンス化**

detect_itg.py：Personクラスの属性値として，hand_gesture(手の上げ下げ)，expression(happy/sad),pointing(指差し対象)を指定，インスタンス化





https://user-images.githubusercontent.com/69960076/180616481-a7dc6d07-5943-4c10-bb2b-1c23805af885.mov



**深度情報と物体認識の紐付け**

detect_rs.py dataset_rs.py

**3次元でのインタラクション認識(bookのみ)**

detect_itg_rs.py dataset_rs.py

**深度情報と骨格情報と物体認識の紐付け**

detect_mp_rs.py dataset_rs.py

**2次元での認識情報の環境地図反映(携帯のトラッキング)**

detect_drawing.py

**3次元での認識情報の環境地図反映**

detect_drawing_rs.py dataset_rs.py


https://user-images.githubusercontent.com/69960076/180616514-b8f8d49d-bbd3-4b06-abbd-31b91561a0d5.mov



**環境地図逆ver**

detect_drawing_rs_reverse.py　dataset_rs.py

**mpの認識情報及び物体とのインタラクション認識の検索，通知**

detect_itg_rst.py(2次元) detect_drawing_rs_rst.py(3次元)　result_itg.csv search_itg.py

**学習用クライアント・サーバシステム**

server.py clients.py dataset_rs.py

**モデル学習用ファイル**

training.py training_diff.py(別の検証データでの精度表示)

**Action学習**

action_init.py action_pre.py

**Interaction学習**

interaction_init.py interaction_pre.py(for Realsense) interaction_init.py interaction_pre.py(for 2D webcam)

**2次元でのインタラクション認識**

interaction_allprint.py(全model結果表示) interaction.py(認識精度に応じた出力)



https://user-images.githubusercontent.com/69960076/180616635-90c07aea-c4b1-49c9-a884-99e1812cd0f3.mp4

**3次元でのインタラクション認識**

interaction_rs.py interaction_rs_2d.py(認識に使うのは2次元pkl)

**インタラクション認識　クライアント・サーバシステム**

interaction_server.py interaction_client.py dataset_rs.py

**ターゲット物体推定**

estimateTarget.py dataset_rs.py

**人の認識情報統合**

utilization.py dataset_rs.py

**複数人骨格認識**

cropping.py(骨格のみ) detect_cropping(動作推論) 

**実験用(表示確認)**

demo/architecture.py(object, object+pose, pose)

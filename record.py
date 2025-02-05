import cv2
import mediapipe as mp
import time

# 初始化 MediaPipe 手部模型
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 設定攝影機
cap = cv2.VideoCapture(0)

# 控制變數
recording = False
output_file = "hand_landmarks.txt"

# 初始化 Hand 模型
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("無法讀取攝影機影像")
            break

        # 翻轉影像以符合鏡像效果
        frame = cv2.flip(frame, 1)

        # 轉換 BGR 影像到 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 偵測手部
        results = hands.process(image)

        # 轉換回 BGR 以顯示
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 畫出手部關鍵點
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 如果開始錄製，儲存關鍵點座標
                if recording:
                    with open(output_file, "a") as f:
                        f.write(f"Timestamp: {time.time()}\n")
                        for id, lm in enumerate(hand_landmarks.landmark):
                            f.write(f"Landmark {id}: {lm.x}, {lm.y}, {lm.z}\n")
                            print(f"Landmark {id}: {lm.x}, {lm.y}, {lm.z}\n")
                        f.write("\n")

        # 顯示影像
        cv2.imshow('Hand Tracking', image)

        # 鍵盤控制
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 鍵結束程式
            break
        elif key == 18:  # Ctrl + R 開始或停止錄製 (Ctrl 鍵為17, R鍵為82，Ctrl+R組合為18)
            recording = not recording
            if recording:
                print("開始錄製手部關鍵點...")
            else:
                print("停止錄製。")

# 釋放資源
cap.release()
cv2.destroyAllWindows()

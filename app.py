import cv2
import mediapipe as mp
import time
import numpy as np
import random

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# ===== Focus 稳定检测 =====
focus_frame_count = 0
required_focus_frames = 15
focus_start_time = None
stable_focus_time = 0.0

# ===== 困倦检测 =====
eye_closed_frames = 0
drowsy_threshold = 10

# ===== 游戏参数 =====
game_score = 0
target_score = 50
game_result = "Playing"

player_x = 320
player_speed = 12

bullets = []
enemies = []

# ===== 2D射击游戏变量 =====
player_x = 320
bullets = []
enemies = []

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(landmarks, w, h):
    left = [33, 160, 158, 133, 153, 144]
    right = [362, 385, 387, 263, 373, 380]

    def eye_ratio(indices):
        p = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        vertical1 = distance(p[1], p[5])
        vertical2 = distance(p[2], p[4])
        horizontal = distance(p[0], p[3])

        if horizontal == 0:
            return 0.0

        return (vertical1 + vertical2) / (2.0 * horizontal)

    left_ear = eye_ratio(left)
    right_ear = eye_ratio(right)
    return (left_ear + right_ear) / 2.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    h, w, _ = frame.shape

    raw_status = "No Face"
    stable_status = "Not Focused"
    eye_status = "Unknown"
    turn_offset = 0

    # 参考线
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 0), 2)
    cv2.line(frame, (0, int(h * 0.62)), (w, int(h * 0.62)), (255, 255, 0), 2)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        nose = lm[1]
        left_eye = lm[33]
        right_eye = lm[263]

        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)
        left_eye_x = int(left_eye.x * w)
        left_eye_y = int(left_eye.y * h)
        right_eye_x = int(right_eye.x * w)
        right_eye_y = int(right_eye.y * h)

        # 关键点可视化
        cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 255), -1)
        cv2.circle(frame, (left_eye_x, left_eye_y), 4, (255, 0, 255), -1)
        cv2.circle(frame, (right_eye_x, right_eye_y), 4, (255, 0, 255), -1)

        # ===== 原始状态 =====
        eye_distance = abs(right_eye_x - left_eye_x)
        left_dist = abs(nose_x - left_eye_x)
        right_dist = abs(right_eye_x - nose_x)
        eye_balance = abs(left_dist - right_dist)
        turn_offset = left_dist - right_dist
        looking_down = nose_y > int(h * 0.62)

        if eye_distance < 70:
            raw_status = "No Face"
        elif looking_down:
            raw_status = "Looking Down"
        elif eye_balance < 35:
            raw_status = "Focused"
        else:
            raw_status = "Looking Away"

        # ===== 闭眼判断 =====
        ear = compute_ear(lm, w, h)

        if ear < 0.22:
            eye_status = "Closed"
            eye_closed_frames += 1
        else:
            eye_status = "Open"
            eye_closed_frames = 0

        # EAR 显示
        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ===== 稳定状态 =====
        if eye_closed_frames > drowsy_threshold:
            stable_status = "Drowsy!"
            focus_frame_count = 0
            focus_start_time = None
            stable_focus_time = 0.0
        else:
            if raw_status == "Focused":
                focus_frame_count += 1

                if focus_frame_count >= required_focus_frames:
                    stable_status = "Focused"

                    if focus_start_time is None:
                        focus_start_time = time.time()

                    stable_focus_time = time.time() - focus_start_time
                else:
                    stable_status = "Focusing..."
            else:
                focus_frame_count = 0
                focus_start_time = None
                stable_focus_time = 0.0
                stable_status = raw_status

    # ===== 游戏逻辑 =====
    if game_result == "Playing":
        # ===== 头控左右移动 =====
        # turn_offset < 0：往一边移动
        # turn_offset > 0：往另一边移动
        if turn_offset < -8:
            player_x -= player_speed
        elif turn_offset > 8:
            player_x += player_speed

        # 限制玩家不能移出屏幕
        if player_x < 20:
            player_x = 20
        elif player_x > w - 20:
            player_x = w - 20

        # 生成敌人
        if random.random() < 0.02:
            enemies.append([random.randint(50, w - 50), 0])

        # 移动敌人
        for enemy in enemies:
            enemy[1] += 3

        # 专注时自动发射子弹
        if stable_status == "Focused":
            if len(bullets) < 12:
                bullets.append([player_x, h - 80])

        # 子弹移动
        for bullet in bullets:
            bullet[1] -= 18

        # 碰撞检测：子弹打中敌人
        for bullet in bullets[:]:
            for enemy in enemies[:]:
                if abs(bullet[0] - enemy[0]) < 28 and abs(bullet[1] - enemy[1]) < 28:
                    bullets.remove(bullet)
                    enemies.remove(enemy)
                    game_score += 5
                    break

        if game_score >= target_score:
            game_result = "You Win!"

        # 清理飞出屏幕的子弹
        bullets = [b for b in bullets if b[1] > 0]

        # 清理掉出屏幕的敌人
        enemies = [e for e in enemies if e[1] < h]

        game_score = max(0.0, min(target_score, game_score))

        if game_score >= target_score:
            game_result = "You Win!"
    # ===== 画 2D 射击元素 =====
    # 玩家
    cv2.circle(frame, (player_x, h - 40), 15, (255, 0, 0), -1)

    # 子弹
    for bullet in bullets:
        cv2.circle(frame, (bullet[0], bullet[1]), 5, (0, 255, 0), -1)

    # 敌人
    for enemy in enemies:
        cv2.circle(frame, (enemy[0], enemy[1]), 10, (0, 0, 255), -1)
    # ===== UI 显示 =====
    cv2.putText(
        frame,
        f"Raw: {raw_status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Stable: {stable_status}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Eyes: {eye_status}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Turn: {turn_offset}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2)

    cv2.putText(
        frame,
        f"Focus Time: {stable_focus_time:.1f}s",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Score: {game_score}/{target_score}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # 分数进度条
    bar_x1, bar_y1 = 20, 280
    bar_x2, bar_y2 = 320, 310
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), 2)

    fill_width = int((game_score / target_score) * (bar_x2 - bar_x1))
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_width, bar_y2), (0, 255, 0), -1)

    # 游戏结果
    if game_result == "You Win!":
        cv2.putText(
            frame,
            "You Win!",
            (w // 2 - 100, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            4
        )
    elif game_result == "Game Over":
        cv2.putText(
            frame,
            "Game Over",
            (w // 2 - 130, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            4
        )

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2
    )

    cv2.imshow("Focus Game V6", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

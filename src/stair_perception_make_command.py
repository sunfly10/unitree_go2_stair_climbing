#!/usr/bin/env python3
# ros2 launch realsense2_camera rs_launch.py 실행
# unitree_ros2 pkg 필요(라이다 토픽 사용--> 라이다 사용 x), 파라미터들은 환경에 맞춰 변경해서 사용 /완전 굿굿

from datetime import datetime #시간, 날짜 등 사용을 위해
from ultralytics import YOLO #계단 인식을 위해
#파이썬에서 ros 사용을 위해
import rclpy 
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data #데이터 수신을 잘 받기 위해
#메시지 타입
from sensor_msgs.msg import Image, PointCloud2 
from std_msgs.msg import String, Float32
#이미지 처리를 위해서는 cv img가 필요
from cv_bridge import CvBridge 
import cv2
import os #운영체제 명령어 상용하기 위해
import sys
import math 
import numpy as np
import time

class StepPerceptionMakeCommand(Node):
    def __init__(self):
        super().__init__('step_perception_make_command_node')

        self.TARGET_CLASS_ID = 1   
        
        self.CONF_SEARCH = 0.70  #처음 찾을 땐 깐깐하게 (확실한 것만)
        self.CONF_TRACK = 0.40   #움직일 땐 너그럽게 (흔들려도 안 놓치게)
        #결과 저장 경로
        self.SAVE_DIR = os.path.expanduser("~/go2_ws/evidence/")
        if not os.path.exists(self.SAVE_DIR): os.makedirs(self.SAVE_DIR)
        #계단 모델 로드
        model_path = os.path.expanduser("~/go2_ws/best.pt")
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            self.get_logger().info(f"모델 로드 성공: {model_path}")
        else:
            self.get_logger().error(f"모델 없음: {model_path}")
            sys.exit()

        self.IMG_W, self.IMG_H = 1280, 720 #이미지 크기
         
        #계단까지 거리 최소 25cm(이 근방이면 오케이), d435i 수직 화각: 58도, 일반적인 피치 최대 각도 : +-0.75rad(43도), 서있을때 33+4(카메라)cm --> 최대각도로 숙였을 때 17cm 정도
        #정지시 --> 지면으로부터 카메라의 위치/계단까지의 거리 = tan(로봇의 현재 숙인 각도 + 29도*((stopline-360)/360)
        self.MAX_PITCH = 0.60 #최대 각도까지는 도달하지 못하게
        self.STOP_LINE = 676 #위의 공식을 활용해 
        self.DEADZONE = 140 #이동시 물체가 흔들리는데 yolo 박스의 가운데를 기준으로 좌우 150까지는 오차 허용(회전 시 사용)
        self.PITCH_LINE = 480 #맨 아래 계단은 시야가 중심이 아닌 하단부에 있음(3m 근방부터 피치 제어 시작)
        self.ALPHA = 0.12 #목표값에 얼마큼 가중치
        self.smooth_pitch = 0.0 #목표 pitch로 바로 동작시키지 않고 과거와 목표값에 가중치를 두어 부드럽게 작동시킨다.
        self.last_sent_pitch = 0.0 #마지막에 보낸 값
        #cv img를 위해 사용하는 변수
        self.bridge = CvBridge()
        self.cv_img = None
        #라이다 관련 변수
        self.lidar_safe = True
        self.lidar_points = None 
        
        self.state = "SEARCH" #맨 처음에는 당연히 search
        self.arrival_timer = 0
        self.is_emergency_saved = False
        self.is_search_saved = False

        self.create_subscription(Image, '/camera/color/image_raw', self.color_cb, qos_profile_sensor_data)
        #self.create_subscription(PointCloud2, '/utlidar/cloud1', self.lidar_cb, qos_profile_sensor_data)
        
        self.pub_cmd = self.create_publisher(String, '/robot/decision_cmd', 10)
        self.pub_pitch = self.create_publisher(Float32, '/robot/body_pitch_cmd', 10)
        
        self.timer = self.create_timer(0.05, self.timer_cb) 
    #opencv img로 변환
    def color_cb(self, msg): 
        try: self.cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e: self.get_logger().error(f"이미지 변환 오류: {e}")

    def lidar_cb(self, msg):
        try:
            #점 하나당 float가 몇 개인지 계산 (자동 감지)
            #msg.point_step은 점 하나당 바이트 수 (예: 32 or 16)
            #float32는 4바이트이므로 4로 나눔
            dims = msg.point_step // 4
            points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, dims) #데이터 float형으로 가져오기
            #로봇이 신경써야할 영역
            nearby_mask = (points[:, 0] < 2.0) & (points[:, 0] > 0.3) & (abs(points[:, 1]) < 1.0)
            self.lidar_points = points[nearby_mask, :2]
            #위험 판단 영역 
            danger_mask = (points[:, 0] > 0.20) & (points[:, 0] < 0.15) &\
                   (np.abs(points[:, 1]) < 0.15) & (np.abs(points[:, 1]) > 0.05) &\
                   (points[:, 2] > 0.15)
            if len(points[danger_mask]) > 10: self.lidar_safe = False
            else: self.lidar_safe = True
        except: pass
    #비상 상황시 라이다 2D 맵 그리기
    def draw_lidar_map(self, img):
        if self.lidar_points is None: return
        map_size = 300
        scale = 100  
        overlay = img.copy()
        start_x = self.IMG_W - map_size - 20
        start_y = self.IMG_H - map_size - 20
        cv2.rectangle(overlay, (start_x, start_y), (start_x + map_size, start_y + map_size), (0, 0, 0), -1) 
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img) 
        robot_x_map = start_x + map_size // 2
        robot_y_map = start_y + map_size - 20
        box_x1 = robot_x_map - int(0.25 * scale) 
        box_x2 = robot_x_map + int(0.25 * scale) 
        box_y1 = robot_y_map - int(0.40 * scale) 
        box_y2 = robot_y_map - int(0.15 * scale) 
        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), 1) 
        for point in self.lidar_points:
            x_real, y_real = point[0], point[1]
            px = robot_x_map - int(y_real * scale)
            py = robot_y_map - int(x_real * scale)
            if not (start_x < px < start_x + map_size and start_y < py < start_y + map_size):
                continue
            if (0.15 < x_real < 0.30) and (abs(y_real) < 0.15): color = (0, 0, 255)
            else: color = (0, 255, 0)
            cv2.circle(img, (px, py), 2, color, -1)
        cv2.circle(img, (robot_x_map, robot_y_map), 8, (255, 0, 0), -1)
        cv2.putText(img, "LiDAR Map", (start_x + 10, start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def calculate_pitch_ratio(self, y2):
        #YOLO는 박스의 일부가 화면 밖으로 나가기 시작하면 박스의 크기를 급격하게 줄이거나 인식을 놓쳐버리는다 --> 밑변 y2를 사용한다, 박스의 중심점 사용 x --> 더 안정적이락 생각
        if y2 < self.PITCH_LINE: return 0.0
        ratio = (y2 - self.PITCH_LINE) / (self.STOP_LINE - self.PITCH_LINE)
        return min(max(ratio, 0.0), 1.0) * self.MAX_PITCH

    def save_evidence(self, img, prefix):
        filename = f"{prefix}_{datetime.now().strftime('%H%M%S_%f')[:-3]}.jpg"
        cv2.imwrite(os.path.join(self.SAVE_DIR, filename), img)
        self.get_logger().info(f"증거 사진 저장됨: {filename}")

    def timer_cb(self):
        if self.cv_img is None: return
        vis_img = self.cv_img.copy()
        #피치 제어, 정지선 그리기
        cv2.line(vis_img, (0, self.PITCH_LINE), (self.IMG_W, self.PITCH_LINE), (0, 255, 255), 2) 
        cv2.line(vis_img, (0, self.STOP_LINE), (self.IMG_W, self.STOP_LINE), (0, 0, 255), 2)   

        target_pitch = self.last_sent_pitch 
        cmd_msg = "STOP"

        if not self.lidar_safe:
            cmd_msg = "STOP"
            self.draw_lidar_map(vis_img) 
            if not self.is_emergency_saved: 
                cv2.putText(vis_img, "EMERGENCY STOP(LIDAR)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                self.save_evidence(vis_img, "EMERGENCY")
                self.is_emergency_saved = True 
            self.state = "SEARCH"
            
            self.pub_cmd.publish(String(data=cmd_msg))
            self.pub_pitch.publish(Float32(data=float(self.smooth_pitch)))
            print(f"State: {self.state} | Cmd: {cmd_msg} | Pitch: {self.smooth_pitch:.2f} (LIDAR CLOSE)", end='\r')
            return
        else:
            self.is_emergency_saved = False 

            #상태에 따라 임계값 변경
            if self.state == "SEARCH":
                current_conf = self.CONF_SEARCH #0.70 (깐깐하게)
            else:
                current_conf = self.CONF_TRACK  #0.40 (너그럽게)

            results = self.model(self.cv_img, verbose=False, conf=current_conf)
            best_box = None
            max_y2 = -1.0 
            #여러 YOLO 박스 중 가장 아래에 있는 박스 사용
            if results and len(results[0].boxes) > 0:
                for box_data in results[0].boxes:
                    if int(box_data.cls[0]) == self.TARGET_CLASS_ID:
                        x1, y1, x2, y2 = box_data.xyxy[0].cpu().numpy()
                        if y2 > max_y2:
                            max_y2 = y2
                            best_box = [x1, y1, x2, y2]
            #최고의 박스 표시(분홍색)
            if best_box is not None:
                bx1, by1, bx2, by2 = map(int, best_box)
                cv2.rectangle(vis_img, (bx1, by1), (bx2, by2), (255, 0, 255), 3) 
                cv2.circle(vis_img, (int((bx1 + bx2) / 2), int((by1 + by2) / 2)), 5, (0, 0, 255), -1)
            #계단 하단부 찾으면 이동
            if self.state == "SEARCH":
                target_pitch = self.last_sent_pitch 
                if best_box is not None:
                    self.save_evidence(vis_img, "FOUND")
                    self.state = "TRACK"
                    self.is_search_saved = False 
                else:
                    cmd_msg = "SEARCH"
                    if not self.is_search_saved:
                        self.save_evidence(vis_img, "SEARCH_FAIL")
                        self.is_search_saved = True 
                    
            elif self.state == "TRACK":
                if best_box is not None:
                    calculated_target = self.calculate_pitch_ratio(max_y2)
                    #정지선 넘는 프레임이 연속해서 3번이면 정지
                    if max_y2 >= self.STOP_LINE:
                        #self.arrival_timer += 1
                        #if self.arrival_timer >= 3: 
                        self.save_evidence(vis_img, "FINISH")
                        self.state = "FINISH"
                            #self.arrival_timer = 0 
                        cmd_msg = "STOP"
                        self.pub_cmd.publish(String(data="CLIMB"))
                        print("도착 완료. 즉시 종료합니다.")
                        time.sleep(0.1)
                        sys.exit(0)
                    else:
                        self.arrival_timer = 0
                        bx = (best_box[0] + best_box[2]) / 2
                        err = (self.IMG_W / 2) - bx
                        #조향 맞추기
                        if abs(err) > self.DEADZONE:
                            cmd_msg = "TURN_LEFT" if err > 0 else "TURN_RIGHT"
                            target_pitch = calculated_target
                        else:
                            cmd_msg = "FORWARD"
                            target_pitch = calculated_target
                else:
                    #TRACK 중에 놓치면 다시 SEARCH로 (이때 임계값 다시 0.7로 올라감)
                    self.state = "SEARCH"
                    self.is_search_saved = False 
                    
        #new pitch = target pitch * a + old pitch * (1 - a) : 지수 이동 평균()
        #과거값에 더 가중치를 --> 부드럽게 움직이게 --> 움직임이 급격하면 카메라의 시야 변동이 커서
        self.smooth_pitch = (target_pitch * self.ALPHA) + (self.smooth_pitch * (1.0 - self.ALPHA))

        self.pub_cmd.publish(String(data=cmd_msg))
        self.pub_pitch.publish(Float32(data=float(self.smooth_pitch)))
        self.last_sent_pitch = self.smooth_pitch

        # 디버깅: 현재 적용된 conf 값도 같이 출력
        print(f"State: {self.state} | Cmd: {cmd_msg} | Conf: {current_conf} | Pitch: {self.smooth_pitch:.2f} ", end='\r')

def main():
    rclpy.init()
    node = StepPerceptionMakeCommand()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()

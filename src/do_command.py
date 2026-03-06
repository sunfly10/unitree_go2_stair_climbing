#!/usr/bin/env python3
# 이동 1008 + 자세 1007 + Pose모드 1028 

#파이썬에서 ros 사용을 위해
import rclpy
from rclpy.node import Node
#메시지 타입
from std_msgs.msg import String, Float32
from unitree_api.msg import Request 
import json 
import sys

class DoCommand(Node):
    def __init__(self):
        super().__init__('do_command_node')
        
        self.create_subscription(String, '/robot/decision_cmd', self.cmd_cb, 10)
        self.create_subscription(Float32, '/robot/body_pitch_cmd', self.pitch_cb, 10)
        
        self.pub_req = self.create_publisher(Request, '/api/sport/request', 10)
        
        self.current_pitch = 0.0
        self.get_logger().info("명령 받는 중...")
        self.command = "WAIT"

        #이 모드가 켜져야 피치(Euler) 제어가 먹힘
        self.init_timer = self.create_timer(1.0, self.initial_setup)
        self.setup_done = False

    def initial_setup(self):
        if not self.setup_done:
            self.get_logger().info("Pose(1028) 모드 ON! (자세 제어 활성화)")
            req = Request()
            req.header.identity.id = 0
            req.header.identity.api_id = 1028 #Pose API ID
            
            # 파라미터: {"data": True} -> 1028 모드 켜기
            param = {"data": True}
            req.parameter = json.dumps(param)
            
            self.pub_req.publish(req)
            
            self.setup_done = True
            self.init_timer.cancel()

    #이동 제어 (API ID: 1008)
    def cmd_cb(self, msg):
        cmd = msg.data
        self.command = cmd
        req = Request()
        req.header.identity.id = 0
        req.header.identity.api_id = 1008 #Move API
        
        vx, vy, wz = 0.0, 0.0, 0.0
        
        if cmd == "FORWARD": vx = 0.2
        elif cmd == "TURN_LEFT": 
            vx = 0.1
            wz = 0.3
        elif cmd == "TURN_RIGHT": 
            vx = 0.1
            wz = -0.3
        elif cmd == "STOP": pass 
        elif cmd == "SEARCH": pass
        elif cmd == "CLIMB": 
            print("계단 앞까지 접근 완료")
            sys.exit()

        param = {"x": vx, "y": vy, "z": wz}
        req.parameter = json.dumps(param)
        self.pub_req.publish(req)
        print(f"Posture Cmd: {self.command}")

    #자세 제어 (API ID: 1007)
    def pitch_cb(self, msg):
        target_pitch = msg.data 

        req = Request()
        req.header.identity.id = 0
        req.header.identity.api_id = 1007 #Euler API
        
        param = {
            "x": 0.0,             
            "y": float(target_pitch), 
            "z": 0.0              
        }
        req.parameter = json.dumps(param)
        
        self.pub_req.publish(req)

        print(f"Pitch: {target_pitch:.2f}")

def main():
    rclpy.init()
    node = DoCommand()
    try: rclpy.spin(node)
    except: pass
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': main()

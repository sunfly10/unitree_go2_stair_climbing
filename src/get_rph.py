#imu의 r p h를 얻어보는 코드:
# imu의 rph
# ROLL: 125.63 deg | PITCH:  -0.66 deg | YAW:   0.17 deg
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data # 센서 데이터용 QoS
from sensor_msgs.msg import Imu
import math

class ImuRPYChecker(Node):
    def __init__(self):
        super().__init__('imu_rpy_checker_node')

        self.sub = self.create_subscription(
            Imu,
            '/utlidar/imu',          # 실제 IMU 토픽 이름 확인
            self.imu_cb,
            qos_profile_sensor_data
        )
        self.get_logger().info("IMU RPY checker started")

    #쿼터니언(x, y, z, w)을 오일러 각(roll, pitch, yaw)으로 변환하는 함수
    def euler_from_quaternion(self, x, y, z, w):
        # roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
        # pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
        # yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return roll_x, pitch_y, yaw_z
    
    def imu_cb(self, msg):
        q = msg.orientation
        roll, pitch, yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.get_logger().info(
            f"ROLL: {math.degrees(roll):6.2f} deg | "
            f"PITCH: {math.degrees(pitch):6.2f} deg | "
            f"YAW: {math.degrees(yaw):6.2f} deg"
        )

def main():
    rclpy.init()
    node = ImuRPYChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

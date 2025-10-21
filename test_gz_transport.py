import time
from gz.transport14 import Node as GzNode
from gz.msgs11.pose_v_pb2 import Pose_V

class GazeboTransportTest:
    def __init__(self):
        self.gz_node = GzNode()
        self.robot_model_name = 'turtlebot4'
        self.model_pose_topic = f"/model/{self.robot_model_name}/pose"
        
        self.pose_received = False
        self.latest_pose = None
        
        print(f"订阅话题: {self.model_pose_topic}")
        
        # Subscribe to model pose topic
        success = self.gz_node.subscribe(Pose_V, self.model_pose_topic, self._pose_callback)
        if success:
            print("✅ 成功订阅话题")
        else:
            print("❌ 订阅话题失败")
    
    def _pose_callback(self, msg):
        """处理位姿消息"""
        try:
            # Find the main robot pose
            robot_pose = None
            for pose in msg.pose:
                if pose.name == self.robot_model_name:
                    robot_pose = pose
                    break
            
            if robot_pose is not None:
                self.latest_pose = {
                    'position': {
                        'x': robot_pose.position.x,
                        'y': robot_pose.position.y,
                        'z': robot_pose.position.z
                    },
                    'orientation': {
                        'x': robot_pose.orientation.x,
                        'y': robot_pose.orientation.y,
                        'z': robot_pose.orientation.z,
                        'w': robot_pose.orientation.w
                    }
                }
                self.pose_received = True
                print(f"🎯 接收到机器人位姿: x={robot_pose.position.x:.3f}, y={robot_pose.position.y:.3f}, z={robot_pose.position.z:.3f}")
        except Exception as e:
            print(f"❌ 处理位姿消息时出错: {e}")
    
    def test_connection(self, timeout=10):
        """测试连接并等待消息"""
        print(f"等待 {timeout} 秒来接收位姿消息...")
        
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if self.pose_received:
                print("✅ 成功接收到位姿数据!")
                print(f"最新位姿数据: {self.latest_pose}")
                return True
            time.sleep(0.1)
        
        print("❌ 超时：未接收到位姿数据")
        return False

if __name__ == "__main__":
    print("🧪 测试 Gazebo Transport 连接...")
    
    test = GazeboTransportTest()
    success = test.test_connection()
    
    if success:
        print("\n🎉 Gazebo Transport 连接测试成功!")
    else:
        print("\n💥 Gazebo Transport 连接测试失败!")
        print("请确保:")
        print("1. Gazebo 仿真正在运行")
        print("2. turtlebot4 模型已加载")
        print("3. 话题 /model/turtlebot4/pose 可用")
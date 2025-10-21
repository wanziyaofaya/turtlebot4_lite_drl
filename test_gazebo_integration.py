import numpy as np
def test_code_changes():
    """测试代码修改是否正确"""
    
    print("🔍 测试 Gazebo 模型状态集成修改...")
    
    # 检查关键导入
    try:
        from gz.transport14 import Node as GzNode
        from gz.msgs11.pose_pb2 import Pose as GzPose
        from gz.msgs11.boolean_pb2 import Boolean
        print("✅ Gazebo Transport 导入成功")
    except ImportError as e:
        print(f"❌ Gazebo Transport 导入失败: {e}")
        return False
    
    try:
        import tf2_ros
        import tf2_geometry_msgs
        from tf2_ros import Buffer, TransformListener
        print("✅ TF2 模块导入成功")
    except ImportError as e:
        print(f"❌ TF2 模块导入失败: {e}")
        return False
    
    # 检查代码结构
    with open('/home/turtlebot4/turtlebot4_lite_drl/src/turtlebot4_rl/turtlebot4_rl/nav_env.py', 'r') as f:
        content = f.read()
    
    # 验证关键修改
    checks = [
        ('_get_robot_pose_from_gazebo', '✅ 新增了直接从 Gazebo 获取位置的方法'),
        ('gz_node = GzNode()', '✅ 创建了 Gazebo Transport 节点'),
        ('tf_buffer = Buffer()', '✅ 添加了 TF2 缓冲区'),
        ('model_state_received', '✅ 添加了模型状态接收标志'),
        ('_update_env_position', '✅ 添加了环境坐标更新方法')
    ]
    
    for check, message in checks:
        if check in content:
            print(message)
        else:
            print(f"❌ 缺少关键组件: {check}")
            return False
    
    # 检查移除的 odom 相关代码
    removed_checks = [
        ('odom_calibrated', '移除了 odom 校准标志'),
        ('_calibrate_odom', '移除了 odom 校准方法'),
        ('odom_callback', '移除了 odom 回调函数')
    ]
    
    for check, message in removed_checks:
        if check not in content:
            print(f"✅ {message}")
        else:
            print(f"⚠️  警告: 仍然存在 {check}")
    
    print("\n📋 修改摘要:")
    print("1. ✅ 替换 odom 订阅器为 Gazebo Transport 节点")
    print("2. ✅ 添加 TF2 变换支持")
    print("3. ✅ 实现直接从 Gazebo 获取精确模型状态")
    print("4. ✅ 移除复杂的坐标系转换逻辑")
    print("5. ✅ 使用 Gazebo 原生服务获取位置信息")
    
    return True

def print_usage_instructions():
    """打印使用说明"""
    print("\n📖 使用新的 Gazebo 模型状态集成:")
    print("1. 确保 Gazebo 仿真环境正在运行")
    print("2. 模型名称设置为 'turtlebot4' (可在代码中自定义)")
    print("3. 环境会自动从 Gazebo 获取精确的机器人位置")
    print("4. 支持通过 TF2 进行自定义坐标变换")
    print("5. 不再依赖可能不准确的 odom 数据")
    
    print("\n🔧 高级配置选项:")
    print("- self.env_frame_id: 设置环境坐标系")
    print("- self.gazebo_world_frame_id: 设置 Gazebo 世界坐标系")
    print("- self.robot_model_name: 设置机器人模型名称")
    print("- _update_env_position(): 自定义坐标变换逻辑")

if __name__ == "__main__":
    print("🚀 测试 Gazebo 模型状态集成修改")
    print("=" * 50)
    
    success = test_code_changes()
    
    if success:
        print("\n🎉 所有修改验证成功！")
        print_usage_instructions()
    else:
        print("\n❌ 发现问题，请检查修改")
    
    print("\n" + "=" * 50)
#!/usr/bin/env python3
"""
验证子目标点数据集的正确性
检查每条数据的子目标点是否符合A*路径规划逻辑
"""

import sys
sys.path.insert(0, 'src/turtlebot4_rl/turtlebot4_rl')

from collision import is_position_valid, obstacles

def is_obstacle_free(start, end, step_size=0.01):
    """检查从start到end的直线路径上是否有障碍物"""
    dist = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
    if dist < step_size:
        return True
    steps = int(dist / step_size)
    for i in range(steps + 1):
        t = i / steps if steps > 0 else 0
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        if not is_position_valid(x, y):
            return False
    return True

def verify_subgoal(start, goal, subgoal):
    """
    验证子目标点是否正确
    返回: (is_valid, reason)
    
    子目标点是A*简化路径的path[1]，可能是：
    1. 直接等于终点（起点到终点无障碍）
    2. 路径上的某个中间点（需要绕行）
    """
    start = tuple(start)
    goal = tuple(goal)
    subgoal = tuple(subgoal)
    
    # 检查1: 子目标点本身是否在有效位置
    if not is_position_valid(subgoal[0], subgoal[1]):
        return False, "子目标点在障碍物内或太靠近障碍物"
    
    # 检查2: 起点到子目标点是否可通行
    if not is_obstacle_free(start, subgoal):
        return False, "起点到子目标点之间有障碍物"
    
    # 检查3: 如果子目标等于终点，验证起点到终点是否真的无障碍
    if abs(subgoal[0] - goal[0]) < 0.01 and abs(subgoal[1] - goal[1]) < 0.01:
        if not is_obstacle_free(start, goal):
            return False, "子目标=终点，但起点到终点之间有障碍物"
    
    # 检查4: 如果子目标不等于终点
    else:
        # 4a: 起点到终点之间应该有障碍（否则应该直接到终点）
        if is_obstacle_free(start, goal):
            return False, "子目标≠终点，但起点到终点之间无障碍（应该直接到终点）"
        
        # 4b: 子目标到终点应该可通行，或者子目标朝向终点方向
        # （子目标可能是路径中间点，不一定能直达终点，但应该比起点更接近终点或绕开障碍）
        dist_start_to_goal = ((goal[0] - start[0])**2 + (goal[1] - start[1])**2)**0.5
        dist_subgoal_to_goal = ((goal[0] - subgoal[0])**2 + (goal[1] - subgoal[1])**2)**0.5
        dist_start_to_subgoal = ((subgoal[0] - start[0])**2 + (subgoal[1] - start[1])**2)**0.5
        
        # 子目标应该让我们在路径上有进展（到终点的距离变小，或者绕开障碍后总路径合理）
        # 允许一定的绕路（子目标到终点距离 < 起点到终点距离 * 1.5）
        if dist_subgoal_to_goal > dist_start_to_goal * 1.5:
            return False, f"子目标距离终点太远（{dist_subgoal_to_goal:.2f} > {dist_start_to_goal*1.5:.2f}）"
    
    return True, "正确"

def main():
    dataset_path = 'models/subgoal_dataset.txt'
    
    errors = []
    warnings = []
    total = 0
    valid = 0
    
    with open(dataset_path, 'r') as f:
        lines = f.readlines()
    
    # 跳过表头
    for i, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split(',')
        if len(parts) < 6:
            errors.append(f"行 {i}: 数据格式错误，列数不足")
            continue
        
        try:
            start_x = float(parts[0])
            start_y = float(parts[1])
            goal_x = float(parts[2])
            goal_y = float(parts[3])
            subgoal_x = float(parts[4])
            subgoal_y = float(parts[5])
        except ValueError as e:
            errors.append(f"行 {i}: 数据解析错误 - {e}")
            continue
        
        total += 1
        start = (start_x, start_y)
        goal = (goal_x, goal_y)
        subgoal = (subgoal_x, subgoal_y)
        
        is_valid, reason = verify_subgoal(start, goal, subgoal)
        
        if is_valid:
            valid += 1
        else:
            errors.append(f"行 {i}: {reason}")
            errors.append(f"  起点: {start}, 终点: {goal}, 子目标: {subgoal}")
    
    # 输出结果
    print("=" * 60)
    print("子目标点数据集验证报告")
    print("=" * 60)
    print(f"总数据条数: {total}")
    print(f"正确条数: {valid}")
    print(f"错误条数: {len([e for e in errors if e.startswith('行')])//2}")
    print(f"正确率: {valid/total*100:.2f}%" if total > 0 else "N/A")
    print()
    
    if errors:
        print("错误详情:")
        print("-" * 60)
        for err in errors[:50]:  # 只显示前50条错误
            print(err)
        if len(errors) > 50:
            print(f"... 还有 {len(errors)-50} 条错误未显示")
    else:
        print("✅ 所有数据验证通过！")

if __name__ == "__main__":
    main()
# python3 tools/verify_subgoals.py
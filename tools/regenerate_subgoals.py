#!/usr/bin/env python3
"""
重新生成子目标点数据集
保留原数据集的 start_x, start_y, goal_x, goal_y 和 lidar 数据
只重新计算 subgoal_x, subgoal_y（使用新的碰撞阈值）
"""
import sys
import os

# 添加源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'turtlebot4_rl', 'turtlebot4_rl'))

from collision import obstacles, point_in_obstacle
import heapq

# 新的碰撞阈值
NEW_CLEARANCE = 0.20

def is_position_valid_custom(x, y, clearance=NEW_CLEARANCE):
    """使用自定义 clearance 的碰撞检测"""
    if point_in_obstacle(x, y):
        return False
    for ox, oy, w, h in obstacles:
        expanded_left = ox - clearance
        expanded_right = ox + w + clearance
        expanded_bottom = oy - clearance
        expanded_top = oy + h + clearance
        if expanded_left < x < expanded_right and expanded_bottom < y < expanded_top:
            nearest_x = min(max(x, ox), ox + w)
            nearest_y = min(max(y, oy), oy + h)
            dx = x - nearest_x
            dy = y - nearest_y
            if dx * dx + dy * dy <= clearance * clearance:
                return False
    return True

def is_obstacle_free(start, end, step_size=0.01):
    """检查直线路径是否无障碍"""
    dist = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
    steps = max(int(dist / step_size), 1)
    for i in range(steps + 1):
        t = i / steps
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        if not is_position_valid_custom(x, y):
            return False
    return True

def remove_redundant_nodes(path):
    """移除冗余节点"""
    if len(path) < 2:
        return path
    if is_obstacle_free(path[0], path[-1]):
        return [path[0], path[-1]]
    simplified_path = [path[0]]
    for i in range(1, len(path) - 1):
        start = simplified_path[-1]
        end = path[i + 1]
        if not is_obstacle_free(start, end):
            simplified_path.append(path[i])
    simplified_path.append(path[-1])
    return simplified_path

def astar_simple(start, goal, resolution=0.01):
    """简化版 A*，不需要 env 和 lidar"""
    def heuristic(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def to_grid(p):
        return (int(round(p[0] / resolution)), int(round(p[1] / resolution)))

    def from_grid(g):
        return (round(g[0] * resolution, 4), round(g[1] * resolution, 4))

    start_g = to_grid(start)
    goal_g = to_grid(goal)

    if start_g == goal_g:
        return [start, goal]

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start_g, None))
    came_from = {}
    cost_so_far = {start_g: 0}
    final_node = None

    while open_list:
        _, cost, current, parent = heapq.heappop(open_list)
        if current in came_from:
            continue
        came_from[current] = parent
        if current == goal_g:
            final_node = current
            break
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            neighbor_xy = from_grid(neighbor)
            if not is_position_valid_custom(neighbor_xy[0], neighbor_xy[1]):
                continue
            step_cost = (dx**2 + dy**2)**0.5 * resolution
            new_cost = cost + step_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor_xy, goal)
                heapq.heappush(open_list, (priority, new_cost, neighbor, current))

    if not final_node:
        return None

    path = []
    node = final_node
    while node:
        path.append(node)
        node = came_from.get(node, None)
    path.reverse()
    full_path = [from_grid(p) for p in path]
    return remove_redundant_nodes(full_path)


def main():
    input_file = 'models/subgoal_dataset_0.35.txt'
    output_file = 'models/subgoal_dataset_0.2.txt'
    
    print(f"读取: {input_file}")
    print(f"输出: {output_file}")
    print(f"新碰撞阈值: {NEW_CLEARANCE}")
    
    success_count = 0
    fail_count = 0
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        header = fin.readline()
        fout.write(header)  # 写入表头
        
        for i, line in enumerate(fin):
            parts = line.strip().split(',')
            if len(parts) < 70:  # 6 + 64
                continue
            
            start_x, start_y = float(parts[0]), float(parts[1])
            goal_x, goal_y = float(parts[2]), float(parts[3])
            lidar_data = parts[6:]  # 保留原始 lidar 字符串
            
            # 重新规划路径
            path = astar_simple((start_x, start_y), (goal_x, goal_y))
            
            if path is None or len(path) < 2:
                fail_count += 1
                if fail_count <= 10:
                    print(f"[WARN] 第 {i+1} 行规划失败: ({start_x}, {start_y}) -> ({goal_x}, {goal_y})")
                continue
            
            subgoal = path[1]
            lidar_str = ','.join(lidar_data)
            fout.write(f"{start_x:.4f},{start_y:.4f},{goal_x:.4f},{goal_y:.4f},{subgoal[0]:.4f},{subgoal[1]:.4f},{lidar_str}\n")
            success_count += 1
            
            if (i + 1) % 10000 == 0:
                print(f"已处理 {i+1} 行, 成功: {success_count}, 失败: {fail_count}")
    
    print(f"\n完成! 成功: {success_count}, 失败: {fail_count}")
    print(f"输出文件: {output_file}")


if __name__ == '__main__':
    main()

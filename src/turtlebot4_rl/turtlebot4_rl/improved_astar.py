import heapq
import numpy as np
from collision import is_position_valid

def is_obstacle_free(start, end, step_size=0.01):
    """检查从start到end的直线路径上是否有障碍物"""
    dist = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
    steps = int(dist / step_size)
    for i in range(steps + 1):
        t = i / steps
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        if not is_position_valid(x, y):
            return False
    return True

def remove_redundant_nodes(path):
            if len(path) < 2:
                return path
            simplified_path = [path[0]]  # 保留起点
            for i in range(1, len(path) - 1):
                start = simplified_path[-1]
                end = path[i + 1]
                # 检查从start到end的连线是否无障碍
                if is_obstacle_free(start, end):
                    continue  # True,无障碍，中间节点冗余，跳过
                else:
                    simplified_path.append(path[i])  # False,保留当前节点
            simplified_path.append(path[-1])  # 保留终点
            return simplified_path

def astar(start, goal, resolution=0.01, env=None):
    if env is None or env.lidar_data is None:
        raise ValueError("Environment with valid LiDAR data is required to calculate obstacle density.")

    # Calculate obstacle density (p value)
    lidar_data = env.lidar_data
    obstacle_count = np.sum((lidar_data >= 0) & (lidar_data <= 0.85))
    p = obstacle_count / len(lidar_data)

    k = 2

    def heuristic(a, b):
        original_heuristic = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        heuristic = (1 + k * p) * original_heuristic
        return heuristic

    def to_grid(p):
        return (int(round(p[0] / resolution)), int(round(p[1] / resolution)))

    def from_grid(g):
        return (round(g[0] * resolution, 4), round(g[1] * resolution, 4))

    start_g = to_grid(start)
    goal_g = to_grid(goal)

    if start_g == goal_g:
        return [start, goal]

    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]

    # 前向与后向队列
    open_fwd = []
    open_bwd = []
    heapq.heappush(open_fwd, (heuristic(start, goal), 0, start_g, None))
    heapq.heappush(open_bwd, (heuristic(goal, start), 0, goal_g, None))

    came_from_fwd = {}
    came_from_bwd = {}
    cost_fwd = {start_g: 0}
    cost_bwd = {goal_g: 0}

    meet_node = None

    while open_fwd and open_bwd:
        # 从前向扩展
        _, cost, current, parent = heapq.heappop(open_fwd)
        if current in came_from_fwd:
            continue
        came_from_fwd[current] = parent

        if current in came_from_bwd:
            meet_node = current
            break

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            neighbor_xy = from_grid(neighbor)
            if not is_position_valid(neighbor_xy[0], neighbor_xy[1]):
                continue
            new_cost = cost + resolution
            if neighbor not in cost_fwd or new_cost < cost_fwd[neighbor]:
                cost_fwd[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor_xy, goal)
                heapq.heappush(open_fwd, (priority, new_cost, neighbor, current))

        # 从后向扩展
        _, cost, current, parent = heapq.heappop(open_bwd)
        if current in came_from_bwd:
            continue
        came_from_bwd[current] = parent

        if current in came_from_fwd:
            meet_node = current
            break

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            neighbor_xy = from_grid(neighbor)
            if not is_position_valid(neighbor_xy[0], neighbor_xy[1]):
                continue
            new_cost = cost + resolution
            if neighbor not in cost_bwd or new_cost < cost_bwd[neighbor]:
                cost_bwd[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor_xy, start)
                heapq.heappush(open_bwd, (priority, new_cost, neighbor, current))

    if not meet_node:
        return None

    # 回溯路径
    path_fwd = []
    node = meet_node
    while node:
        path_fwd.append(node)
        node = came_from_fwd.get(node, None)
    path_fwd.reverse()

    path_bwd = []
    node = came_from_bwd.get(meet_node, None)
    while node:
        path_bwd.append(node)
        node = came_from_bwd.get(node, None)

    full_path = [from_grid(p) for p in path_fwd + path_bwd]
    new_path = remove_redundant_nodes(full_path)
    return new_path


# 调用的时候要写path = astar(start, goal, resolution=0.01, env=env)
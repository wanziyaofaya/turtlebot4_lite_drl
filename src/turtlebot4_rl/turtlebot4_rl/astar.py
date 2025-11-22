import heapq
from collision import point_in_obstacle


def astar(start, goal, resolution=0.01, env=None):
    """
    A*寻路算法，障碍物由collision.py定义，网格分辨率为resolution。
    start, goal: (x, y) 坐标
    返回路径列表 [(x0, y0), (x1, y1), ...]
    """
    def heuristic(a, b):
        # 使用欧氏距离作为启发式函数
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    # 网格化坐标
    def to_grid(p):
        # 保证浮点精度，避免精度丢失
        return (int(round(p[0] / resolution)), int(round(p[1] / resolution)))
    def from_grid(g):
        # 保证还原时精度不丢失
        return (round(g[0] * resolution, 4), round(g[1] * resolution, 4))

    start_g = to_grid(start)
    goal_g = to_grid(goal)

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start_g, None))
    came_from = {}
    cost_so_far = {start_g: 0}

    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]

    while open_set:
        _, cost, current, parent = heapq.heappop(open_set)
        if current == goal_g:
            # 回溯路径
            path = [current]
            while parent:
                path.append(parent)
                parent = came_from.get(parent, None)
            path = [from_grid(p) for p in reversed(path)]
            return path
        if current in came_from:
            continue
        came_from[current] = parent
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            neighbor_xy = from_grid(neighbor)
            if point_in_obstacle(neighbor_xy[0], neighbor_xy[1]):
                continue
            if neighbor in cost_so_far and cost_so_far[neighbor] <= cost + resolution:
                continue
            cost_so_far[neighbor] = cost + resolution
            priority = cost_so_far[neighbor] + heuristic(neighbor_xy, goal)
            heapq.heappush(open_set, (priority, cost_so_far[neighbor], neighbor, current))
    return None  # 无路径可达



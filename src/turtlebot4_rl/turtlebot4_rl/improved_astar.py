
import heapq
from collision import point_in_obstacle


def astar(start, goal, resolution=0.01):
    """
    双向A*寻路算法，障碍物由collision.py定义，网格分辨率为resolution。
    start, goal: (x, y) 坐标
    返回路径列表 [(x0, y0), (x1, y1), ...]
    """
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
            if point_in_obstacle(neighbor_xy[0], neighbor_xy[1]):
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
            if point_in_obstacle(neighbor_xy[0], neighbor_xy[1]):
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
    return full_path



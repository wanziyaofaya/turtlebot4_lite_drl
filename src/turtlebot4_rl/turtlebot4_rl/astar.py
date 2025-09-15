import heapq
from collision import point_in_obstacle

def line_cross_obstacle(p1, p2, resolution=0.01):
    """
    判断从p1到p2的连线是否穿过障碍物，采样间隔为resolution。
    """
    x1, y1 = p1
    x2, y2 = p2
    dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    steps = int(dist / resolution)
    for i in range(1, steps):
        t = i / steps
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        if point_in_obstacle(x, y):
            return True
    return False

def remove_redundant_nodes(path, resolution=0.01):
    """
    删除路径上的冗余节点。
    原则：如果第i点和第j点连线不穿过障碍物，则中间点都可以删除。
    """
    if not path or len(path) <= 2:
        return path
    new_path = [path[0]]
    i = 0
    while i < len(path) - 1:
        # 最远能连到哪个点
        j = i + 1
        while j < len(path):
            if line_cross_obstacle(path[i], path[j], resolution):
                break
            j += 1
        # j-1是最后一个可以连的点
        new_path.append(path[j-1])
        i = j - 1
    return new_path

def astar(start, goal, resolution=0.01):
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
        return (round(p[0] / resolution), round(p[1] / resolution))
    def from_grid(g):
        return (g[0] * resolution, g[1] * resolution)

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

# 示例用法
if __name__ == "__main__":
    start = (-9.0, 3.0)
    goal = (-2.0, -1.0)
    path = astar(start, goal)
    if path:
        print("A*原始路径:")
        for p in path:
            print(p)
        print("\nA*精简后路径:")
        simple_path = remove_redundant_nodes(path)
        for p in simple_path:
            print(p)
    else:
        print("无可行路径！")

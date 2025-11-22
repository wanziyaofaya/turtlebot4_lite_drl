import numpy as np
import json

# 地图边界和最小距离要求（与nav_env.py一致）
MAP_BOUNDS = {
    'x_min': -2,
    'x_max': 2,
    'y_min': -2,
    'y_max': 2
}
MIN_DISTANCE = 2.0
NUM_PAIRS = 6000
DEFAULT_CLEARANCE = 0.4

obstacles = [
    #4*4
    (-2.005 - 0.01 / 2, 0 - 4 / 2, 0.01, 4),
    (2.005 - 0.01 / 2, 0 - 4 / 2, 0.01, 4),
    (0 - 4 / 2, -2.005 - 0.01 / 2, 4, 0.01),
    (0 - 4 / 2, 2.005 - 0.01 / 2, 4, 0.01),
    (-1.05 - 0.2 / 2, -1.2 - 0.1 / 2, 0.2, 0.1),
    (1.05 - 0.3 / 2, 1.25 - 0.1 / 2, 0.3, 0.1),
    (-0.5 - 0.3 / 2, 1.1 - 0.2 / 2, 0.3, 0.2),
    (0.1 - 0.2 / 2, -0.2 - 0.2 / 2, 0.2, 0.2),
    (1.1 - 0.2 / 2, -1.05 - 0.2 / 2, 0.2, 0.2),
    (-1.7 - 0.3 / 2, -0.15 - 0.3 / 2, 0.3, 0.3)

    #6*6
    # (-3.005 - 0.01 / 2, 0 - 6 / 2, 0.01, 6),
    # (3.005 - 0.01 / 2, 0 - 6 / 2, 0.01, 6),
    # (0 - 6 / 2, -3.005 - 0.01 / 2, 6, 0.01),
    # (0 - 6 / 2, 3.005 - 0.01 / 2, 6, 0.01),
    # (2.05 - 0.4 / 2, 1.85 - 0.1 / 2, 0.4, 0.1),
    # (-1.05 - 0.3 / 2, -2.1 - 0.8 / 2, 0.3, 0.8),
    # (-0.35 - 0.2 / 2, -0.7 - 0.2 / 2, 0.2, 0.2),
    # (1.1 - 0.2 / 2, -2.05 - 0.2 / 2, 0.2, 0.2),
    # (-2.7 - 0.3 / 2, -0.15 - 0.3 / 2, 0.3, 0.3),
    # (0.8 - 0.1 / 2, 0.8 - 0.1 / 2, 0.1, 0.1),
    # (1.35 - 0.3 / 2, -0.45 - 0.1 / 2, 0.3, 0.1),
    # (-2.25 - 0.5 / 2, 1.35 - 0.3 / 2, 0.5, 0.3),
    # (-0.9 - 0.6 / 2, 2.25 - 0.2 / 2, 0.6, 0.2)
]

def point_in_obstacle(x, y):
    """
    判断点 (x, y) 是否在任意一个障碍物内。
    返回 True 表示在障碍物内，False 表示不在。
    """
    for ox, oy, w, h in obstacles:
        if ox <= x <= ox + w and oy <= y <= oy + h:
            return True
    return False

def is_spawn_position_valid(x, y, bounds=None, clearance=DEFAULT_CLEARANCE):
    if bounds:
        if not (bounds['x_min'] + clearance <= x <= bounds['x_max'] - clearance and
                bounds['y_min'] + clearance <= y <= bounds['y_max'] - clearance):
            return False

    if point_in_obstacle(x, y):
        return False

    for ox, oy, w, h in obstacles:
        expanded_left = ox - clearance
        expanded_right = ox + w + clearance
        expanded_bottom = oy - clearance
        expanded_top = oy + h + clearance

        if expanded_left <= x <= expanded_right and expanded_bottom <= y <= expanded_top:
            nearest_x = min(max(x, ox), ox + w)
            nearest_y = min(max(y, oy), oy + h)
            dx = x - nearest_x
            dy = y - nearest_y
            if dx * dx + dy * dy <= clearance * clearance:
                return False

    return True

def generate_random_positions(map_bounds, min_distance):
    max_attempts = 3000
    for _ in range(max_attempts):
        start_x = round(np.random.uniform(map_bounds['x_min'], map_bounds['x_max']), 2)
        start_y = round(np.random.uniform(map_bounds['y_min'], map_bounds['y_max']), 2)
        if not is_spawn_position_valid(start_x, start_y, bounds=map_bounds):
            continue

        goal_x = round(np.random.uniform(map_bounds['x_min'], map_bounds['x_max']), 2)
        goal_y = round(np.random.uniform(map_bounds['y_min'], map_bounds['y_max']), 2)
        if not is_spawn_position_valid(goal_x, goal_y, bounds=map_bounds):
            continue

        distance = np.sqrt((goal_x - start_x)**2 + (goal_y - start_y)**2)
        if distance >= min_distance:
            return [start_x, start_y], [goal_x, goal_y]
    # 如果无法生成有效位置，使用默认值
    return [0.0, 0.0], [2.0, 2.0]

def main():
    pairs = []
    for _ in range(NUM_PAIRS):
        start, goal = generate_random_positions(MAP_BOUNDS, MIN_DISTANCE)
        pairs.append({'start': start, 'goal': goal})
    # 保存为JSON文件
    with open('positions_6000.json', 'w') as f:
        json.dump(pairs, f, indent=2)
    print(f"已生成{NUM_PAIRS}对起终点，保存至positions_6000.json")

if __name__ == "__main__":
    main()

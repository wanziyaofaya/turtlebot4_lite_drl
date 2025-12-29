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
    # (-0.9 - 0.4 / 2, 2.25 - 0.2 / 2, 0.4, 0.2)
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


DEFAULT_CLEARANCE = 0.4
def is_spawn_position_valid(x, y, bounds=None, clearance=DEFAULT_CLEARANCE):
    """
    判断给定位置在生成起点/终点时是否合理：
    1. 不在障碍物矩形内
    2. 距离障碍物边界至少为 clearance
    3. 可选：在地图边界范围内保留 clearance 缓冲
    """
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

        if expanded_left < x < expanded_right and expanded_bottom < y < expanded_top:
            nearest_x = min(max(x, ox), ox + w)
            nearest_y = min(max(y, oy), oy + h)
            dx = x - nearest_x
            dy = y - nearest_y
            if dx * dx + dy * dy <= clearance * clearance:
                return False

    return True

CLEARANCE = 0.20
def is_position_valid(x, y, bounds=None, clearance=CLEARANCE):
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

        if expanded_left < x < expanded_right and expanded_bottom < y < expanded_top:
            nearest_x = min(max(x, ox), ox + w)
            nearest_y = min(max(y, oy), oy + h)
            dx = x - nearest_x
            dy = y - nearest_y
            if dx * dx + dy * dy <= clearance * clearance:
                return False
    return True
import numpy as np
from typing import Optional

class RuleBasedSubgoalGenerator:
    """Rule-based subgoal generator.

    Notes:
    - This generator is intended to consume the environment's *raw* LiDAR ranges
      (e.g. 640 beams in meters), such as `TurtleBotNavEnv.raw_data`.
    """

    def __init__(
        self,
        lidar_size: int = 640,
        max_range: float = 9.0,
        debug: bool = False,
        *,
        gap_diff_threshold: float = 0.25,
        max_range_run_min_len: int = 35,
        max_range_sample_step: int = 10,
        free_space_candidate_dist: float = 0.66,
        gap_candidate_scale: float = 0.90,
        safety_margin: float = 0.10,
        max_range_epsilon: float = 1e-3,
        enable_segment_check: bool = True,
        segment_check_step: float = 0.2,
    ):
        self.lidar_size = int(lidar_size)
        self.max_range = float(max_range)
        self.debug = bool(debug)
        self.gap_diff_threshold = float(gap_diff_threshold)
        self.max_range_run_min_len = int(max_range_run_min_len)
        self.max_range_sample_step = max(1, int(max_range_sample_step))
        self.free_space_candidate_dist = float(free_space_candidate_dist)
        self.gap_candidate_scale = float(gap_candidate_scale)
        self.safety_margin = float(safety_margin)
        self.max_range_epsilon = float(max_range_epsilon)
        self.enable_segment_check = bool(enable_segment_check)
        self.segment_check_step = float(segment_check_step)

    def _is_segment_collision_free(self, x0: float, y0: float, x1: float, y1: float, *, bounds: Optional[dict]) -> bool:
        """Return True if the straight segment from (x0,y0) to (x1,y1) stays valid.

        Implementation: sample points along the segment and require every sample to be
        `is_position_valid`. This prevents choosing a subgoal whose endpoint is valid
        but the line-of-sight crosses an obstacle.
        """
        from turtlebot4_rl.collision import is_position_valid

        step = float(self.segment_check_step)
        if step <= 0.0:
            step = 0.05

        dx = float(x1) - float(x0)
        dy = float(y1) - float(y0)
        dist = float(np.hypot(dx, dy))
        if dist <= 1e-9:
            return bool(is_position_valid(float(x1), float(y1), bounds=bounds))

        # Include both ends; skip the very first sample (robot pose) since it's assumed valid.
        n = int(np.ceil(dist / step)) + 1
        n = max(n, 2)
        ts = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
        for t in ts[1:]:
            x = float(x0 + dx * float(t))
            y = float(y0 + dy * float(t))
            if not bool(is_position_valid(x, y, bounds=bounds)):
                return False
        return True

    def _beam_deg(self, beam_index: float) -> float:
        # Map beam index -> angle in degrees.
        # Your LiDAR FOV is [-180, 180]. For N beams, assume they span the full range.
        # Using (N-1) makes the first beam exactly -180 and the last exactly +180.
        n = max(2, int(self.lidar_size))
        return -180.0 + float(beam_index) * (360.0 / float(n - 1))

    def _beam_rad_from_scan(self, beam_index: float, *, scan_angle_min: float, scan_angle_increment: float) -> float:
        """Map beam index -> angle in radians using LaserScan metadata."""
        return float(scan_angle_min) + float(beam_index) * float(scan_angle_increment)

    def _append_free_space_candidates(self, nodes, start_idx: int, end_idx: int, *, odomX, odomY, angle):
        # Place a candidate every N beams inside a max-range run.
        if end_idx < start_idx:
            return
        run_len = int(end_idx - start_idx + 1)
        if run_len < self.max_range_run_min_len:
            return
        step = int(self.max_range_sample_step)
        # Sample around the center of each step-sized chunk.
        half = (step - 1) / 2.0
        for k in range(start_idx, end_idx + 1, step):
            beam = min(float(end_idx), float(k) + half)
            beam_angl = self._beam_deg(beam)
            dist = float(min(self.free_space_candidate_dist, self.max_range))
            qx = dist * np.cos(np.radians(beam_angl + angle))
            qy = dist * np.sin(np.radians(beam_angl + angle))
            nodes.append([qx + odomX, qy + odomY])

    def _has_consecutive_clear_beams(self, lidar: np.ndarray, *, threshold: float, min_run_len: int) -> bool:
        """Return True if there exists a consecutive run of beams with ranges > threshold.

        Args:
            lidar: 1D array of ranges (meters), length == self.lidar_size.
            threshold: meters.
            min_run_len: minimum consecutive count.
        """
        run = 0
        for d in lidar:
            if float(d) > float(threshold):
                run += 1
                if run >= int(min_run_len):
                    return True
            else:
                run = 0
        return False

    def _qualifying_clear_beam_indices(self, lidar: np.ndarray, *, threshold: float, min_run_len: int) -> np.ndarray:
        """Return indices of beams that belong to any qualifying consecutive clear run.

        A qualifying run is a consecutive segment where lidar[i] > threshold and
        run length >= min_run_len.
        """
        lidar = np.asarray(lidar, dtype=np.float32)
        indices = []
        run_start = None
        run_len = 0
        for i in range(lidar.shape[0]):
            if float(lidar[i]) > float(threshold):
                if run_start is None:
                    run_start = i
                    run_len = 1
                else:
                    run_len += 1
            else:
                if run_start is not None and run_len >= int(min_run_len):
                    indices.extend(range(run_start, run_start + run_len))
                run_start = None
                run_len = 0

        # tail segment
        if run_start is not None and run_len >= int(min_run_len):
            indices.extend(range(run_start, run_start + run_len))

        if not indices:
            return np.asarray([], dtype=np.int32)
        return np.asarray(indices, dtype=np.int32)
    
    def get_subgoal(
        self,
        lidar_data,
        odomX,
        odomY,
        angle,
        dist_s,
        dist_g,
        goalX,
        goalY,
        *,
        bounds: Optional[dict] = None,
        scan_angle_min: Optional[float] = None,
        scan_angle_increment: Optional[float] = None,
        scan_yaw_offset_rad: float = 0.0,
    ) -> Optional[np.ndarray]:
        """
        输入: lidar_data (np.ndarray), 机器人位置和朝向
        输出: subgoal (np.ndarray) - 生成的子目标点坐标 (x, y)

        规则（按你的需求实现）：
        - 仅判断传入的 640 维激光值
                - 如果存在一段“连续 >20 束”的激光值都 > 3.0m，则：
                    仅沿这些满足条件的激光束方向，在 3.0m 处放置候选点
        - 候选点评分 = 候选点与全局目标的欧氏距离；选取最小者
        - 若不满足连续区间条件，返回 None
        """
        if lidar_data is None:
            return None
        lidar = np.asarray(lidar_data, dtype=np.float32)
        if lidar.size != int(self.lidar_size):
            return None
        # Sanitize NaN/Inf and clamp to max range.
        lidar = np.nan_to_num(lidar, nan=self.max_range, posinf=self.max_range, neginf=0.0)
        lidar = np.clip(lidar, 0.0, self.max_range)

        # Condition: exists a consecutive run longer than the given beam count with range > 3.0m.
        # “连续大于20束” -> min_run_len = 21.
        threshold = 3.0
        min_run_len = 21
        qualifying_idx = self._qualifying_clear_beam_indices(lidar, threshold=threshold, min_run_len=min_run_len)
        if qualifying_idx.size == 0:
            return None

        # Generate candidates only along those qualifying beam directions, at exactly 3.0m.
        # Prefer LaserScan angle_min/angle_increment (radians) when provided; otherwise
        # fall back to assuming a uniform [-180, 180] deg mapping.
        candidate_dist = float(min(threshold, self.max_range))
        # `angle` is expected to be robot yaw in radians (world frame)
        yaw_rad = float(angle)
        scan_yaw_offset_rad = float(scan_yaw_offset_rad)
        nodes = np.empty((int(qualifying_idx.size), 2), dtype=np.float32)
        for j, i in enumerate(qualifying_idx.tolist()):
            if scan_angle_min is not None and scan_angle_increment is not None:
                beam_rad = self._beam_rad_from_scan(
                    float(i),
                    scan_angle_min=float(scan_angle_min),
                    scan_angle_increment=float(scan_angle_increment),
                )
            else:
                beam_rad = float(np.radians(self._beam_deg(float(i))))

            heading_rad = float(beam_rad + yaw_rad + scan_yaw_offset_rad)
            qx = candidate_dist * np.cos(heading_rad)
            qy = candidate_dist * np.sin(heading_rad)
            nodes[j, 0] = float(qx + float(odomX))
            nodes[j, 1] = float(qy + float(odomY))

        # Filter invalid candidates: invalid points do not participate in scoring.
        from turtlebot4_rl.collision import is_position_valid

        candidate_indices = []
        for i in range(nodes.shape[0]):
            x_i = float(nodes[i, 0])
            y_i = float(nodes[i, 1])
            if bool(is_position_valid(x_i, y_i, bounds=bounds)):
                candidate_indices.append(i)

        if not candidate_indices:
            return None

        # Score = distance to global goal; evaluate in increasing score order.
        # This allows us to do expensive segment checks only for the best candidates.
        goalX_f = float(goalX)
        goalY_f = float(goalY)
        cand = nodes[np.asarray(candidate_indices, dtype=np.int32)]
        dx = goalX_f - cand[:, 0].astype(np.float64)
        dy = goalY_f - cand[:, 1].astype(np.float64)
        scores = np.hypot(dx, dy)
        order = np.argsort(scores)

        odomX_f = float(odomX)
        odomY_f = float(odomY)
        for k in order.tolist():
            x_sg = float(cand[k, 0])
            y_sg = float(cand[k, 1])
            if self.enable_segment_check:
                if not self._is_segment_collision_free(odomX_f, odomY_f, x_sg, y_sg, bounds=bounds):
                    continue
            return np.asarray([x_sg, y_sg], dtype=np.float32)

        return None

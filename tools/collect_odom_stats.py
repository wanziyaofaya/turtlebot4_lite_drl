#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import time
import argparse
import csv
import os

class OdomCollector(Node):
    def __init__(self, n_samples=1000, out_file=None):
        super().__init__('odom_collector')
        self.sub = self.create_subscription(Odometry, '/odom', self.cb, 10)
        self.n = n_samples
        self.out_file = out_file

        self.times = []      # stamps in seconds (float)
        self.positions = []  # list of (x,y)

        self.count = 0

        # Prepare CSV if needed
        if self.out_file:
            # ensure directory exists
            os.makedirs(os.path.dirname(self.out_file) or '.', exist_ok=True)
            self._csv_fd = open(self.out_file, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_fd)
            self._csv_writer.writerow(['idx', 'stamp', 'x', 'y', 'dt', 'jump'])
        else:
            self._csv_fd = None
            self._csv_writer = None

    def cb(self, msg):
        # Try to read ROS2 header stamp; fall back to time.time() if not present
        try:
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception:
            stamp = time.time()

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)

        # Append
        self.times.append(stamp)
        self.positions.append((x, y))

        # Compute dt and jump relative to previous if exists
        dt = ''
        jump = ''
        if len(self.times) > 1:
            dt_val = self.times[-1] - self.times[-2]
            dt = float(dt_val)
            prev = np.array(self.positions[-2], dtype=np.float32)
            cur = np.array(self.positions[-1], dtype=np.float32)
            jump_val = float(np.linalg.norm(cur - prev))
            jump = jump_val
        
        # Write CSV row
        if self._csv_writer is not None:
            self._csv_writer.writerow([self.count, stamp, x, y, dt, jump])

        self.count += 1
        if self.count >= self.n:
            # Finish and print stats
            self.print_stats()
            rclpy.shutdown()

    def print_stats(self):
        times = np.array(self.times)
        pos = np.array(self.positions)
        if len(times) < 2:
            print('Not enough samples to compute stats.')
            return
        dt = np.diff(times)
        jumps = np.linalg.norm(np.diff(pos, axis=0), axis=1)

        def stats(arr):
            return {
                'count': int(len(arr)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'p50': float(np.percentile(arr, 50)),
                'p90': float(np.percentile(arr, 90)),
                'p99': float(np.percentile(arr, 99)),
                'max': float(np.max(arr))
            }

        dt_stats = stats(dt)
        jump_stats = stats(jumps)

        print('\n=== ODOM STATS ===')
        print('samples (dt):', dt_stats['count'])
        print('dt mean: {:.6f}s, std: {:.6f}s, p50: {:.6f}s, p90: {:.6f}s, p99: {:.6f}s, max: {:.6f}s'.format(
            dt_stats['mean'], dt_stats['std'], dt_stats['p50'], dt_stats['p90'], dt_stats['p99'], dt_stats['max']))
        print('\nJump stats (m):')
        print('count: {}, mean: {:.6f} m, std: {:.6f} m, p50: {:.6f} m, p90: {:.6f} m, p99: {:.6f} m, max: {:.6f} m'.format(
            jump_stats['count'], jump_stats['mean'], jump_stats['std'], jump_stats['p50'], jump_stats['p90'], jump_stats['p99'], jump_stats['max']))

        if self._csv_fd is not None:
            self._csv_fd.flush()
            print(f'CSV saved to: {self.out_file}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1000, help='Number of odom samples to collect')
    parser.add_argument('--out', type=str, default='odom_stats.csv', help='CSV output file (optional)')
    args = parser.parse_args()

    rclpy.init()
    node = OdomCollector(n_samples=args.samples, out_file=args.out)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Interrupted by user')
        node.print_stats()
    finally:
        if node._csv_fd is not None:
            node._csv_fd.close()
        try:
            node.destroy_node()
        except Exception:
            # Node may already be destroyed or rclpy shutdown called in callback
            pass
        # Only shutdown if rclpy context is still initialized
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
# python3 tools/collect_odom_stats.py --samples 10000 --out odom_stats.csv
"""Utilities for parsing obstacle definitions from the Gazebo maze SDF."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import xml.etree.ElementTree as ET


def _parse_float_sequence(raw_values: Iterable[str]) -> List[float]:
	return [float(value) for value in raw_values]


def _parse_pose(pose_text: str | None) -> List[float]:
	if not pose_text:
		return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	return _parse_float_sequence(pose_text.split())


def _parse_box_size(box_element: ET.Element | None) -> List[float]:
	if box_element is None:
		raise ValueError("Box geometry missing <box> element.")
	size_text = box_element.findtext("size")
	if size_text is None:
		raise ValueError("Box geometry missing <size> element.")
	return _parse_float_sequence(size_text.split())


def extract_obstacles_from_sdf(sdf_path: Path) -> List[Dict[str, Any]]:
	tree = ET.parse(sdf_path)
	root = tree.getroot()

	obstacles_model = root.find(".//model[@name='obstacles']")
	if obstacles_model is None:
		raise ValueError("No <model name='obstacles'> section found in SDF file.")

	obstacles: List[Dict[str, Any]] = []
	for link in obstacles_model.findall("link"):
		name = link.get("name", "")
		pose = _parse_pose(link.findtext("pose"))

		collision = link.find("collision")
		if collision is None:
			continue
		box_element = collision.find("./geometry/box")
		try:
			size = _parse_box_size(box_element)
		except ValueError:
			continue

		obstacles.append({"name": name, "pose": pose, "size": size})

	return obstacles


def convert_to_rectangles(obstacles: Sequence[Dict[str, Any]]) -> List[Tuple[float, float, float, float]]:
	rectangles: List[Tuple[float, float, float, float]] = []
	for obstacle in obstacles:
		pose = obstacle.get("pose", [0.0, 0.0])
		size = obstacle.get("size", [0.0, 0.0])
		if len(pose) < 2 or len(size) < 2:
			continue
		ll_x = pose[0] - size[0] / 2.0
		ll_y = pose[1] - size[1] / 2.0
		rectangles.append((ll_x, ll_y, size[0], size[1]))
	return rectangles


def main() -> None:
	sdf_file = Path(__file__).resolve().parents[2] / "turtlebot4_gz_bringup" / "worlds" / "maze.sdf"
	obstacles = extract_obstacles_from_sdf(sdf_file)
	rectangles = convert_to_rectangles(obstacles)
	for rect in rectangles:
		print(rect)


if __name__ == "__main__":
	main()

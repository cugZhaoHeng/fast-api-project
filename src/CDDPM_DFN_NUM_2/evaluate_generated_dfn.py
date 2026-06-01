import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


DEFAULT_TILE_SIZE = 128
DEFAULT_HOUGH_THRESHOLD = 15
DEFAULT_MIN_LINE_LENGTH = 10
DEFAULT_MAX_LINE_GAP = 4
DEFAULT_LINE_DISTANCE_TOL = 8.0
DEFAULT_CENTER_DISTANCE_TOL = 28.0
DEFAULT_ANGLE_TOL = 8.0


def normalize_orientation_deg(angle_deg: float) -> float:
    while angle_deg > 90.0:
        angle_deg -= 180.0
    while angle_deg <= -90.0:
        angle_deg += 180.0
    return angle_deg


def axial_angle_diff_deg(angle_a: float, angle_b: float) -> float:
    diff = abs(angle_a - angle_b) % 180.0
    return min(diff, 180.0 - diff)


def weighted_axial_mean_deg(
    angles_deg: List[float], weights: Optional[List[float]] = None
) -> Optional[float]:
    if not angles_deg:
        return None

    angles = np.asarray(angles_deg, dtype=np.float64)
    if weights is None:
        weights_arr = np.ones_like(angles)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64)

    radians = np.deg2rad(angles)
    sin_sum = np.sum(weights_arr * np.sin(2.0 * radians))
    cos_sum = np.sum(weights_arr * np.cos(2.0 * radians))

    if np.isclose(sin_sum, 0.0) and np.isclose(cos_sum, 0.0):
        return None

    mean_angle = 0.5 * np.rad2deg(np.arctan2(sin_sum, cos_sum))
    return normalize_orientation_deg(float(mean_angle))


def axial_angle_error_deg(predicted_deg: float, target_deg: float) -> float:
    return axial_angle_diff_deg(predicted_deg, target_deg)


def point_line_distance(
    point_xy: np.ndarray, line_center_xy: np.ndarray, angle_deg: float
) -> float:
    angle_rad = math.radians(angle_deg)
    direction = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float64)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    return abs(float(np.dot(point_xy - line_center_xy, normal)))


def extract_line_segments(
    image_gray: np.ndarray,
    threshold: int = 127,
    hough_threshold: int = DEFAULT_HOUGH_THRESHOLD,
    min_line_length: int = DEFAULT_MIN_LINE_LENGTH,
    max_line_gap: int = DEFAULT_MAX_LINE_GAP,
) -> List[Dict]:
    _, binary = cv2.threshold(image_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    segments: List[Dict] = []
    if lines is None:
        return segments

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in line]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1.0:
            continue

        angle_deg = normalize_orientation_deg(math.degrees(math.atan2(dy, dx)))
        center_xy = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)
        segments.append(
            {
                "endpoints": (x1, y1, x2, y2),
                "length": float(length),
                "angle_deg": float(angle_deg),
                "center_xy": center_xy,
            }
        )

    return segments


def update_group_with_segment(group: Dict, segment: Dict) -> None:
    segment_angle_rad = math.radians(segment["angle_deg"])
    weight = segment["length"]
    total_weight = group["total_weight"] + weight

    group["sum_sin2"] += weight * math.sin(2.0 * segment_angle_rad)
    group["sum_cos2"] += weight * math.cos(2.0 * segment_angle_rad)
    group["center_xy"] = (
        group["center_xy"] * group["total_weight"] + segment["center_xy"] * weight
    ) / total_weight
    group["total_weight"] = total_weight
    group["segment_count"] += 1
    group["max_length"] = max(group["max_length"], segment["length"])
    group["angle_deg"] = normalize_orientation_deg(
        0.5 * math.degrees(math.atan2(group["sum_sin2"], group["sum_cos2"]))
    )


def merge_line_segments(
    segments: List[Dict],
    angle_tol_deg: float = DEFAULT_ANGLE_TOL,
    line_distance_tol: float = DEFAULT_LINE_DISTANCE_TOL,
    center_distance_tol: float = DEFAULT_CENTER_DISTANCE_TOL,
) -> List[Dict]:
    groups: List[Dict] = []

    for segment in sorted(segments, key=lambda item: item["length"], reverse=True):
        best_group = None
        best_score = None

        for group in groups:
            angle_diff = axial_angle_diff_deg(segment["angle_deg"], group["angle_deg"])
            if angle_diff > angle_tol_deg:
                continue

            line_dist = point_line_distance(
                point_xy=segment["center_xy"],
                line_center_xy=group["center_xy"],
                angle_deg=group["angle_deg"],
            )
            if line_dist > line_distance_tol:
                continue

            center_dist = float(np.linalg.norm(segment["center_xy"] - group["center_xy"]))
            allowed_center_dist = center_distance_tol + 0.5 * (
                group["max_length"] + segment["length"]
            )
            if center_dist > allowed_center_dist:
                continue

            score = angle_diff + 0.1 * line_dist + 0.01 * center_dist
            if best_score is None or score < best_score:
                best_score = score
                best_group = group

        if best_group is None:
            segment_angle_rad = math.radians(segment["angle_deg"])
            groups.append(
                {
                    "center_xy": segment["center_xy"].copy(),
                    "angle_deg": segment["angle_deg"],
                    "sum_sin2": segment["length"] * math.sin(2.0 * segment_angle_rad),
                    "sum_cos2": segment["length"] * math.cos(2.0 * segment_angle_rad),
                    "total_weight": segment["length"],
                    "segment_count": 1,
                    "max_length": segment["length"],
                }
            )
        else:
            update_group_with_segment(best_group, segment)

    return groups


def evaluate_tile(image_gray: np.ndarray) -> Dict:
    segments = extract_line_segments(image_gray)
    groups = merge_line_segments(segments)

    dominant_mu_deg = weighted_axial_mean_deg(
        [group["angle_deg"] for group in groups],
        [group["total_weight"] for group in groups],
    )

    return {
        "estimated_count": len(groups),
        "dominant_mu_deg": dominant_mu_deg,
        "raw_segment_count": len(segments),
        "group_orientations_deg": [round(group["angle_deg"], 4) for group in groups],
        "group_weights": [round(group["total_weight"], 4) for group in groups],
    }


def split_grid_image(
    image_gray: np.ndarray, tile_size: int = DEFAULT_TILE_SIZE, grid_size: Optional[int] = None
) -> List[np.ndarray]:
    height, width = image_gray.shape[:2]

    if grid_size is None:
        if height % tile_size != 0 or width % tile_size != 0:
            raise ValueError(
                f"Cannot infer grid size from image of shape {image_gray.shape} and tile_size={tile_size}"
            )
        grid_rows = height // tile_size
        grid_cols = width // tile_size
        if grid_rows != grid_cols:
            raise ValueError(
                f"Expected a square grid, but inferred rows={grid_rows}, cols={grid_cols}"
            )
        grid_size = grid_rows

    expected_height = grid_size * tile_size
    expected_width = grid_size * tile_size
    if height != expected_height or width != expected_width:
        raise ValueError(
            f"Image shape {image_gray.shape} does not match grid_size={grid_size} and tile_size={tile_size}"
        )

    tiles: List[np.ndarray] = []
    for row in range(grid_size):
        for col in range(grid_size):
            row_start = row * tile_size
            col_start = col * tile_size
            tiles.append(
                image_gray[
                    row_start : row_start + tile_size,
                    col_start : col_start + tile_size,
                ]
            )

    return tiles


def evaluate_generated_grid(
    image_path: Path,
    tile_size: int = DEFAULT_TILE_SIZE,
    grid_size: Optional[int] = None,
    target_count: Optional[int] = None,
    target_mu: Optional[float] = None,
    target_kappa: Optional[float] = None,
    save_report: bool = True,
) -> Dict:
    image_path = Path(image_path)
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    tiles = split_grid_image(image_gray, tile_size=tile_size, grid_size=grid_size)

    tile_reports: List[Dict] = []
    all_group_angles: List[float] = []
    all_group_weights: List[float] = []

    for idx, tile in enumerate(tiles):
        tile_report = evaluate_tile(tile)
        tile_report["tile_index"] = idx
        tile_reports.append(tile_report)

        for angle_deg, weight in zip(
            tile_report["group_orientations_deg"], tile_report["group_weights"]
        ):
            all_group_angles.append(angle_deg)
            all_group_weights.append(weight)

    estimated_counts = [tile_report["estimated_count"] for tile_report in tile_reports]
    dominant_mus = [
        tile_report["dominant_mu_deg"]
        for tile_report in tile_reports
        if tile_report["dominant_mu_deg"] is not None
    ]

    overall_dominant_mu_deg = weighted_axial_mean_deg(
        all_group_angles, all_group_weights
    )
    mean_estimated_count = float(np.mean(estimated_counts)) if estimated_counts else 0.0

    summary = {
        "num_tiles": len(tile_reports),
        "mean_estimated_count": mean_estimated_count,
        "std_estimated_count": float(np.std(estimated_counts)) if estimated_counts else 0.0,
        "overall_dominant_mu_deg": overall_dominant_mu_deg,
        "mean_tile_dominant_mu_deg": (
            weighted_axial_mean_deg(dominant_mus) if dominant_mus else None
        ),
        "count_error": (
            mean_estimated_count - target_count if target_count is not None else None
        ),
        "mu_error_deg": (
            axial_angle_error_deg(overall_dominant_mu_deg, target_mu)
            if target_mu is not None and overall_dominant_mu_deg is not None
            else None
        ),
    }

    report = {
        "image_path": str(image_path.resolve()),
        "grid_size": grid_size if grid_size is not None else int(math.sqrt(len(tiles))),
        "tile_size": tile_size,
        "targets": {
            "fracture_count": target_count,
            "mean_mu_deg": target_mu,
            "kappa": target_kappa,
        },
        "summary": summary,
        "tiles": tile_reports,
    }

    report_path = image_path.with_name(f"{image_path.stem}_eval.json")
    if save_report:
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    report["report_path"] = str(report_path)

    return report


def print_report(report: Dict) -> None:
    summary = report["summary"]
    targets = report["targets"]

    print("=" * 56)
    print(f"Image: {report['image_path']}")
    print(
        "Targets -> "
        f"count: {targets['fracture_count']}, "
        f"mean_mu: {targets['mean_mu_deg']}, "
        f"kappa: {targets['kappa']}"
    )
    print(
        "Summary -> "
        f"mean_estimated_count: {summary['mean_estimated_count']:.2f}, "
        f"std_estimated_count: {summary['std_estimated_count']:.2f}, "
        f"overall_dominant_mu_deg: {summary['overall_dominant_mu_deg']}"
    )
    if summary["count_error"] is not None:
        print(f"Count error: {summary['count_error']:+.2f}")
    if summary["mu_error_deg"] is not None:
        print(f"Mu error (axial): {summary['mu_error_deg']:.2f} deg")
    print(f"Report saved to: {report['report_path']}")
    print("=" * 56)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate generated DFN images for estimated fracture count and dominant orientation."
    )
    parser.add_argument("--image", required=True, help="Path to the raw generated grid image.")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--target-mu", type=float, default=None)
    parser.add_argument("--target-kappa", type=float, default=None)
    parser.add_argument("--no-save-report", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    report = evaluate_generated_grid(
        image_path=Path(args.image),
        tile_size=args.tile_size,
        grid_size=args.grid_size,
        target_count=args.target_count,
        target_mu=args.target_mu,
        target_kappa=args.target_kappa,
        save_report=not args.no_save_report,
    )
    print_report(report)


if __name__ == "__main__":
    main()

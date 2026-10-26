"""Instance-level Panoptic Quality metric helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PQCounts:
    """Additive counts used to compute dataset-level panoptic quality."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    matched_iou: float = 0.0

    @property
    def pq(self) -> float:
        denominator = self.true_positives + 0.5 * (
            self.false_positives + self.false_negatives
        )
        return self.matched_iou / denominator if denominator else 1.0

    def add(self, other: "PQCounts") -> None:
        self.true_positives += other.true_positives
        self.false_positives += other.false_positives
        self.false_negatives += other.false_negatives
        self.matched_iou += other.matched_iou


def panoptic_quality(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    iou_threshold: float = 0.5,
) -> PQCounts:
    """Match predicted instances to targets using the PQ IoU threshold."""
    if ground_truth.shape != prediction.shape:
        raise ValueError("Ground-truth and prediction masks must have equal shapes")
    gt = np.asarray(ground_truth, dtype=np.int64)
    pred = np.asarray(prediction, dtype=np.int64)
    gt_ids, gt_inverse = np.unique(gt[gt > 0], return_inverse=True)
    pred_ids, pred_inverse = np.unique(pred[pred > 0], return_inverse=True)
    gt_area = np.bincount(gt_inverse, minlength=len(gt_ids))
    pred_area = np.bincount(pred_inverse, minlength=len(pred_ids))
    gt_index = {int(instance_id): index for index, instance_id in enumerate(gt_ids)}
    pred_index = {int(instance_id): index for index, instance_id in enumerate(pred_ids)}
    gt_flat = gt.ravel()
    pred_flat = pred.ravel()
    overlap_pixels = (gt_flat > 0) & (pred_flat > 0)
    pairs = np.stack((gt_flat[overlap_pixels], pred_flat[overlap_pixels]), axis=1)
    if pairs.size:
        unique_pairs, intersections = np.unique(pairs, axis=0, return_counts=True)
        candidates = []
        for (gt_id, pred_id), intersection in zip(unique_pairs, intersections):
            gt_position = gt_index[int(gt_id)]
            pred_position = pred_index[int(pred_id)]
            union = gt_area[gt_position] + pred_area[pred_position] - intersection
            iou = float(intersection / union)
            if iou > iou_threshold:
                candidates.append((iou, int(gt_id), int(pred_id)))
    else:
        candidates = []
    candidates.sort(reverse=True)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matched_iou = 0.0
    for iou, gt_id, pred_id in candidates:
        if gt_id not in matched_gt and pred_id not in matched_pred:
            matched_gt.add(gt_id)
            matched_pred.add(pred_id)
            matched_iou += iou
    true_positives = len(matched_gt)
    return PQCounts(
        true_positives=true_positives,
        false_positives=len(pred_ids) - true_positives,
        false_negatives=len(gt_ids) - true_positives,
        matched_iou=matched_iou,
    )

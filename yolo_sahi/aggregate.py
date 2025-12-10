from typing import List

import torch

from postprocess import clip_boxes


def aggregate_results(results: List[torch.Tensor]):
    """Aggregate the results of the different patches."""
    assert len(results) == 8, "There should be 8 results."

    results[0] = clip_boxes(results[0], (640, 640))

    results[1] = clip_boxes(results[1], (640, 640))
    results[1][:, :4] += torch.tensor([[0, 440, 0, 440]]).to(results[1].device)

    results[2] = clip_boxes(results[2], (640, 640))
    results[2][:, :4] += torch.tensor([[640, 0, 640, 0]]).to(results[2].device)

    results[3] = clip_boxes(results[3], (640, 640))
    results[3][:, :4] += torch.tensor([[640, 440, 640, 440]]).to(results[3].device)

    results[4] = clip_boxes(results[4], (640, 640))
    results[4][:, :4] += torch.tensor([[1280, 0, 1280, 0]]).to(results[4].device)

    results[5] = clip_boxes(results[5], (640, 640))
    results[5][:, :4] += torch.tensor([[1280, 440, 1280, 440]]).to(results[5].device)

    results[6] = clip_boxes(results[6], (640, 640))
    results[6][:, :4] += torch.tensor([[320, 220, 320, 220]]).to(results[6].device)

    results[7] = clip_boxes(results[7], (640, 640))
    results[7][:, :4] += torch.tensor([[960, 220, 960, 220]]).to(results[7].device)

    return torch.cat(results, dim=0)


def greedy_merge(boxes: torch.Tensor, nmm_ios: float, eps: float = 1e-9):
    """Merge the boxes using greedy algorithm."""
    assert boxes.shape[1] == 6, "Boxes should be a (n, 6) tensor."
    # step 1. sort the boxes by confidence
    if boxes.numel() == 0:
        return boxes

    conf = boxes[:, 4]
    sort_idx = conf.argsort(descending=True)
    boxes = boxes[sort_idx]

    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    areas = (end_x - start_x) * (end_y - start_y)

    n = len(boxes)
    keep_mask = torch.ones(n, dtype=torch.bool, device=boxes.device)
    merged_boxes = []

    for i in range(n):
        if not keep_mask[i]:
            continue

        # Compute IoU with all remaining boxes
        remaining = keep_mask[i + 1 :]
        if not remaining.any():
            # Last box, just add it
            merged_boxes.append(boxes[i])
            break

        remaining_idx = torch.arange(i + 1, n, device=boxes.device)[remaining]

        # Vectorized IoS computation
        x1 = torch.maximum(start_x[i], start_x[remaining_idx])
        y1 = torch.maximum(start_y[i], start_y[remaining_idx])
        x2 = torch.minimum(end_x[i], end_x[remaining_idx])
        y2 = torch.minimum(end_y[i], end_y[remaining_idx])
        w = (x2 - x1).clip(0.0)
        h = (y2 - y1).clip(0.0)
        intersection = w * h
        ios = intersection / (torch.minimum(areas[i], areas[remaining_idx]) + eps)

        # Find boxes to merge (including current box)
        merge_mask = ios > nmm_ios
        merge_indices = remaining_idx[merge_mask]

        # Merge boxes: min for x1,y1, max for x2,y2 (include current box i)
        if len(merge_indices) > 0:
            # Merge with overlapping boxes
            merged_box = boxes[i].clone()
            merged_box[0] = torch.minimum(merged_box[0], boxes[merge_indices, 0].min())
            merged_box[1] = torch.minimum(merged_box[1], boxes[merge_indices, 1].min())
            merged_box[2] = torch.maximum(merged_box[2], boxes[merge_indices, 2].max())
            merged_box[3] = torch.maximum(merged_box[3], boxes[merge_indices, 3].max())
        else:
            # No merge, just use current box
            merged_box = boxes[i]
        # Keep confidence and class from highest confidence box (already sorted)
        merged_boxes.append(merged_box)

        # Mark merged boxes as processed
        keep_mask[remaining_idx[merge_mask]] = False

    if merged_boxes:
        return torch.stack(merged_boxes, dim=0)
    return boxes[:0]  # Return empty tensor with correct shape


def non_max_merge(result: torch.Tensor, nmm_ios: float):
    """Merge the results of the different patches using non-maximum merging."""
    # result: (n, 6) tensor of [x1, y1, x2, y2, conf, cls]
    # nmm_ios: IoS threshold for NMM
    # return: (n, 6) tensor of [x1, y1, x2, y2, conf, cls]
    if result.numel() == 0:
        return result

    # step 1. split the result into different classes
    det_classes = result[:, 5].unique()

    if len(det_classes) == 1:
        # Single class, no need to split
        return greedy_merge(result, nmm_ios)

    merged_results = []
    # Use more efficient grouping by class
    for det_class in det_classes:
        class_mask = result[:, 5] == det_class
        det_class_result = result[class_mask]
        # step 2. merge boxes if iou > iou_nmm
        filtered_boxes = greedy_merge(det_class_result, nmm_ios)
        if len(filtered_boxes) > 0:
            merged_results.append(filtered_boxes)

    if merged_results:
        return torch.cat(merged_results, dim=0)
    return result[:0]  # Return empty tensor with correct shape

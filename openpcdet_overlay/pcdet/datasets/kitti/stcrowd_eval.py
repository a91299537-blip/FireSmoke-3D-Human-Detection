import io as sysio

import numpy as np


DEFAULT_DISTANCE_THRESHOLDS = (0.25, 0.5, 1.0)


def _print_str(value, *args, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *args, file=sstream)
    return sstream.getvalue()


def _as_numpy(array, shape=(-1,)):
    if array is None:
        return np.zeros(shape, dtype=np.float32)
    return np.asarray(array)


def _get_class_boxes(anno, class_name, box_key):
    names = _as_numpy(anno.get('name'), shape=(0,))
    boxes = _as_numpy(anno.get(box_key), shape=(0, 7)).reshape(-1, 7)

    # KITTI infos may keep DontCare in name fields while gt_boxes_lidar only
    # stores real objects. Keep the shared prefix to avoid indexing past boxes.
    valid_len = min(len(names), len(boxes))
    if valid_len == 0:
        return np.zeros((0, 7), dtype=np.float32)

    names = names[:valid_len]
    boxes = boxes[:valid_len]
    mask = names == class_name
    return boxes[mask].astype(np.float32, copy=False)


def _get_class_detections(dt_annos, class_name):
    detections = []
    for frame_idx, anno in enumerate(dt_annos):
        names = _as_numpy(anno.get('name'), shape=(0,))
        boxes = _as_numpy(anno.get('boxes_lidar'), shape=(0, 7)).reshape(-1, 7)
        scores = _as_numpy(anno.get('score'), shape=(0,)).reshape(-1)
        valid_len = min(len(names), len(boxes), len(scores))
        if valid_len == 0:
            continue

        names = names[:valid_len]
        boxes = boxes[:valid_len]
        scores = scores[:valid_len]
        for box, score in zip(boxes[names == class_name], scores[names == class_name]):
            detections.append((float(score), frame_idx, box[:3].astype(np.float32, copy=False)))

    detections.sort(key=lambda item: item[0], reverse=True)
    return detections


def _compute_ap(recall, precision):
    if recall.size == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    recall_steps = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[recall_steps + 1] - mrec[recall_steps]) * mpre[recall_steps + 1]))


def _evaluate_single_class(gt_annos, dt_annos, class_name, distance_thresholds):
    gt_centers = [
        _get_class_boxes(anno, class_name, 'gt_boxes_lidar')[:, :3]
        for anno in gt_annos
    ]
    num_gt = int(sum(len(centers) for centers in gt_centers))
    detections = _get_class_detections(dt_annos, class_name)

    results = {}
    for dist_thresh in distance_thresholds:
        if num_gt == 0:
            results[dist_thresh] = {
                'ap': 0.0,
                'recall': 0.0,
                'num_gt': 0,
                'num_dt': len(detections),
            }
            continue

        matched = [np.zeros(len(centers), dtype=bool) for centers in gt_centers]
        tp = np.zeros(len(detections), dtype=np.float32)
        fp = np.zeros(len(detections), dtype=np.float32)

        for det_idx, (_, frame_idx, det_center) in enumerate(detections):
            cur_gt_centers = gt_centers[frame_idx]
            cur_matched = matched[frame_idx]
            unmatched_inds = np.where(~cur_matched)[0]

            if unmatched_inds.size == 0:
                fp[det_idx] = 1.0
                continue

            distances = np.linalg.norm(cur_gt_centers[unmatched_inds] - det_center[None, :], axis=1)
            best_local_idx = int(np.argmin(distances))
            best_distance = distances[best_local_idx]
            if best_distance <= dist_thresh:
                gt_idx = unmatched_inds[best_local_idx]
                cur_matched[gt_idx] = True
                tp[det_idx] = 1.0
            else:
                fp[det_idx] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / max(num_gt, 1)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)
        ap = _compute_ap(recall, precision)

        results[dist_thresh] = {
            'ap': ap,
            'recall': float(recall[-1]) if recall.size > 0 else 0.0,
            'num_gt': num_gt,
            'num_dt': len(detections),
        }

    return results


def get_stcrowd_eval_result(gt_annos, dt_annos, class_names, distance_thresholds=DEFAULT_DISTANCE_THRESHOLDS):
    if not isinstance(class_names, (list, tuple)):
        class_names = [class_names]

    distance_thresholds = tuple(float(x) for x in distance_thresholds)
    result = ''
    ret_dict = {}
    class_maps = []

    result += _print_str('STCrowd 3D center-distance evaluation')
    result += _print_str('distance thresholds: ' + ', '.join(f'{x:.2f}m' for x in distance_thresholds))

    for class_name in class_names:
        class_result = _evaluate_single_class(gt_annos, dt_annos, class_name, distance_thresholds)
        aps = [class_result[x]['ap'] for x in distance_thresholds]
        class_map = float(np.mean(aps)) if len(aps) > 0 else 0.0
        class_maps.append(class_map)

        result += _print_str(f'{class_name}:')
        for dist_thresh in distance_thresholds:
            metrics = class_result[dist_thresh]
            ap_percent = metrics['ap'] * 100.0
            recall_percent = metrics['recall'] * 100.0
            result += _print_str(
                f'  AP@{dist_thresh:.2f}m: {ap_percent:.4f}, '
                f'recall: {recall_percent:.4f}, '
                f'gt: {metrics["num_gt"]}, dt: {metrics["num_dt"]}'
            )
            ret_dict[f'{class_name}_AP_{dist_thresh:.2f}m'] = ap_percent
            ret_dict[f'{class_name}_recall_{dist_thresh:.2f}m'] = recall_percent

        result += _print_str(f'  mAP: {class_map * 100.0:.4f}')
        ret_dict[f'{class_name}_mAP'] = class_map * 100.0

    mean_map = float(np.mean(class_maps)) if len(class_maps) > 0 else 0.0
    result += _print_str(f'Overall mAP: {mean_map * 100.0:.4f}')
    ret_dict['mAP'] = mean_map * 100.0

    return result, ret_dict

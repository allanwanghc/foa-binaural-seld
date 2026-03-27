"""
SELD evaluation metrics adapted from DCASE baseline.

Computes: Error Rate (ER), F-score (F1), Localization Error (LE),
Localization Recall (LR), and overall SELD score.

Uses frame-level evaluation with Hungarian algorithm for multi-track
association.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

eps = np.finfo(float).eps


def distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2):
    """
    Angular distance between two cartesian coordinates on unit sphere.

    Returns:
        Angular distance in degrees.
    """
    N1 = np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2 + 1e-10)
    N2 = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2 + 1e-10)
    x1, y1, z1 = x1 / N1, y1 / N1, z1 / N1
    x2, y2, z2 = x2 / N2, y2 / N2, z2 / N2

    dist = x1 * x2 + y1 * y2 + z1 * z2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def least_distance_between_gt_pred(gt_list, pred_list):
    """
    Find optimal assignment between GT and predicted DOAs using
    the Hungarian algorithm.

    Args:
        gt_list: (N_gt, 3) array of cartesian DOAs
        pred_list: (N_pred, 3) array of cartesian DOAs

    Returns:
        cost: array of angular distances for matched pairs
        row_ind: GT indices of matched pairs
        col_ind: Pred indices of matched pairs
    """
    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]

    if gt_len == 0 or pred_len == 0:
        return np.array([]), np.array([], dtype=int), np.array([], dtype=int)

    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))

    x1 = gt_list[ind_pairs[:, 0], 0]
    y1 = gt_list[ind_pairs[:, 0], 1]
    z1 = gt_list[ind_pairs[:, 0], 2]
    x2 = pred_list[ind_pairs[:, 1], 0]
    y2 = pred_list[ind_pairs[:, 1], 1]
    z2 = pred_list[ind_pairs[:, 1], 2]

    distances = distance_between_cartesian_coordinates(x1, y1, z1, x2, y2, z2)
    cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distances

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind]

    return cost, row_ind, col_ind


class SELDMetrics:
    """
    Compute DCASE SELD metrics: ER, F1, LE, LR, SELD_score.

    Uses frame-level evaluation with macro-averaging across classes.

    Args:
        nb_classes: number of sound event classes
        doa_threshold: DOA threshold in degrees for location-sensitive detection
    """

    def __init__(self, nb_classes=8, doa_threshold=20):
        self._nb_classes = nb_classes
        self._spatial_T = doa_threshold

        # Location-sensitive detection
        self._TP = np.zeros(self._nb_classes)
        self._FP = np.zeros(self._nb_classes)
        self._FP_spatial = np.zeros(self._nb_classes)
        self._FN = np.zeros(self._nb_classes)
        self._Nref = np.zeros(self._nb_classes)

        self._S = 0
        self._D = 0
        self._I = 0

        # Class-sensitive localization
        self._total_DE = np.zeros(self._nb_classes)
        self._DE_TP = np.zeros(self._nb_classes)
        self._DE_FP = np.zeros(self._nb_classes)
        self._DE_FN = np.zeros(self._nb_classes)

    def reset(self):
        """Reset all accumulators."""
        self._TP[:] = 0
        self._FP[:] = 0
        self._FP_spatial[:] = 0
        self._FN[:] = 0
        self._Nref[:] = 0
        self._S = 0
        self._D = 0
        self._I = 0
        self._total_DE[:] = 0
        self._DE_TP[:] = 0
        self._DE_FP[:] = 0
        self._DE_FN[:] = 0

    def update(self, pred, gt):
        """
        Update metrics with predictions and ground truth for a sequence.

        Args:
            pred: dict mapping frame_idx -> dict mapping class_idx ->
                  dict mapping track_idx -> [x, y, z]
            gt: same format as pred
        """
        for frame_idx in range(max(len(gt), len(pred))):
            loc_FN, loc_FP = 0, 0
            gt_frame = gt.get(frame_idx, {})
            pred_frame = pred.get(frame_idx, {})

            for class_cnt in range(self._nb_classes):
                nb_gt = len(gt_frame.get(class_cnt, {}))
                nb_pred = len(pred_frame.get(class_cnt, {}))

                if nb_gt > 0:
                    self._Nref[class_cnt] += nb_gt

                if nb_gt > 0 and nb_pred > 0:
                    gt_doas = np.array(list(gt_frame[class_cnt].values()))[:, :3]
                    pred_doas = np.array(list(pred_frame[class_cnt].values()))[:, :3]

                    doa_err_list, row_inds, col_inds = least_distance_between_gt_pred(
                        gt_doas, pred_doas
                    )

                    # Count metrics
                    Pc = nb_pred
                    Rc = nb_gt
                    FNc = max(0, Rc - Pc)
                    FPcinf = max(0, Pc - Rc)
                    Kc = min(Pc, Rc)

                    Lc = np.sum(doa_err_list > self._spatial_T)
                    FPct = Lc
                    TPct = Kc - FPct

                    self._total_DE[class_cnt] += doa_err_list.sum()
                    self._TP[class_cnt] += TPct
                    self._DE_TP[class_cnt] += Kc
                    self._FP[class_cnt] += FPcinf
                    self._DE_FP[class_cnt] += FPcinf
                    self._FP_spatial[class_cnt] += FPct
                    self._FN[class_cnt] += FNc
                    self._DE_FN[class_cnt] += FNc

                    loc_FP += FPcinf + FPct
                    loc_FN += FNc

                elif nb_gt > 0 and nb_pred == 0:
                    loc_FN += nb_gt
                    self._FN[class_cnt] += nb_gt
                    self._DE_FN[class_cnt] += nb_gt

                elif nb_gt == 0 and nb_pred > 0:
                    loc_FP += nb_pred
                    self._FP[class_cnt] += nb_pred
                    self._DE_FP[class_cnt] += nb_pred

            self._S += min(loc_FP, loc_FN)
            self._D += max(0, loc_FN - loc_FP)
            self._I += max(0, loc_FP - loc_FN)

    def compute(self):
        """
        Compute final SELD metrics (macro-averaged).

        Returns:
            dict with keys: ER, F, LE, LR, SELD_score
        """
        ER = (self._S + self._D + self._I) / (self._Nref.sum() + eps)

        # F-score (macro)
        F = self._TP / (eps + self._TP + self._FP_spatial + 0.5 * (self._FP + self._FN))
        F = F.mean()

        # Localization Error (macro)
        LE = self._total_DE / (self._DE_TP + eps)
        LE[self._DE_TP == 0] = 180.0
        LE = LE.mean()

        # Localization Recall (macro)
        LR = self._DE_TP / (eps + self._DE_TP + self._DE_FN)
        LR = LR.mean()

        # SELD score (early stopping metric)
        SELD_score = np.mean([ER, 1 - F, LE / 180, 1 - LR])

        return {
            'ER': float(ER),
            'F': float(F),
            'LE': float(LE),
            'LR': float(LR),
            'SELD_score': float(SELD_score)
        }


def multi_accdoa_to_events(pred_output, nb_classes, thresh_unify=15):
    """
    Convert Multi-ACCDOA model output to event dictionary for evaluation.

    Takes the raw model output of shape (T, 3*3*num_classes) where each track
    has 3 axes (x, y, z) per class. Activity is detected when norm(xyz) > 0.5.

    The 3 tracks are unified when they predict the same class at similar
    locations (within thresh_unify degrees).

    Args:
        pred_output: numpy array (T, num_track * 3 * num_classes)
            e.g., (T, 72) for 3 tracks * 3 axes * 8 classes
        nb_classes: number of sound classes
        thresh_unify: angular threshold in degrees for unifying tracks

    Returns:
        output_dict: dict[frame_idx] -> dict[class_idx] -> dict[track_idx] -> [x, y, z]
    """
    # Each track has 3*nb_classes values (x, y, z for each class)
    track_size = 3 * nb_classes

    # Extract per-track predictions: track k starts at k * track_size
    tracks_xyz = []   # list of (x, y, z) arrays per track
    tracks_sed = []   # list of activity detection per track
    tracks_doa = []   # list of raw doa arrays per track

    for k in range(3):
        offset = k * track_size
        xk = pred_output[:, offset:offset + nb_classes]
        yk = pred_output[:, offset + nb_classes:offset + 2 * nb_classes]
        zk = pred_output[:, offset + 2 * nb_classes:offset + 3 * nb_classes]
        sed_k = np.sqrt(xk ** 2 + yk ** 2 + zk ** 2) > 0.5
        doa_k = pred_output[:, offset:offset + track_size]
        tracks_xyz.append((xk, yk, zk))
        tracks_sed.append(sed_k)
        tracks_doa.append(doa_k)

    output_dict = {}
    for frame_cnt in range(pred_output.shape[0]):
        frame_events = {}
        for class_cnt in range(nb_classes):
            track_out_idx = 0
            events_for_class = {}

            def _get_doa(track_id, cls):
                return [
                    tracks_doa[track_id][frame_cnt][cls],
                    tracks_doa[track_id][frame_cnt][cls + nb_classes],
                    tracks_doa[track_id][frame_cnt][cls + 2 * nb_classes]
                ]

            def _check_similar(t0, t1, cls):
                if tracks_sed[t0][frame_cnt][cls] and tracks_sed[t1][frame_cnt][cls]:
                    d0 = _get_doa(t0, cls)
                    d1 = _get_doa(t1, cls)
                    dist = distance_between_cartesian_coordinates(
                        d0[0], d0[1], d0[2], d1[0], d1[1], d1[2]
                    )
                    return dist < thresh_unify
                return False

            flag_0sim1 = _check_similar(0, 1, class_cnt)
            flag_1sim2 = _check_similar(1, 2, class_cnt)
            flag_2sim0 = _check_similar(2, 0, class_cnt)

            sim_sum = flag_0sim1 + flag_1sim2 + flag_2sim0

            if sim_sum == 0:
                for k in range(3):
                    if tracks_sed[k][frame_cnt][class_cnt] > 0.5:
                        events_for_class[track_out_idx] = _get_doa(k, class_cnt)
                        track_out_idx += 1

            elif sim_sum == 1:
                if flag_0sim1:
                    d0 = np.array(_get_doa(0, class_cnt))
                    d1 = np.array(_get_doa(1, class_cnt))
                    events_for_class[track_out_idx] = ((d0 + d1) / 2).tolist()
                    track_out_idx += 1
                    if tracks_sed[2][frame_cnt][class_cnt] > 0.5:
                        events_for_class[track_out_idx] = _get_doa(2, class_cnt)
                        track_out_idx += 1
                elif flag_1sim2:
                    if tracks_sed[0][frame_cnt][class_cnt] > 0.5:
                        events_for_class[track_out_idx] = _get_doa(0, class_cnt)
                        track_out_idx += 1
                    d1 = np.array(_get_doa(1, class_cnt))
                    d2 = np.array(_get_doa(2, class_cnt))
                    events_for_class[track_out_idx] = ((d1 + d2) / 2).tolist()
                    track_out_idx += 1
                elif flag_2sim0:
                    if tracks_sed[1][frame_cnt][class_cnt] > 0.5:
                        events_for_class[track_out_idx] = _get_doa(1, class_cnt)
                        track_out_idx += 1
                    d2 = np.array(_get_doa(2, class_cnt))
                    d0 = np.array(_get_doa(0, class_cnt))
                    events_for_class[track_out_idx] = ((d2 + d0) / 2).tolist()
                    track_out_idx += 1

            else:
                d0 = np.array(_get_doa(0, class_cnt))
                d1 = np.array(_get_doa(1, class_cnt))
                d2 = np.array(_get_doa(2, class_cnt))
                events_for_class[track_out_idx] = ((d0 + d1 + d2) / 3).tolist()
                track_out_idx += 1

            if events_for_class:
                frame_events[class_cnt] = events_for_class

        if frame_events:
            output_dict[frame_cnt] = frame_events

    return output_dict


def label_to_events(label, nb_classes):
    """
    Convert ground truth Multi-ACCDOA ADPIT label to event dictionary.

    Args:
        label: numpy array (T, 6, 4, num_classes) ADPIT format
        nb_classes: number of sound classes

    Returns:
        output_dict: dict[frame_idx] -> dict[class_idx] ->
                     dict[track_idx] -> [x, y, z]
    """
    output_dict = {}
    # Use only the first 3 tracks (A, B0, B1) which represent unique sources
    # Actually for evaluation, we use the case-A target (track 0) which
    # contains the "no overlap" version - but we need all unique sources.
    # The proper approach is to reconstruct from ADPIT:
    # Track 0 (A): all sources when no same-class overlap
    # Tracks 1-2 (B0, B1): when 2 same-class sources
    # Tracks 3-5 (C0, C1, C2): when 3 same-class sources

    for frame_cnt in range(label.shape[0]):
        frame_events = {}
        for class_cnt in range(nb_classes):
            events_for_class = {}
            track_idx = 0

            # Check each ADPIT track for activity
            # Use track 0 (case A) as primary
            act_a = label[frame_cnt, 0, 0, class_cnt]
            if act_a > 0.5:
                events_for_class[track_idx] = [
                    label[frame_cnt, 0, 1, class_cnt],
                    label[frame_cnt, 0, 2, class_cnt],
                    label[frame_cnt, 0, 3, class_cnt]
                ]
                track_idx += 1

            # Check if B tracks indicate additional same-class sources
            act_b0 = label[frame_cnt, 1, 0, class_cnt]
            act_b1 = label[frame_cnt, 2, 0, class_cnt]
            if act_b0 > 0.5 and act_b1 > 0.5:
                # Two same-class sources - B0 and B1 have them
                events_for_class = {}
                track_idx = 0
                events_for_class[track_idx] = [
                    label[frame_cnt, 1, 1, class_cnt],
                    label[frame_cnt, 1, 2, class_cnt],
                    label[frame_cnt, 1, 3, class_cnt]
                ]
                track_idx += 1
                events_for_class[track_idx] = [
                    label[frame_cnt, 2, 1, class_cnt],
                    label[frame_cnt, 2, 2, class_cnt],
                    label[frame_cnt, 2, 3, class_cnt]
                ]
                track_idx += 1

            # Check if C tracks indicate 3 same-class sources
            act_c0 = label[frame_cnt, 3, 0, class_cnt]
            act_c1 = label[frame_cnt, 4, 0, class_cnt]
            act_c2 = label[frame_cnt, 5, 0, class_cnt]
            if act_c0 > 0.5 and act_c1 > 0.5 and act_c2 > 0.5:
                events_for_class = {}
                track_idx = 0
                for c_track in [3, 4, 5]:
                    events_for_class[track_idx] = [
                        label[frame_cnt, c_track, 1, class_cnt],
                        label[frame_cnt, c_track, 2, class_cnt],
                        label[frame_cnt, c_track, 3, class_cnt]
                    ]
                    track_idx += 1

            if events_for_class:
                frame_events[class_cnt] = events_for_class

        if frame_events:
            output_dict[frame_cnt] = frame_events

    return output_dict

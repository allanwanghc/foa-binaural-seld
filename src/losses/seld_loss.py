"""
ADPIT (Auxiliary Duplicating Permutation Invariant Training) loss for SELD.

Adapted from DCASE baseline MSELoss_ADPIT in seldnet_model.py.

For Multi-ACCDOA with 3 output tracks, the ADPIT loss considers all 13
possible permutations (1 + 6 + 6) and selects the minimum loss:
    - 1 permutation for case A (no overlap from same class)
    - 6 permutations for case B (2 sources from same class)
    - 6 permutations for case C (3 sources from same class)
"""

import torch
import torch.nn as nn


class ADPITLoss(nn.Module):
    """
    Auxiliary Duplicating Permutation Invariant Training loss.

    For Multi-ACCDOA with 3 tracks, computes MSE loss for all 13 permutations
    of track assignments and selects the minimum.

    The target has 6 auxiliary tracks (ADPIT format):
        tracks 0: case A (no same-class overlap)
        tracks 1-2: case B (2 same-class sources) with 2 tracks
        tracks 3-5: case C (3 same-class sources) with 3 tracks

    Each target track has shape [act, x, y, z] * num_classes, where act is
    multiplied with [x, y, z] to zero out inactive sources.

    Input pred: (B, T, num_track * 3 * num_classes)
        e.g., (B, T, 72) for 3 tracks * 3 axes (xyz) * 8 classes
        After reshape: (B, T, 9, num_classes)
    Input target: (B, T, 6, num_axis, num_classes)
        where num_axis = 4 (act, x, y, z) or 5 (act, x, y, z, dist)

    The activity mask (axis 0) is multiplied with coordinates (axes 1:)
    to produce the effective target.
    """

    def __init__(self):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')

    def _each_calc(self, output, target):
        """Compute class-wise frame-level MSE loss."""
        return self._each_loss(output, target).mean(dim=2)  # Average over axes

    def forward(self, output, target):
        """
        Args:
            output: (B, T, num_track * 3 * num_classes)
                    e.g., (B, T, 3*3*8) = (B, T, 72) for xyz only
            target: (B, T, 6, num_axis, num_classes)
                    e.g., (B, T, 6, 4, 8) where 4=[act, x, y, z]

        Returns:
            loss: scalar MSE loss (minimum over 13 permutations)
        """
        # Build ADPIT auxiliary targets by multiplying activity with coordinates
        # target[:, :, k, 0:1, :] is the activity (0 or 1)
        # target[:, :, k, 1:, :] is [x, y, z] (or [x, y, z, dist])
        # We only use first 4 axes: act, x, y, z
        num_axes_used = min(4, target.shape[3])

        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:num_axes_used, :]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:num_axes_used, :]
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:num_axes_used, :]
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:num_axes_used, :]
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:num_axes_used, :]
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:num_axes_used, :]

        # 1 permutation of A (no overlap from same class)
        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)

        # 6 permutations of B (overlap with 2 sources from same class)
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)

        # 6 permutations of C (overlap with 3 sources from same class)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        # Reshape output to match target shape
        output = output.reshape(
            output.shape[0], output.shape[1],
            target_A0A0A0.shape[2], target_A0A0A0.shape[3]
        )

        # Padding: each permutation group is padded with the other groups
        # to avoid zero targets (which would bias the loss)
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1

        # Compute loss for all 13 permutations
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        # Stack and find minimum permutation per frame per class
        loss_stack = torch.stack((
            loss_0, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6,
            loss_7, loss_8, loss_9, loss_10, loss_11, loss_12
        ), dim=0)

        loss_min = torch.min(loss_stack, dim=0).indices

        # Select loss corresponding to minimum permutation
        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        return loss

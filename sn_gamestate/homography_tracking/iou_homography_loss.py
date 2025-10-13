import numpy as np
from .mask_algorithm import display_pitch_overlay_new_mask


class MaskingLoss:

    def __init__(self, unet_mask, H_start):
        self.unet_mask = unet_mask
        self.H_start = H_start
        self.counter = 0


    def function(self, h_parameters):
        #self.counter += 1
        try:
            H_diff = np.array([
                [h_parameters[0], h_parameters[1] , h_parameters[2]],
                [h_parameters[3], h_parameters[4], h_parameters[5]],
                [h_parameters[6], h_parameters[7], 0]
            ])
            H = self.H_start + H_diff

            mask = display_pitch_overlay_new_mask(H)
            iou = self._iou(mask)
            #print(iou)
            return 1 - iou
        except Exception as e:
            print(f'Loss function errored: {e}')
            return 1.0

    def _iou(self, mask):
        def compute_iou(mask1, mask2):
            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()
            if union == 0:
                return 0  # or 1.0 if both are completely empty and you want to define that as perfect match
            return intersection / union
        return compute_iou(mask, self.unet_mask)

    def return_homography(self, h_parameters):
        H_diff = np.array([
            [h_parameters[0], h_parameters[1] , h_parameters[2]],
            [h_parameters[3], h_parameters[4], h_parameters[5]],
            [h_parameters[6], h_parameters[7], 0]
        ])
        #print(f'Called {self.counter} times')

        H = self.H_start + H_diff
        print(H)
        return H
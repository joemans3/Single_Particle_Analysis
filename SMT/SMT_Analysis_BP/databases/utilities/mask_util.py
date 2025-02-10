from collections.abc import Callable

import numpy as np


def mask_pixel_indexes(mask: np.ndarray, mask_val: int = 255) -> np.ndarray:
    # we want to find the [[x,y],[x,y]] coordinates of the masked_img_space which are a certain value defined by mask_value
    indxes = np.where(mask == mask_val)
    domain_xy = np.array(
        [indxes[1], indxes[0]]
    ).T  # this is to correct for the fact that img space and where invert the x,y axis
    # now we can create the masked domain
    return domain_xy


def check_point_in_mask_gen(
    mask: np.ndarray, mask_val: int = 255
) -> Callable[[float, float], bool]:
    masked_domain_xy = mask_pixel_indexes(mask, mask_val)

    def check_point(x: float, y: float) -> bool:
        list_np_arrays = masked_domain_xy
        array_to_check = np.array([int(x), int(y)])
        is_in_list = np.any(np.all(array_to_check == list_np_arrays, axis=1))
        if is_in_list:
            return True
        else:
            return False

    return check_point

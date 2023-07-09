import cv2
import numpy as np
from numba import njit, cuda
from config import detection

import warnings
import logging
from numba.core.errors import NumbaPerformanceWarning

warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


class Flagger:
    def __init__(
        self,
        opponent_color,
        red_lower,
        red_upper,
        red_lower_2,
        red_upper_2,
        blue_lower,
        blue_upper,
    ):
        """
        Constructor.
        opponent_color: 'RED' or "BLUE' indicating the color of the opponent team.
        red_lower: tuple indicating a HSV value indicating the lower limit of red color
        red_upper: tuple indicating a HSV value indicating the upper limit of red color
        blue_lower: tuple indicating a HSV value indicating the lower limit of blue color
        blue_upper: tuple indicating a HSV value indicating the upper limit of blue color
        """
        self.opponent_color = opponent_color

        self.red_lower = np.array(red_lower, dtype=np.uint8)
        self.red_upper = np.array(red_upper, dtype=np.uint8)
        self.red_lower_2 = np.array(red_lower_2, dtype=np.uint8)
        self.red_upper_2 = np.array(red_upper_2, dtype=np.uint8)
        self.blue_lower = np.array(blue_lower, dtype=np.uint8)
        self.blue_upper = np.array(blue_upper, dtype=np.uint8)
        
        # breaks with fiducials code, new implementation below
        # self.red_or_blue = (
        #     red_or_blue_cuda if detection["USE_CUDA_FLAGGER"] else red_or_blue
        # )

    # Check if the robot in the frame is enemy robot or not. Returns true if it is enemy robot
    def check_enemy(self, frame):
        """
        Checks if the frame contains the robot of enemy team or ally.
        Paramter:
        frame = input image

        Returns:
        True if enemy robot detected else False
        """
        # Convert BGR format ot HSV format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color = self.red_or_blue(frame)

        return color == self.opponent_color
    
    def red_or_blue(self, frame):
        if detection["USE_CUDA_FLAGGER"]:
            return red_or_blue_cuda(
                frame,
                self.blue_lower,
                self.blue_upper,
            )
        return red_or_blue(frame, self.blue_lower, self.blue_upper,self.red_lower,self.red_upper, self.red_lower_2,self.red_upper_2)



@cuda.jit(fastmath=True)
def mask_kernel(frame, blue_mask, blue_lower, blue_upper):
    x, y = cuda.grid(2)
    if x < frame.shape[0] and y < frame.shape[1]:
        if (
            frame[x, y, 0] >= blue_lower[0]
            and frame[x, y, 0] <= blue_upper[0]
            and frame[x, y, 1] >= blue_lower[1]
            and frame[x, y, 1] <= blue_upper[1]
            and frame[x, y, 2] >= blue_lower[2]
            and frame[x, y, 2] <= blue_upper[2]
        ):
            blue_mask[x, y] = 1
        else:
            blue_mask[x, y] = 0


# determines if the robot in the frame is of red or blue color
# @TODO: This is not fast because of the for loop, we need to find a way to vectorize this
# @TODO: finetune threads_per_block and blocks_per_grid
def red_or_blue_cuda(
    frame: np.ndarray,
    blue_lower: np.ndarray,
    blue_upper: np.ndarray,
) -> str:
    """
    Checks what is the color of the robot in the image using cuda.
    Parameters:
    frame = input image

    Returns:
    "RED" if red is detected the most else "BLUE"
    """
    # Get Blue masks
    # Allocate device memory
    d_frame = cuda.to_device(frame)
    blue_mask = np.zeros_like(frame[:, :, 0], dtype=np.int32)
    d_blue_mask = cuda.to_device(blue_mask)
    # Define grid and block dimensions
    threads_per_block = (32, 24)
    blocks_per_grid_x = (
        frame.shape[0] + threads_per_block[0] - 1
    ) // threads_per_block[0]
    blocks_per_grid_y = (
        frame.shape[1] + threads_per_block[1] - 1
    ) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    # Launch kernel
    mask_kernel[blocks_per_grid, threads_per_block](
        d_frame, d_blue_mask, blue_lower, blue_upper
    )
    # Copy results from device to host
    blue_mask = d_blue_mask.copy_to_host()

    b_area = np.count_nonzero(blue_mask)
    return "BLUE" if b_area > 15 else "RED"


@njit(cache=True)
def red_or_blue(
    frame: np.ndarray,
    blue_lower: np.ndarray,
    blue_upper: np.ndarray,
    red_lower: np.ndarray,
    red_upper: np.ndarray,
    red_lower_2: np.ndarray,
    red_upper_2: np.ndarray,
    
) -> str:
    """
    Checks what is the color of the robot in the image.
    Parameters:
    frame = input image

    Returns:
    "RED" if red is detected the most else "BLUE"
    """
    # Get Blue masks
    blue_mask = (
        (frame[:, :, 0] >= blue_lower[0])
        & (frame[:, :, 0] <= blue_upper[0])
        & (frame[:, :, 1] >= blue_lower[1])
        & (frame[:, :, 1] <= blue_upper[1])
        & (frame[:, :, 2] >= blue_lower[2])
        & (frame[:, :, 2] <= blue_upper[2])
    )
    b_area = np.count_nonzero(blue_mask)

    red_mask = (
        (frame[:, :, 0] >= red_lower[0])
        & (frame[:, :, 0] <= red_upper[0])
        & (frame[:, :, 1] >= red_lower[1])
        & (frame[:, :, 1] <= red_upper[1])
        & (frame[:, :, 2] >= red_lower[2])
        & (frame[:, :, 2] <= red_upper[2])
    )
    r_area = np.count_nonzero(red_mask)
    #if b_area > r_area:
    #     return "BLUE"
    # elif r_area > b_area:
    #     return "RED"
    # else: return "NULL"
    
    if b_area > 50: #and #b_area > r_area:
        return "BLUE"
    #else:
    return "RED"
    #return "BLUE"
    # if b_area > 15:
    #     return "BLUE"
    # return "RED"



from .iou_homography_loss import MaskingLoss
from scipy.optimize import differential_evolution

def improve_iou(H_neighbor, iou_mask):

    loss = MaskingLoss(
        unet_mask=iou_mask,
        H_start=H_neighbor,
    )

    bounds = [
            (-10e-1, 10e-1),
            (-10e-1, 10e-1),
            (-10e3, 10e3),
            (-10e-1, 10e-1),
            (-10e-1, 10e-1),
            (-10e3, 10e3),
            (-10e-1, 10e-1),
            (-10e-1, 10e-1),
            ]

    result = differential_evolution(loss.function, bounds, popsize=15, maxiter=50, tol=1e-3, workers=-1)

    print(result.fun)

    H = loss.return_homography(result.x)

    return H
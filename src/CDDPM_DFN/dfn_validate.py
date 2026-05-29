import cv2
import numpy as np


def count_fractures(img):

    # img = (img * 255).astype(np.uint8)

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    lines = cv2.HoughLinesP(
        binary, rho=1, theta=np.pi / 180, threshold=15, minLineLength=10, maxLineGap=5
    )

    if lines is None:
        return 0

    return len(lines)


if __name__ == "__main__":
    image_path = "/home/tet/zhaoheng/fast-api-project/src/CDDPM_DFN_NUM_CLASSIFIER/images/generated_dfn_20260522_130355.png"
    image_path = (
        "/home/tet/zhaoheng/fast-api-project/data/dfn_data/images/dfn_00000.png"
    )
    image_path = "/home/tet/zhaoheng/fast-api-project/data/dfn_data/images/dfn_09999.png"
    image_path = "/home/tet/zhaoheng/fast-api-project/data/dfn_data/images/dfn_09987.png"
    image_path = "/home/tet/zhaoheng/fast-api-project/src/CDDPM_DFN_NUM_CLASSIFIER/images/generated_dfn_20260522_125956.png"
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(f"img的类型：{type(img)}, shape: {img.shape}")
    print(img)
    l = count_fractures(img)
    print(f"l: {l}")

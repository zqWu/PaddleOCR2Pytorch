import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
import cv2
import tools.infer.pytorchocr_utility as utility

from tools.infer.predict_det_npu import TextDetector


def _init_det_args():
    args = utility.parse_args()
    args.det_model_path = '../infer_models/ch_ptocr_v4_det_server_infer.pth'
    args.det_yaml_path = '../configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml'
    return args


def default_text_detector():
    args = _init_det_args()
    text_detector = TextDetector(args)
    return text_detector


def text_detect(cv2_img):
    if cv2_img is None:
        raise ValueError('text_detect, cv_img is None')
    text_detector = default_text_detector()
    return text_detector(cv2_img)


if __name__ == '__main__':
    cv2_img = cv2.imread('../doc/imgs_2/3_2.png')
    # cv2_img = cv2.imread('../doc/imgs_2/00009282.jpg')
    dt_boxes, elapse = text_detect(cv2_img)
    print(f'dt_boxes={dt_boxes}')  # 4点组成的 四边形
    print(f'elapse={elapse}')
# dt_boxes=[
# [[  6.  50.]
#   [926.  50.]
#   [926.  73.]
#   [  6.  73.]]
#
#  [[  5.   9.]
#   [947.   9.]
#   [947.  35.]
#   [  5.  35.]]
# ]

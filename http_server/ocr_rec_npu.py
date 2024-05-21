import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

from project_root import relative_path_in_root
from tools.infer.predict_rec_npu import TextRecognizer

sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
import cv2
import tools.infer.pytorchocr_utility as utility


def _init_args():
    args = utility.parse_args()
    args.rec_model_path = relative_path_in_root('infer_models/ch_ptocr_v4_rec_server_infer.pth')
    args.rec_yaml_path = relative_path_in_root('configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml')
    return args


def default_text_recognizer():
    args = _init_args()
    text_recognizer = TextRecognizer(args)
    return text_recognizer


def text_recognize(cv2_img):
    if cv2_img is None:
        raise ValueError('text_recognize, cv_img is None')
    text_recognizer = default_text_recognizer()
    return text_recognizer(cv2_img)


if __name__ == '__main__':
    cv2_img = cv2.imread('../doc/imgs_words/ch/word_1.jpg')
    # cv2_img = cv2.imread('../doc/imgs_2/3_2.png') # 多行文本不能识别
    rec_res, predict_time = text_recognize([cv2_img])
    print(f'rec_res={rec_res}')  # rec_res=[('韩国小馆', 0.996515)]
    print(f'predict_time={predict_time}')

    pass

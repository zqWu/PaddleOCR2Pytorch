import io
from datetime import datetime, date, time
from pathlib import Path
from typing import Union, Optional, List

import cv2
import numpy as np
from PIL import Image

_DATE_FORMAT_ = '%Y_%m_%d'
_TIME_FORMAT_ = '%H_%M_%S'
_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def add_white_border(cv2_img, border_size=10):
    """生成边界填充"""
    color_white = (255, 255, 255)
    top, bottom, left, right = border_size, border_size, border_size, border_size
    return cv2.copyMakeBorder(cv2_img,
                              top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color_white)


def get_cv2_image_w_h(cv2_img):
    h, w, c = cv2_img.shape
    return w, h


def join_two_path(path1: Union[Path, str], path2: Union[Path, str]) -> str:
    if isinstance(path1, str):
        path1 = Path(path1)
    final_path = Path.joinpath(path1, path2)
    return str(final_path)


def current_date() -> str:
    """当前日期 yyyy_mm_dd"""
    return datetime.now().strftime(_DATE_FORMAT_)


def try_parse_date(date_str: str) -> Optional[date]:
    try:
        # 尝试将字符串转换为日期对象
        date_obj = datetime.strptime(date_str, _DATE_FORMAT_)
        return date_obj.date()
    except ValueError:
        # 处理可能不合法的日期格式
        # print(f'日期字符串 {date_str} 不是有效的 {_DATE_FORMAT_} 格式')
        return None


def current_time() -> str:
    """当前时间 HH_MM_SS"""
    return datetime.now().strftime(_TIME_FORMAT_)


def try_parse_time(time_str: str) -> Optional[time]:
    try:
        # 尝试将字符串转换为日期对象
        time_obj = datetime.strptime(time_str, _TIME_FORMAT_)
        return time_obj.time()
    except ValueError:
        # 处理可能不合法的日期格式
        # print(f'日期字符串 {time_str} 不是有效的 {_TIME_FORMAT_} 格式')
        return None


def current_datetime() -> str:
    return datetime.now().strftime(_DATETIME_FORMAT)


def image_byte_to_cv2_image(img_bytes):
    np_arr = np.frombuffer(img_bytes, np.uint8)
    cv2_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return cv2_image


def image_byte_to_pil_image(img_bytes):
    return Image.open(io.BytesIO(img_bytes))


def cv2_image_to_image_byte(cv2_image):
    success, png_bytes = cv2.imencode('.png', cv2_image)
    if success:
        return png_bytes.tobytes()
    else:
        return None


def crop_cv2_image(cv2_image, bbox: List):
    # cropped_img = img[top:bottom, left:right]
    return cv2_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def crop_pil_image(pil_image: Image, bbox: List):
    left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
    crop_image = pil_image.crop((left, top, right, bottom))
    return crop_image


def cv2_image_to_png_bytes(cv2_image):
    is_success, im_buf_arr = cv2.imencode(".png", cv2_image)
    if is_success:
        return im_buf_arr.tobytes()


def save_cv2_as_png(cv2_image, file_path):
    cv2.imwrite(file_path, cv2_image)


def save_text(txt: str, file_path):
    with open(file_path, mode='wt') as f:
        f.write(txt)

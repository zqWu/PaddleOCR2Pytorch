import io
import pathlib
from typing import List

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from http_server.logger import get_logger
from http_server.ocr_api import OcrRow
from http_server.service import service_det_and_rec, service_img_ocr

logger = get_logger(__name__)

app = FastAPI()

# cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


async def _file_to_cv2(file: UploadFile):
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))
    cv2_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return cv2_img


@app.get('/ruok')
async def ruok():
    logger.debug(f'imok')
    return {'result': 'imok'}


@app.post('/api/v1/img_ocr', summary='上传图片文件, 进行ocr文本识别')
async def img_ocr(file: UploadFile):
    support_ext = ['.jpg', '.jpeg', '.png']
    if pathlib.Path(file.filename).suffix not in support_ext:
        return {'error': f'仅仅支持{support_ext}'}

    img = await _file_to_cv2(file)
    return service_img_ocr(img)


@app.post('/api/v1/det_and_extra', summary='ocr识别图片, 返回 [{text, bbox, score}]')
async def v2_det_and_extra(file: UploadFile):
    support_ext = ['.jpg', '.jpeg', '.png']
    if pathlib.Path(file.filename).suffix not in support_ext:
        return {'error': f'仅仅支持{support_ext}'}
    img = await _file_to_cv2(file)
    rows: List[OcrRow] = service_det_and_rec(img)
    return rows


if __name__ == '__main__':
    import uvicorn

    print(__name__)
    uvicorn.run('rest_api:app', host='0.0.0.0', port=22008, workers=1)
    # debug 时会运行2个, 不知道 why
    # 但是命令行则正常: uvicorn http_server.rest_api:app --host 0.0.0.0 --port 22008 --workers 1
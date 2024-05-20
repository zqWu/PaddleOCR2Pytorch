# project for ocr run on npu device


# conda env
- doc_parser_ocr, 由培能安装了 npu


# 备忘
```bash
# det 模型
python ./tools/infer/predict_det.py \
--image_dir ./doc/imgs/00009282.jpg \
--det_model_path /data1/scripts/PaddleOCR2Pytorch/infer_models/ch_ptocr_v4_det_server_infer.pth \
--det_yaml_path ./configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml

# det 模型 + npu
python ./tools/infer/predict_det_npu.py \
--image_dir ./doc/imgs/00009282.jpg \
--det_model_path ./infer_models/ch_ptocr_v4_det_server_infer.pth \
--det_yaml_path ./configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml


python ./tools/infer/predict_det_npu.py \
--image_dir ./doc/imgs_2 \
--det_model_path ./infer_models/ch_ptocr_v4_det_server_infer.pth \
--det_yaml_path ./configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml

# rec 模型
python ./tools/infer/predict_rec.py \
--image_dir ./doc/imgs_words/ch/word_1.jpg \
--rec_model_path ./infer_models/ch_ptocr_v4_rec_server_infer.pth \
--rec_yaml_path ./configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml \
--rec_image_shape='3,48,320'

# rec 模型 + npu
python ./tools/infer/predict_rec_npu.py \
--image_dir ./doc/imgs_words/ch/word_1.jpg \
--rec_model_path ./infer_models/ch_ptocr_v4_rec_server_infer.pth \
--rec_yaml_path ./configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml \
--rec_image_shape='3,48,320'

python ./tools/infer/predict_cls.py \
--image_dir ./doc/imgs_words/ch/word_4.jpg  \
--cls_model_path /data1/scripts/PaddleOCR2Pytorch/infer_models/ch_ptocr_mobile_v2.0_cls_infer.pth
```
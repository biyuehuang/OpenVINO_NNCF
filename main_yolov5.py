import sys
from utils.dataloaders import create_dataloader
from utils.general import check_dataset
from pathlib import Path
from utils.general import download
import os
from openvino.runtime import serialize
from openvino.tools import mo

from openvino.runtime import Core

DATASET_CONFIG = "./data/coco128.yaml"
IMAGE_SIZE = 640
MODEL_NAME = "yolov5"
#MODEL_PATH = f"./{MODEL_NAME}"
#onnx_path = f"{MODEL_NAME}.onnx"

#fp32_path      = "/home/kiwi/ov/uniview/yolov5/IR_FP32/yolov5s_openvino_model/yolov5s.xml"
#nncf_int8_path = "/home/kiwi/ov/uniview/yolov5/INT8/yolov5s_openvino_model/yolov5s.xml"
#pt_path = "/home/kiwi/ov/uniview/yolov5/yolov5s.pt"

#fp32_path = "/home/kiwi/ov/uniview/yolov5/IR_FP32/yolov5m_openvino_model/yolov5m.xml"
#nncf_int8_path =  "/home/kiwi/ov/uniview/yolov5/INT8/yolov5m_openvino_model/yolov5m.xml"
#pt_path = "/home/kiwi/ov/uniview/yolov5/yolov5m.pt"


fp32_path = "/home/kiwi/ov/uniview/yolov5/IR_FP32/yolov5l_openvino_model/yolov5l.xml"
nncf_int8_path = "/home/kiwi/ov/uniview/yolov5/INT8/yolov5l_openvino_model/yolov5l.xml"
pt_path = "/home/kiwi/ov/uniview/yolov5/yolov5l.pt"

#print(f"Export ONNX to OpenVINO FP32 IR to: {fp32_path}")
#model = mo.convert_model(onnx_path,compress_to_fp16=False)
#serialize(model, fp32_path)

def create_data_source():
    """
    Creates COCO 2017 validation data loader. The method downloads COCO 2017
    dataset if it does not exist.
    """
    if not os.path.exists("./datasets/coco128"):
        urls = ["https://ultralytics.com/assets/coco128.zip"]
        download(urls, dir="datasets")

    data = check_dataset(DATASET_CONFIG)
    val_dataloader = create_dataloader(
            data["val"], imgsz=IMAGE_SIZE, batch_size=1, stride=32, pad=0.5, workers=1
    )[0]

    return val_dataloader


import nncf
def transform_fn(data_item):
    # unpack input images tensor
    images = data_item[0]
    # convert input tensor into float format
    images = images.float()
    # scale input
    images = images / 255
    # convert torch tensor to numpy array
    images = images.cpu().detach().numpy()
    return images

data_source = create_data_source()
# Wrap framework-specific data source into the `nncf.Dataset` object.
nncf_calibration_dataset = nncf.Dataset(data_source, transform_fn)

core = Core()
ov_model = core.read_model(fp32_path)
quantized_model = nncf.quantize(
    ov_model, nncf_calibration_dataset, preset=nncf.QuantizationPreset.MIXED, subset_size=300
)
#nncf_int8_path = f"{MODEL_PATH}/NNCF_INT8_openvino_model/{MODEL_NAME}_int8.xml"
serialize(quantized_model, nncf_int8_path)

from export import attempt_load, yaml_save
from val import run as validation_fn


model = attempt_load(
     pt_path, device="cpu", inplace=True, fuse=True
)
metadata = {"stride": int(max(model.stride)), "names": model.names}  # model metadata
yaml_save(Path(nncf_int8_path).with_suffix(".yaml"), metadata)
#yaml_save(Path(fp32_path).with_suffix(".yaml"), metadata)

print("Checking the accuracy of the original model:")
fp32_metrics = validation_fn(
    data=DATASET_CONFIG,
    weights=Path(fp32_path).parent,
    batch_size=1,
    workers=1,
    plots=False,
    device="cpu",
    iou_thres=0.65,
)

fp32_ap5 = fp32_metrics[0][2]
fp32_ap_full = fp32_metrics[0][3]
print(f"mAP@.5 = {fp32_ap5}")
print(f"mAP@.5:.95 = {fp32_ap_full}")

print("Checking the accuracy of the NNCF int8 model:")
int8_metrics = validation_fn(
    data=DATASET_CONFIG,
    weights=Path(nncf_int8_path).parent,
    batch_size=1,
    workers=1,
    plots=False,
    device="cpu",
    iou_thres=0.65,
)

nncf_int8_ap5 = int8_metrics[0][2]
nncf_int8_ap_full = int8_metrics[0][3]
print(f"mAP@.5 = {nncf_int8_ap5}")
print(f"mAP@.5:.95 = {nncf_int8_ap_full}")

#!benchmark_app -m {fp16_path} -d CPU -api async -t 15

#!benchmark_app -m {nncf_int8_path} -d CPU -api async -t 15


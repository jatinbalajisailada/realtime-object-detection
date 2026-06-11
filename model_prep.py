### this code was originally programmed in NoteBooks

#using YOLO(26) model from ultralytics
from ultralytics import YOLO

#1 loading the pre-trained YOLO26 Nano model
yoloModel = YOLO("yolo26n.pt")

#2 exporting the model as ONNX format
#making dynamic=Flase makes input size fixed, important for stable c++ edge deployment
onnx_path = yoloModel.export(format = "onnx", imgsz = 640, dynamic = False)
print(f"Model is exported to {onnx_path}")

#quantization of model is neccessary
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

fp32Model = "yolo26n.onnx"
int8Model = "yolo26n_int8.onnx"

print("starting int8 quantization...")

#1 compress the model by using Dynamic Quantization
quantize_dynamic(
  model_input = fp32Model,
  model_output = int8Model,
  weight_type = QuantType.QInt8
)

print("completed int8!")

#2 displaying compression(quantization) effects
originalSize = os.path.getsize(fp32Model) / (1024*1024)
quantisedSize = os.path.getsize(int8Model) / (1024*1024)

print(f"Original ONNX Size:  {originalSize:.2f} MB")
print(f"Quantized INT8 Size: {quantisedSize:.2f} MB")
print(f"Total Compression:   {((originalSize - quantisedSize) / originalSize) * 100:.1f}%")
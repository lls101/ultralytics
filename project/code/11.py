import onnxruntime as ort

# 打印可用的执行提供者
print("Available providers:", ort.get_available_providers())

# 创建推理会话
ort_session = ort.InferenceSession(r"D:\Workspace\models\anno\yolov8x_r0240702.onnx")

# 打印会话使用的提供者
print("Session providers:", ort_session.get_providers())

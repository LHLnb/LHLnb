# 通过摄像头识别人脸情绪

这个示例会打开本机摄像头，实时检测画面中的人脸，并给出情绪识别结果（如 happy / sad / neutral）。

## 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2. 运行

```bash
python emotion_camera.py
```

- 按 `Q` 键退出。
- 若提示摄像头权限问题，请在系统设置里允许 Python/终端访问摄像头。

## 3. 说明

- 代码每隔约 `0.3` 秒进行一次情绪分析，避免每帧推理导致卡顿。
- 使用 `DeepFace` 的 `emotion` 分析能力，`enforce_detection=False` 可在短时未检测到人脸时避免程序直接报错退出。

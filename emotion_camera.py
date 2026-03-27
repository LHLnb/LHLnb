"""通过摄像头实时识别人脸情绪。

依赖：
    pip install -r requirements.txt
运行：
    python emotion_camera.py
"""

from __future__ import annotations

import time
from typing import Any

import cv2
from deepface import DeepFace


WINDOW_NAME = "Emotion Recognition (Press Q to quit)"
ANALYZE_INTERVAL_SECONDS = 0.3


def _format_emotions(emotion_result: dict[str, float], top_k: int = 3) -> list[str]:
    """将情绪概率字典格式化为可显示文本。"""
    sorted_emotions = sorted(emotion_result.items(), key=lambda item: item[1], reverse=True)
    lines: list[str] = []
    for label, score in sorted_emotions[:top_k]:
        lines.append(f"{label}: {score:.1f}%")
    return lines


def _extract_first_result(raw_result: Any) -> dict[str, Any]:
    """兼容 DeepFace 可能返回 list 或 dict 的结果。"""
    if isinstance(raw_result, list):
        return raw_result[0] if raw_result else {}
    if isinstance(raw_result, dict):
        return raw_result
    return {}


def main() -> None:
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("无法打开摄像头，请检查设备权限或摄像头是否被占用。")

    last_analyze_at = 0.0
    last_dominant = "unknown"
    last_emotions: dict[str, float] = {}

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("读取摄像头画面失败，正在退出。")
                break

            now = time.time()
            if now - last_analyze_at >= ANALYZE_INTERVAL_SECONDS:
                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=["emotion"],
                        enforce_detection=False,
                        detector_backend="opencv",
                        silent=True,
                    )
                    parsed = _extract_first_result(result)
                    last_dominant = str(parsed.get("dominant_emotion", "unknown"))
                    emotions = parsed.get("emotion", {})
                    if isinstance(emotions, dict):
                        last_emotions = {str(k): float(v) for k, v in emotions.items()}
                except Exception as exc:  # noqa: BLE001
                    last_dominant = "error"
                    last_emotions = {}
                    print(f"分析失败：{exc}")

                last_analyze_at = now

            cv2.putText(
                frame,
                f"Dominant: {last_dominant}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            for i, line in enumerate(_format_emotions(last_emotions, top_k=3), start=1):
                cv2.putText(
                    frame,
                    line,
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

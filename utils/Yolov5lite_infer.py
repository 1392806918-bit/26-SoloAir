import cv2
import numpy as np
import os
import logging


class YOLOv5Lite:
    def __init__(self, model_path, class_path, confThreshold=0.5, nmsThreshold=0.5, target_class_id=None):
        self.backend = os.path.splitext(model_path)[1].lower()
        self.net = None
        self.rknn = None
        self.classes = [line.strip() for line in open(class_path)]
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.target_class_id = target_class_id
        if self.backend == ".onnx":
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.log_severity_level = 3
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.net = ort.InferenceSession(model_path, so, providers=providers)
            self.input_shape = (self.net.get_inputs()[0].shape[2], self.net.get_inputs()[0].shape[3])
        elif self.backend == ".rknn":
            from rknnlite.api import RKNNLite

            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(model_path)
            if ret != 0:
                raise RuntimeError(f"load_rknn failed: {model_path}")
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
            if ret != 0:
                raise RuntimeError("init_runtime failed")
            self.input_shape = (320, 320)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

    def letterBox(self, img):
        h, w = img.shape[:2]
        newh, neww = self.input_shape
        scale = min(neww/w, newh/h)
        nh, nw = int(h*scale), int(w*scale)
        top = (newh-nh)//2
        left = (neww-nw)//2
        resized = cv2.resize(img, (nw, nh))
        canvas = np.zeros((newh, neww, 3), dtype=np.uint8)
        canvas[top:top+nh, left:left+nw] = resized
        return canvas, nh, nw, top, left

    def detect(self, frame):
        img, nh, nw, top, left = self.letterBox(frame)
        if self.backend == ".onnx":
            blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.expand_dims(np.transpose(blob, (2, 0, 1)), axis=0)
            outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0]
        elif self.backend == ".rknn":
            blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            blob = np.expand_dims(blob, axis=0)
            outputs = self.rknn.inference(inputs=[blob])
            if outputs is None:
                raise RuntimeError("rknn inference failed")
            outs = outputs[0]
        else:
            raise RuntimeError(f"Unsupported backend: {self.backend}")
        return self.postprocess(frame, outs, (nh, nw, top, left))

    def postprocess(self, frame, outs, pad_hw):
        outs = np.asarray(outs)
        if outs.ndim == 3 and outs.shape[0] == 1:
            outs = outs[0]
        if outs.ndim != 2:
            raise ValueError(f"Unsupported output shape: {outs.shape}")

        if outs.shape[-1] == 6:
            return self._postprocess_nms_output(frame, outs, pad_hw)

        if outs.shape[0] == 84:
            outs = outs.T
        elif outs.shape[1] != 84:
            raise ValueError(f"Unsupported raw YOLO output shape: {outs.shape}")

        return self._postprocess_raw_output(frame, outs, pad_hw)

    def _postprocess_nms_output(self, frame, outs, pad_hw):
        newh, neww, padh, padw = pad_hw
        h, w = frame.shape[:2]
        ratiow, ratioh = w/neww, h/newh
        results = []
        boxes, confidences, classIds = [], [], []
        for det in outs:
            conf, cls_id = det[4], det[5]
            cls_id = int(cls_id)
            if self.target_class_id is not None and cls_id != self.target_class_id:
                continue
            if conf > self.confThreshold:
                x1 = int((det[0]-padw)*ratiow)
                y1 = int((det[1]-padh)*ratioh)
                x2 = int((det[2]-padw)*ratiow)
                y2 = int((det[3]-padh)*ratioh)
                boxes.append([x1, y1, x2-x1, y2-y1])
                confidences.append(float(conf))
                classIds.append(cls_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        if indices is not None and len(indices) > 0:
            indices = np.array(indices).flatten()
            for i in indices:
                x, y, w_, h_ = boxes[i]
                x1, y1, x2, y2 = x, y, x + w_, y + h_
                results.append([x1, y1, x2, y2, confidences[i], classIds[i]])
        return results

    def _postprocess_raw_output(self, frame, outs, pad_hw):
        newh, neww, padh, padw = pad_hw
        h, w = frame.shape[:2]
        ratiow, ratioh = w / neww, h / newh
        boxes, confidences, classIds = [], [], []

        box_xywh = outs[:, :4]
        scores = outs[:, 4:]
        if self.target_class_id is not None:
            cls_id = int(self.target_class_id)
            if cls_id < 0 or cls_id >= scores.shape[1]:
                return []
            class_confidences = scores[:, cls_id]
            class_ids = np.full(class_confidences.shape, cls_id, dtype=np.int32)
        else:
            class_ids = np.argmax(scores, axis=1).astype(np.int32)
            class_confidences = scores[np.arange(scores.shape[0]), class_ids]

        keep = class_confidences > self.confThreshold
        for box, conf, cls_id in zip(box_xywh[keep], class_confidences[keep], class_ids[keep]):
            cx, cy, bw, bh = box
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            x1 = int((x1 - padw) * ratiow)
            y1 = int((y1 - padh) * ratioh)
            x2 = int((x2 - padw) * ratiow)
            y2 = int((y2 - padh) * ratioh)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w <= 0 or box_h <= 0:
                continue
            boxes.append([x1, y1, box_w, box_h])
            confidences.append(float(conf))
            classIds.append(int(cls_id))

        results = []
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        if indices is not None and len(indices) > 0:
            indices = np.array(indices).flatten()
            for i in indices:
                x, y, w_, h_ = boxes[i]
                results.append([x, y, x + w_, y + h_, confidences[i], classIds[i]])
        return results

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None


import threading
import time

class CameraStream:
    def __init__(self, src=0, width=640, height=480, fps=20):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.reconnect = isinstance(src, str) and src.startswith("rtsp://")
        self._last_reconnect = 0
        self.cap = self._open_capture()
        self.ret, self.frame = self.cap.read()
        logging.info("Initial camera read ret=%s shape=%s", self.ret, getattr(self.frame, "shape", None))
        self.running = True
        t = threading.Thread(target=self.update, daemon=True)
        t.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.ret, self.frame = ret, frame
                continue
            self.ret = False
            if self.reconnect and time.time() - self._last_reconnect > 2.0:
                self._last_reconnect = time.time()
                logging.warning("Video source read failed; reconnecting %s", self.src)
                self.cap.release()
                time.sleep(0.5)
                self.cap = self._open_capture()
            else:
                time.sleep(0.03)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

    def _open_capture(self):
        if isinstance(self.src, str) and self.src.startswith("rtsp://"):
            cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.src)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        logging.info(
            "Video source opened=%s src=%s requested=%sx%s@%s actual=%sx%s@%s",
            cap.isOpened(),
            self.src,
            self.width,
            self.height,
            self.fps,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
        )
        return cap

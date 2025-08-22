import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os

# ==== キャリブレーションクラス ====
class EncoderCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_dir, input_shape, cache_file="encoder_calib.cache"):
        super().__init__()
        self.image_dir = image_dir
        self.c, self.h, self.w = input_shape  # (3,720,1280)
        self.cache_file = cache_file
        self.current_index = 0

        # 画像リスト
        exts = [".jpg", ".jpeg", ".png", ".bmp"]
        self.files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                      if os.path.splitext(f)[1].lower() in exts]

        # GPU メモリ確保（バッチサイズなし）
        self.device_input = cuda.mem_alloc(
            trt.volume((self.c, self.h, self.w)) * np.dtype(np.float32).itemsize
        )

    def preprocess(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.w, self.h)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        return img

    def get_batch_size(self):
        return 1  # 疑似的に1バッチ

    def get_batch(self, names):
        if self.current_index >= len(self.files):
            return None

        img = self.preprocess(self.files[self.current_index])
        batch = np.ascontiguousarray(img, dtype=np.float32)

        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

class DecoderCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, npy_dir, input_shape, cache_file="decoder_calib.cache"):
        super().__init__()
        self.npy_dir = npy_dir
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.current_index = 0

        # .npy ファイル一覧
        self.files = [os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith(".npy")]
        self.files.sort()

        # GPU メモリ確保
        self.device_input = cuda.mem_alloc(
            trt.volume(self.input_shape) * np.dtype(np.float32).itemsize
        )

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.current_index >= len(self.files):
            return None

        arr = np.load(self.files[self.current_index]).astype(np.float32)
        batch = np.ascontiguousarray(arr)
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)



# ==== エンジンビルド ====
def build_int8_engine(onnx_path, calib, engine_file="model_int8.engine"):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    config.set_flag(trt.BuilderFlag.INT8)
    # 入出力をflot16に変換したい場合
    # config.set_flag(trt.BuilderFlag.FP16)

    if hasattr(config, "set_int8_calibrator"):
        config.set_int8_calibrator(calib)
    else:
        config.int8_calibrator = calib

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build INT8 engine")

    with open(engine_file, "wb") as f:
        f.write(serialized_engine)
    print(f"[INFO] INT8 エンジンを保存しました: {engine_file}")


if __name__ == "__main__":
    encoder_model = "models/encoder13.onnx"
    image_dir = "images"
    input_shape = (3, 720, 1280)

    calib = EncoderCalibrator(image_dir, input_shape)
    build_int8_engine(encoder_model, calib, engine_file="encoder_int8.engine")


    decoder_model = "models/decoder13.onnx"
    npy_dir = "saved_calib_data"  # encoder の出力を保存してある場所
    input_shape = (45, 80, 16)

    calib = DecoderCalibrator(npy_dir, input_shape)
    build_int8_engine(decoder_model, calib, engine_file="decoder_int8.engine")

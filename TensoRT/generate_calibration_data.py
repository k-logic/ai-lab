import onnxruntime as ort
import numpy as np
import cv2
import os

# ==== 画像前処理 ====
def load_and_preprocess_image(path, size=(1280, 720)):
    """画像を読み込み → RGB変換 → resize → CHW float32 正規化"""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"画像が見つかりません: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC → CHW
    return image

# ==== キャリブレーションデータ生成 ====
def generate_calibration_data(encoder_path, image_dir, save_dir, providers, size=(1280, 720)):
    """画像フォルダを処理してキャリブレーション用の npy を保存"""
    os.makedirs(save_dir, exist_ok=True)

    # ONNX Runtime セッションを一度だけ作成
    sess_opt = ort.SessionOptions()
    sess_opt.log_severity_level = 0
    session = ort.InferenceSession(encoder_path, sess_options=sess_opt, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"[INFO] Encoder Input : {input_name}, shape={session.get_inputs()[0].shape}")
    print(f"[INFO] Encoder Output: {output_name}, shape={session.get_outputs()[0].shape}")

    # 対象ファイル（画像のみ）
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    files = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in exts]

    for i, fname in enumerate(files):
        path = os.path.join(image_dir, fname)
        try:
            # 前処理 & 推論
            image_chw = load_and_preprocess_image(path, (1280, 720))
            encoded = session.run([output_name], {input_name: image_chw})[0]

            # 保存
            np.save(os.path.join(save_dir, f"encoded_{i:04d}.npy"), encoded.astype(np.float32))
            print(f"[INFO] {i}: {fname} → 保存完了 {encoded.shape}")
        except Exception as e:
            print(f"[WARN] {fname} を処理できませんでした: {e}")

    print(f"[INFO] 全 {len(files)} 枚の処理が完了しました。出力は {save_dir}/ に保存されました。")

# ==== 使用例 ====
if __name__ == "__main__":
    encoder_path = "models/encoder11.onnx"
    image_dir = "images/"
    save_dir = "saved_calib_data"

    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './engine_cache',
            'trt_max_workspace_size': 2147483648,
            'trt_fp16_enable': True,
        }),
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider'
    ]

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"入力フォルダが存在しません: {image_dir}")

    generate_calibration_data(encoder_path, image_dir, save_dir, providers)

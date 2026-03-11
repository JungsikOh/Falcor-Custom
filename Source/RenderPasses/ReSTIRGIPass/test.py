import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def load_exr(path):
    """EXR 파일을 로드합니다. (OpenCV IO_ENABLE_OPENEXR 옵션 필요)"""
    # 환경변수 설정 (OpenCV EXR 활성화)
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    # BGR -> RGB 변환
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)

def calculate_rel_mse(gt, test, epsilon=1e-2):
    """Relative MSE 계산 (렌더링 품질 평가 표준)"""
    # 0으로 나누기 방지를 위해 epsilon 사용
    rel_mse = np.mean(((test - gt) ** 2) / (gt**2 + epsilon))
    return rel_mse

def tonemap_srgb(x):
    """SSIM 측정을 위해 HDR 데이터를 [0, 1] 범위의 sRGB 유사 공간으로 톤매핑"""
    return np.clip(x / (1.0 + x), 0, 1)

def evaluate_quality(ref_path, test_path):
    # 1. 이미지 로드
    ref = load_exr(ref_path)
    test = load_exr(test_path)

    # 2. 해상도 일치 확인 (다를 경우 리사이징)
    if ref.shape != test.shape:
        print(f"Warning: 해상도가 다릅니다. {test_path}를 {ref_path}에 맞게 조정합니다.")
        test = cv2.resize(test, (ref.shape[1], ref.shape[0]))

    # 3. 데이터 클리닝 (NaN, Inf 제거)
    ref = np.nan_to_num(ref, nan=0.0, posinf=1.0, neginf=0.0)
    test = np.nan_to_num(test, nan=0.0, posinf=1.0, neginf=0.0)

    # 지표 계산 시작
    # [1] MSE (Mean Squared Error)
    mse_val = np.mean((ref - test) ** 2)

    # [2] RelMSE (Rendering Standard)
    rel_mse_val = calculate_rel_mse(ref, test)

    # [3] PSNR (Peak Max 기준)
    data_range = np.max(ref) if np.max(ref) > 0 else 1.0
    psnr_val = psnr(ref, test, data_range=data_range)

    # [4] SSIM (Structural Similarity on Tonemapped image)
    ref_tm = tonemap_srgb(ref)
    test_tm = tonemap_srgb(test)
    # skimage는 multichannel 파라미터가 최신 버전에서 channel_axis로 변경됨
    ssim_val = ssim(ref_tm, test_tm, data_range=1.0, channel_axis=2)

    return {
        "MSE": mse_val,
        "RelMSE": rel_mse_val,
        "PSNR": psnr_val,
        "SSIM (Tonemapped)": ssim_val
    }

# 실행 예시
if __name__ == "__main__":
    # GT(참조) 이미지와 디노이징된 이미지 경로
    reference_img = "D:/dev/research/falcor/output_frame/test8/Mogwai_pt_bistro3.ReSTIRGIPass.color.262.exr"
    denoised_img = "D:/dev/research/falcor/output_frame/test8/Mogwai_pt_bistro3.AccumulatePass.output.262.exr"

    try:
        results = evaluate_quality(reference_img, denoised_img)

        print("-" * 30)
        print(f"Evaluated: {denoised_img}")
        print(f"Reference: {reference_img}")
        print("-" * 30)
        for metric, value in results.items():
            print(f"{metric:20s}: {value:.6f}")
        print("-" * 30)

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Tip: 'pip install opencv-python scikit-image' 명령어로 필요한 라이브러리를 설치하세요.")

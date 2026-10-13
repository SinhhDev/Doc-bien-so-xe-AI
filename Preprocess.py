# Preprocess.py - Module chứa các hàm tiền xử lý ảnh (tách kênh sáng, tăng tương phản, làm mờ và nhị phân hóa)

import math

import cv2
import numpy as np

# Các biến cấu hình cấp module ####################################################################
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)  # Kích thước bộ lọc Gauss (kích cỡ càng lớn thì ảnh càng mờ)
ADAPTIVE_THRESH_BLOCK_SIZE = 19       # Kích thước vùng lân cận để tính ngưỡng nhị phân hóa thích nghi
ADAPTIVE_THRESH_WEIGHT = 9            # Hằng số trừ đi từ giá trị trung bình có trọng số

###################################################################################################
def preprocess(imgOriginal):
    """
    Hàm tiền xử lý tổng hợp: Chuyển ảnh gốc sang ảnh xám (kênh V), tăng tương phản,
    làm mịn giảm nhiễu và chuyển thành ảnh nhị phân đảo ngược.
    """
    # Trích xuất kênh cường độ sáng (Value) từ hệ màu HSV để làm ảnh xám (Grayscale)
    imgGrayscale = extractValue(imgOriginal)
    # imgGrayscale = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY) # Nên dùng hệ màu HSV để ít bị ảnh hưởng bởi ánh sáng

    # Tăng cường độ tương phản để làm nổi bật biển số, giúp dễ tách khỏi nền hơn
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    # cv2.imwrite("imgGrayscalePlusTopHatMinusBlackHat.jpg", imgMaxContrastGrayscale)

    height, width = imgGrayscale.shape

    # Làm mịn ảnh bằng bộ lọc Gauss 5x5 (sigma = 0) để loại bỏ nhiễu hạt
    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    # cv2.imwrite("gauss.jpg", imgBlurred)

    # Nhị phân hóa thích nghi (Adaptive Threshold) tạo ảnh đen trắng (chữ trắng trên nền đen)
    imgThresh = cv2.adaptiveThreshold(
        imgBlurred,
        255.0,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_THRESH_BLOCK_SIZE,
        ADAPTIVE_THRESH_WEIGHT
    )

    # Trả về ảnh xám và ảnh nhị phân
    return imgGrayscale, imgThresh
# Kết thúc hàm preprocess

###################################################################################################
def extractValue(imgOriginal):
    """
    Hàm chuyển đổi ảnh từ không gian màu BGR sang HSV và tách lấy kênh Value (cường độ sáng).
    Không dùng trực tiếp kênh màu RGB vì một màu cụ thể (ví dụ màu đỏ) thường pha trộn
    nhiều thành phần, khó xác định độ sáng chuẩn bằng kênh Value của HSV.
    """
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    # Tách thành 3 kênh: Hue (Màu sắc), Saturation (Độ bão hòa), Value (Giá trị cường độ sáng)
    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
# Kết thúc hàm extractValue

###################################################################################################
def maximizeContrast(imgGrayscale):
    """
    Hàm làm tăng độ tương phản lên mức tối đa bằng các phép toán hình thái học (Top-Hat và Black-Hat).
    Công thức: Ảnh kết quả = Ảnh xám + TopHat - BlackHat
    """
    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Tạo mặt nạ cấu trúc (kernel) 3x3

    # Phép Top-Hat: Làm nổi bật các chi tiết sáng trên nền tối
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations=10)
    # cv2.imwrite("tophat.jpg", imgTopHat)

    # Phép Black-Hat: Làm nổi bật các chi tiết tối trên nền sáng
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations=10)
    # cv2.imwrite("blackhat.jpg", imgBlackHat)

    # Cộng ảnh gốc với Top-Hat rồi trừ đi Black-Hat để kéo giãn độ tương phản
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    # cv2.imshow("imgGrayscalePlusTopHatMinusBlackHat", imgGrayscalePlusTopHatMinusBlackHat)
    # Trả về ảnh đã được tăng cường độ tương phản
    return imgGrayscalePlusTopHatMinusBlackHat
# Kết thúc hàm maximizeContrast
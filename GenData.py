# GenData.py - Tạo dữ liệu huấn luyện (nhãn và ảnh làm phẳng) cho mô hình KNN từ ảnh mẫu training_chars.png
import sys

import cv2
import numpy as np

# Các biến cấu hình cấp module
MIN_CONTOUR_AREA = 40      # Diện tích đường viền (contour) tối thiểu để được coi là một ký tự hợp lệ

RESIZED_IMAGE_WIDTH = 20   # Chiều rộng chuẩn hóa của ảnh ký tự (20 pixel)
RESIZED_IMAGE_HEIGHT = 30  # Chiều cao chuẩn hóa của ảnh ký tự (30 pixel)

###################################################################################################
def main():
    # Đọc ảnh chứa các ký tự mẫu dùng để huấn luyện
    imgTrainingNumbers = cv2.imread("training_chars.png")
    if imgTrainingNumbers is None:
        print("Lỗi: Không tìm thấy hoặc không đọc được file 'training_chars.png'")
        sys.exit(1)
    # imgTrainingNumbers = cv2.resize(imgTrainingNumbers, dsize=None, fx=0.5, fy=0.5)

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh gốc sang ảnh xám (Grayscale)
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)               # Làm mờ bằng bộ lọc Gauss 5x5 để giảm nhiễu

    # Chuyển đổi ảnh xám sang ảnh nhị phân đen trắng bằng phương pháp nhị phân hóa thích nghi
    imgThresh = cv2.adaptiveThreshold(
        imgBlurred,                     # Ảnh đầu vào đã làm mờ
        255,                            # Gán giá trị trắng tuyệt đối (255) cho các điểm ảnh vượt ngưỡng
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Dùng trọng số Gauss thay vì trung bình cộng để cho kết quả sắc nét hơn
        cv2.THRESH_BINARY_INV,          # Đảo ngược màu để ký tự có màu trắng, nền có màu đen
        11,                             # Kích thước vùng lân cận (11x11) dùng để tính toán giá trị ngưỡng
        2                               # Hằng số trừ đi từ giá trị trung bình có trọng số
    )

    cv2.imshow("imgThresh", imgThresh)  # Hiển thị ảnh nhị phân để đối chiếu

    imgThreshCopy = imgThresh.copy()    # Tạo bản sao của ảnh nhị phân vì hàm findContours có thể làm thay đổi ảnh gốc

    # Tìm tất cả các đường viền (contours) bao ngoài các ký tự
    npaContours, hierarchy = cv2.findContours(
        imgThreshCopy,                  # Ảnh đầu vào (sử dụng bản sao)
        cv2.RETR_EXTERNAL,              # Chỉ lấy các đường viền ngoài cùng của đối tượng
        cv2.CHAIN_APPROX_SIMPLE         # Nén các đoạn thẳng ngang, dọc, chéo và chỉ giữ lại các điểm đầu mút
    )

    # Khởi tạo mảng numpy rỗng để lưu trữ dữ liệu ảnh ký tự đã làm phẳng (0 hàng, 600 cột)
    npaFlattenedImages = np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    # Khởi tạo danh sách rỗng để lưu mã ASCII của từng ký tự do người dùng nhấn phím phân loại
    intClassifications = []

    # Danh sách mã ASCII của các chữ số (0-9) và chữ cái (A-Z) hợp lệ trên biển số xe (không tính dấu '-' và '.')
    intValidChars = [
        ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
        ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
        ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
        ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')
    ]

    for npaContour in npaContours:                                      # Duyệt qua từng đường viền tìm được
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:              # Kiểm tra nếu diện tích đủ lớn mới xét là ký tự
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)     # Lấy tọa độ và kích thước khung hình chữ nhật bao quanh

            # Vẽ khung hình chữ nhật màu đỏ bao quanh ký tự đang chờ người dùng nhập nhãn
            cv2.rectangle(
                imgTrainingNumbers,         # Vẽ trực tiếp lên ảnh mẫu ban đầu
                (intX, intY),               # Góc trên bên trái
                (intX + intW, intY + intH), # Góc dưới bên phải
                (0, 0, 255),                # Màu đỏ (BGR)
                2                           # Độ dày nét vẽ
            )

            # Cắt (crop) vùng ảnh chứa ký tự từ ảnh nhị phân và chuẩn hóa về kích thước 20x30
            imgROI = imgThresh[intY:intY + intH, intX:intX + intW]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            cv2.imshow("imgROI", imgROI)                                # Hiển thị ảnh ký tự vừa cắt
            cv2.imshow("imgROIResized", imgROIResized)                  # Hiển thị ảnh ký tự sau khi chuẩn hóa 20x30
            cv2.imshow("training_numbers.png", imgTrainingNumbers)      # Hiển thị ảnh tổng thể có đánh dấu khung đỏ

            intChar = cv2.waitKey(0)        # Chờ người dùng nhấn phím tương ứng với ký tự đang khoanh đỏ

            if intChar == 27:               # Nếu nhấn phím ESC (mã ASCII 27) thì đóng cửa sổ và thoát chương trình
                cv2.destroyAllWindows()
                sys.exit()
            elif intChar in intValidChars:  # Nếu phím vừa nhấn nằm trong danh sách ký tự hợp lệ (0-9, A-Z)
                # Lưu mã ASCII của ký tự vào danh sách nhãn (tổng cộng bộ dữ liệu chuẩn có 32 ký tự x 5 mẫu = 160 mẫu)
                intClassifications.append(intChar)

                # Làm phẳng ảnh ký tự 20x30 thành mảng 1 chiều gồm 600 điểm ảnh (1 hàng x 600 cột)
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

                # Nối vector ảnh hiện tại vào ma trận tổng hợp các ảnh mẫu
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)
    # Kết thúc vòng lặp duyệt contour

    # Chuyển danh sách nhãn từ kiểu số nguyên sang mảng numpy kiểu số thực (float32)
    fltClassifications = np.array(intClassifications, np.float32)

    # Định hình lại mảng nhãn thành ma trận cột 1 chiều để ghi ra file
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))

    print("\n\nHoàn tất quá trình tạo dữ liệu huấn luyện (training complete)!!\n")

    # Ghi dữ liệu nhãn và dữ liệu điểm ảnh đã làm phẳng ra file текстовый (.txt)
    np.savetxt("classifications.txt", npaClassifications)
    np.savetxt("flattened_images.txt", npaFlattenedImages)

    cv2.destroyAllWindows()  # Giải phóng bộ nhớ và đóng toàn bộ cửa sổ hiển thị
    return

if __name__ == "__main__":
    main()
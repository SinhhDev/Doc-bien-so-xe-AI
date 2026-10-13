import math

import cv2
import numpy as np

import Preprocess

# Tham số cho bước nhị phân hóa thích nghi (Adaptive Threshold)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

# Ngưỡng diện tích và tỷ lệ khung hình của ký tự so với biển số
Min_char_area = 0.015
Max_char_area = 0.06

Min_char = 0.01
Max_char = 0.09

Min_ratio_char = 0.25
Max_ratio_char = 0.7

# Kích thước giới hạn của biển số
max_size_plate = 18000
min_size_plate = 5000

# Kích thước chuẩn hóa của ảnh ký tự đưa vào mô hình KNN
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

tongframe = 0
biensotimthay = 0

# Tải dữ liệu và huấn luyện mô hình KNN
npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape(
    (npaClassifications.size, 1))  # Đưa mảng nhãn về 1 chiều để truyền vào hàm train
kNearest = cv2.ml.KNearest_create()  # Khởi tạo đối tượng KNN
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

# Đọc file video đầu vào (có thể đổi đường dẫn video tại đây)
cap = cv2.VideoCapture('data/video/video1.mp4')
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Lấy tổng số khung hình của video

# ================= CẤU HÌNH GHI VIDEO KẾT QUẢ =================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    'output_1920x1080.mp4',
    fourcc,
    30.0,
    (1920, 1080)
)
# ==============================================================

while (cap.isOpened()):
    # Đọc từng khung hình từ video
    ret, img = cap.read()
    if not ret or img is None:  # Kiểm tra nếu hết video hoặc lỗi khung hình thì dừng vòng lặp
        print("\nĐã xử lý xong toàn bộ video!")
        break

    tongframe = tongframe + 1
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)

    # Tiền xử lý ảnh: chuyển ảnh xám và nhị phân hóa
    imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
    canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Phát hiện biên bằng thuật toán Canny
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)  # Phép giãn nở (Dilation) để làm liền nét biên

    # Tìm và lọc các đường viền (contour) có khả năng là biển số xe
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 contour có diện tích lớn nhất
    screenCnt = []
    for c in contours:
        peri = cv2.arcLength(c, True)  # Tính chu vi của contour
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Xấp xỉ đa giác từ đường viền
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        ratio = w / h
        # Giữ lại các hình có 4 góc và tỷ lệ rộng/cao phù hợp với biển số vuông hoặc biển số dài
        if (len(approx) == 4) and (0.8 <= ratio <= 1.5 or 4.5 <= ratio <= 6.5):
            screenCnt.append(approx)

    if len(screenCnt) == 0:
        detected = 0
        print("No plate detected")
    else:
        detected = 1

    if detected == 1:
        n = 1
        for cnt_plate in screenCnt:

            ################## Tính góc nghiêng của biển số ###############
            (x1, y1) = cnt_plate[0, 0]
            (x2, y2) = cnt_plate[1, 0]
            (x3, y3) = cnt_plate[2, 0]
            (x4, y4) = cnt_plate[3, 0]
            array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            array.sort(reverse=True, key=lambda x: x[1])  # Sắp xếp tọa độ theo trục y giảm dần
            (x1, y1) = array[0]
            (x2, y2) = array[1]

            doi = abs(y1 - y2)
            ke = abs(x1 - x2)
            if ke == 0:
                angle = 90.0
            else:
                angle = math.atan(doi / ke) * (180.0 / math.pi)
            ###############################################################

            # Tạo mặt nạ (mask) che các vùng ngoài biển số
            mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
            new_image = cv2.drawContours(mask, [cnt_plate], 0, 255, -1)

            # Cắt (crop) vùng ảnh chứa biển số
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))

            roi = img[topx:bottomx + 1, topy:bottomy + 1]
            imgThresh = imgThreshplate[topx:bottomx + 1, topy:bottomy + 1]

            ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

            # Xoay biển số về vị trí nằm ngang
            if x1 < x2:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
            else:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

            roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
            imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))

            # Phóng to ảnh biển số gấp 3 lần để tách ký tự chính xác hơn
            roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
            imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

            # Tiền xử lý vùng biển số để chuẩn bị tách ký tự
            kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
            cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Phân tách từng ký tự trên biển số
            char_x_ind = {}
            char_x = []
            height, width, _ = roi.shape
            roiarea = height * width
            # print("roiarea", roiarea)
            for ind, cnt in enumerate(cont):
                area = cv2.contourArea(cnt)
                (x, y, w, h) = cv2.boundingRect(cont[ind])
                ratiochar = w / h
                # Lọc các contour thỏa mãn diện tích và tỷ lệ rộng/cao của một ký tự
                if (Min_char * roiarea < area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                    if x in char_x:  # Tăng nhẹ x nếu trùng tọa độ để không bị ghi đè dữ liệu
                        x = x + 1
                    char_x.append(x)
                    char_x_ind[x] = ind

            # Nhận dạng ký tự bằng thuật toán KNN
            if len(char_x) in range(7, 10):  # Chỉ xử lý biển số hợp lệ có từ 7 đến 9 ký tự
                cv2.drawContours(img, [cnt_plate], -1, (0, 255, 0), 3)
                char_x = sorted(char_x)  # Sắp xếp các ký tự từ trái sang phải theo trục x
                strFinalString = ""
                first_line = ""
                second_line = ""
                for i in char_x:
                    (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    imgROI = thre_mor[y:y + h, x:x + w]  # Cắt rời ảnh từng ký tự
                    imgROIResized = cv2.resize(imgROI,
                                               (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # Đưa ảnh ký tự về kích thước 20x30
                    npaROIResized = imgROIResized.reshape(
                        (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # Làm phẳng ảnh thành mảng 1 chiều (1x600)
                    npaROIResized = np.float32(npaROIResized)  # Chuyển mảng về kiểu số thực float32
                    _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=3)  # Phân loại ký tự bằng KNN với k=3
                    strCurrentChar = str(chr(int(npaResults[0][0])))  # Chuyển mã ASCII kết quả sang ký tự chữ/số
                    cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

                    # Phân chia ký tự thuộc dòng 1 hay dòng 2 dựa trên tọa độ y
                    if (y < height / 3):
                        first_line = first_line + strCurrentChar
                    else:
                        second_line = second_line + strCurrentChar

                strFinalString = first_line + second_line
                print("\n License Plate " + str(n) + " is: " + first_line + " - " + second_line + "\n")
                cv2.putText(img, strFinalString, (topy, topx), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
                n = n + 1
                biensotimthay = biensotimthay + 1
                cv2.imshow("a", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

    # Hiển thị khung hình video kết quả (thu nhỏ 50% để vừa màn hình)
    imgcopy = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow('License plate', imgcopy)

    # ===== GHI KHUNG HÌNH VÀO FILE VIDEO (1920x1080) =====
    frame_1080p = cv2.resize(img, (1920, 1080))
    out.write(frame_1080p)
    # =====================================================

    print("number of plates found", biensotimthay)
    print("total frame", tongframe)
    denom = total_video_frames if total_video_frames > 0 else tongframe
    print("plate found rate:", 100 * biensotimthay / denom, "%")

    # Nhấn phím 'q' để dừng video sớm
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng bộ nhớ và đóng toàn bộ cửa sổ
cap.release()
out.release()
cv2.destroyAllWindows()
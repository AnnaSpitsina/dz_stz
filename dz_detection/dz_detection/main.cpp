#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {

    Mat image = imread("test.png"); //test и test1 лежат в папочке test picture
    if (image.empty()) {
        cerr << "Не удалось открыть или найти изображение!" << endl;
        return -1;
    }

    // преобразование картинки в цветовое пространство HSV
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    // определение красного и оранжевого цвета
    Scalar lower_red = Scalar(0, 60, 50);
    Scalar upper_red = Scalar(13, 255, 255);

    // создание маски для красного цвета
    Mat mask_red;
    inRange(hsv, lower_red, upper_red, mask_red);
    // инверсия маски
    Mat mask_black;
    bitwise_not(mask_red, mask_black);
    // бинаризация для выделения черных областей
    Mat binary;
    threshold(mask_black, binary, 200, 255, THRESH_BINARY);
    // объединение близко расположенных точек
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);
    // ищу контуры черных областей
    vector<vector<Point>> contours_black;
    findContours(binary, contours_black, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // фильтруем их
    vector<vector<Point>> filtered_contours;
    for (const auto& contour : contours_black) {
        // аппроксимируем контур многоугольником с небольшим количеством углов
        double epsilon = 0.03 * arcLength(contour, true);
        vector<Point> approx;
        approxPolyDP(contour, approx, epsilon, true);
        if (approx.size() == 4 && isContourConvex(approx)) {
            Rect rect = boundingRect(approx);
            double aspect_ratio = static_cast<double>(rect.width) / rect.height;
            if (rect.area() > 100 && aspect_ratio > 0.1 && aspect_ratio < 10) {
                if (rect.tl().x > 10 && rect.tl().y > 10 && rect.br().x < image.cols - 10 && rect.br().y < image.rows - 10) {
                    filtered_contours.push_back(approx);
                }
            }
        }
    }

    // отображение
    drawContours(image, filtered_contours, -1, Scalar(0, 0, 255), 2);
    for (size_t i = 0; i < filtered_contours.size(); ++i) {
        // прямоугольник, ограничивающий текущую область
        Rect roi_rect = boundingRect(filtered_contours[i]);
        rectangle(image, roi_rect, Scalar(0, 255, 0), 2);
        Mat roi = image(roi_rect).clone();
        string window_name = "Region " + to_string(i + 1);
        namedWindow(window_name, WINDOW_NORMAL);
        imshow(window_name, roi);
    }
    namedWindow("Result", WINDOW_AUTOSIZE);
    imshow("Result", image);
    waitKey(0);

    return 0;
}

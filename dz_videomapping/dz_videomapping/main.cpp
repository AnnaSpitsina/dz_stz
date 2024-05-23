#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp> // Детекторы FAST, BRISK, ORB
#include <opencv2/xfeatures2d.hpp> // Экспериментальные или лицензированные детекторы (SIFT  SURF)

int main() {
    cv::Mat src, prevFrame, nextFrame;
    cv::VideoCapture cap("../video/test1.avi");
    if (!cap.isOpened()) {
        std::cerr << "Не можем открыть видео" << std::endl;
        return -1;
    }

    bool stop = false;
    // Определим частоту кадров на видео
    double rate = cap.get(cv::CAP_PROP_FPS);
    // Рассчитаем задержку в миллисекундах
    int delay = 1000 / rate;

    // Создаем объекты для детекторов SURF, SIFT и BRISK
    cv::Ptr<cv::Feature2D> surf = cv::xfeatures2d::SURF::create();
    cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create();
    cv::Ptr<cv::Feature2D> brisk = cv::BRISK::create();

    // Инициализируем переменные для рисования траектории
    cv::Point2f prevPos(500, 0);
    cv::Mat trajectory = cv::Mat::zeros(1000, 1000, CV_8UC3);

    while (!stop) {
        // Проверяем доступность кадра
        bool result = cap.grab();
        // Если он готов, считываем
        if (result)
            cap >> src;
        else
            continue;

        nextFrame = src.clone();

        if (prevFrame.empty()) {
            prevFrame = nextFrame.clone();
            continue;
        }

        // Находим ключевые точки
        std::vector<cv::KeyPoint> keypoints_surf, keypoints_sift, keypoints_brisk;
        cv::Mat descriptors_surf, descriptors_sift, descriptors_brisk;
        surf->detectAndCompute(nextFrame, cv::noArray(), keypoints_surf, descriptors_surf);
        sift->detectAndCompute(nextFrame, cv::noArray(), keypoints_sift, descriptors_sift);
        brisk->detectAndCompute(nextFrame, cv::noArray(), keypoints_brisk, descriptors_brisk);

        // Отображаем их на кадре
        cv::Mat img_surf, img_sift, img_brisk;
        cv::drawKeypoints(nextFrame, keypoints_surf, img_surf, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawKeypoints(nextFrame, keypoints_sift, img_sift, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawKeypoints(nextFrame, keypoints_brisk, img_brisk, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // Выполняем сопоставление
        std::vector<cv::DMatch> matches_surf, matches_sift, matches_brisk;
        cv::Ptr<cv::DescriptorMatcher> matcher_surf = cv::DescriptorMatcher::create("BruteForce");
        cv::Ptr<cv::DescriptorMatcher> matcher_sift = cv::DescriptorMatcher::create("BruteForce");
        cv::Ptr<cv::DescriptorMatcher> matcher_brisk = cv::DescriptorMatcher::create("BruteForce");
        matcher_surf->match(descriptors_surf, descriptors_surf, matches_surf);
        matcher_sift->match(descriptors_sift, descriptors_sift, matches_sift);
        matcher_brisk->match(descriptors_brisk, descriptors_brisk, matches_brisk);
        cv::Mat img_matches_surf, img_matches_sift, img_matches_brisk;
        cv::drawMatches(nextFrame, keypoints_surf, nextFrame, keypoints_surf, matches_surf, img_matches_surf);
        cv::drawMatches(nextFrame, keypoints_sift, nextFrame, keypoints_sift, matches_sift, img_matches_sift);
        cv::drawMatches(nextFrame, keypoints_brisk, nextFrame, keypoints_brisk, matches_brisk, img_matches_brisk);

        cv::imshow("SURF Matches", img_matches_surf);
        cv::imshow("SIFT Matches", img_matches_sift);
        cv::imshow("BRISK Matches", img_matches_brisk);

        // Преобразуем кадры в оттенки серого
        cv::Mat prevGray, nextGray;
        cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(nextFrame, nextGray, cv::COLOR_BGR2GRAY);

        // Оптический поток
        std::vector<cv::Point2f> prevPoints, nextPoints;
        prevPoints.push_back(prevPos);

        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prevGray, nextGray, prevPoints, nextPoints, status, err);
        cv::Point2f nextPos = nextPoints[0];

        // Инвертируем направление Y
        cv::line(trajectory, cv::Point(prevPos.x, 500 - prevPos.y), cv::Point(nextPos.x, 500 - nextPos.y), cv::Scalar(0, 255, 0), 2);
        cv::circle(trajectory, cv::Point(nextPos.x, 500 - nextPos.y), 2, cv::Scalar(0, 0, 255), -1);

        cv::imshow("Trajectory", trajectory);

        int key = cv::waitKey(delay);
        if (key == 27) // ESC
            stop = true;

        prevPos = nextPos;
        prevFrame = nextFrame.clone();
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}


#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <QDebug>
#include <eigen3/Eigen/Dense>
#include <locale>

using DataMatrix = std::vector<std::pair<double, double>>;

DataMatrix readDataFromFile(const std::string &filename) {
    DataMatrix data;
    std::ifstream file(filename);
    double x, y;

    while (file >> x >> y) {
        data.emplace_back(x, y);
    }

    return data;
}

// оценка параметров модели
std::pair<double, double> estimateParameters(const DataMatrix &data) {
    Eigen::MatrixXd T(data.size(), 2);
    Eigen::VectorXd Z(data.size());

    for (size_t i = 0; i < data.size(); ++i) {
        T(i, 0) = data[i].second;
        T(i, 1) = 1;
        Z(i) = std::log10(data[i].first);
    }

    Eigen::VectorXd solution = (T.transpose() * T).ldlt().solve(T.transpose() * Z);  //МНК
    return {solution(0), solution(1)};
}

//       1024*(Ro*10^(KT-KTo)                              1024*Ro                           1024*Ro
//Z = --------------------------  --> lgZ = KT - KTo + lg------------ = KT + C, где С = lg ----------- - KTo
//             Rc+Ro                                       (Rc+Ro)                           (Rc+Ro)
double modelFunction(double k, double c, double t) {
    return std::pow(10, k * t + c);
}

// фильтрации данных
DataMatrix filterDataRansac(const DataMatrix &data) {

    double limit = 5; // порог для внутренних точек
    int num_iterations = 100;
    int maxInliers = 0;
    std::pair<double, double> bestModelParams;

    for (int i = 0; i < num_iterations; ++i) {
        // две случайные точки
        std::pair<double, double> sample1 = data[rand() % data.size()];
        std::pair<double, double> sample2 = data[rand() % data.size()];

        // оценка параметров модели по ним
        DataMatrix subset = {sample1, sample2};
        auto modelParams = estimateParameters(subset);

        // количество внутренних точек
        int inliers = 0;
        for (const auto &sample : data) {
            if (std::abs(sample.first - modelFunction(modelParams.first, modelParams.second, sample.second)) < limit)
                ++inliers;
        }
        if (inliers > maxInliers) {
            maxInliers = inliers;
            bestModelParams = modelParams;
        }
    }

    // фильтрация по лучшему результату
    DataMatrix filteredData;
    for (const auto &sample : data) {
        if (std::abs(sample.first - modelFunction(bestModelParams.first, bestModelParams.second, sample.second)) < limit) {
            filteredData.push_back(sample);
        }
    }
    return filteredData;
}

// генерации точек и запись в файл для дальнейшего построения в Exel))
void generateAndSavePoints(const std::pair<double, double> &params, const std::string &filename, double t_min, double t_max, int num_points) {
    std::ofstream outFile(filename);

    double step = (t_max - t_min) / (num_points - 1);
    for (int i = 0; i < num_points; ++i) {
        double t = t_min + i * step;
        double z = modelFunction(params.first, params.second, t);
        outFile << t << " " << z << "\n";
    }
}

int main() {

    DataMatrix data = readDataFromFile("../dz_calibration/ZTdata.txt");
    //qDebug() << data;
    DataMatrix filteredData = filterDataRansac(data);
    auto estimatedParameters = estimateParameters(filteredData);
    std::cout << "Оценка параметров: K = " << estimatedParameters.first << ", C = " << estimatedParameters.second << "\n";

    // для Excel
    double z_min = 0.0;
    double z_max = 200.0;
    int num_points = 40;
    generateAndSavePoints(estimatedParameters, "../dz_calibration/result_data.txt", z_min, z_max, num_points);
    return 0;
}

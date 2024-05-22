#include <iostream>
#include <cmath>
#include <vector>
#include <eigen3/Eigen/Dense>

// первый диффур
double f(double t, double y) {
    return -0.7 * t - 0.7 * y;
}

// метод Рунге-Кутты 4-го порядка для уравнения
double rk4(double t, double y, double& h, double tol) {
    double k1, k2, k3, k4, y_new, diff;
    do {
        k1 = h * f(t, y);
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1);
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2);
        k4 = h * f(t + h, y + k3);
        y_new = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
        diff = std::abs(y_new - y);
        if (diff > tol) {
            h *= 0.5;
        } else if (diff < tol / 10) {
            h *= 2;
        }
    } while (diff > tol);
    return y_new;
}

// система диффуров
Eigen::VectorXd f(const Eigen::VectorXd& u, double t) {
    Eigen::VectorXd du(2);
    du[0] = 9 * u[0] + 24 * u[1] + 5 * cos(t) - (1.0 / 3.0) * sin(t);
    du[1] = -24 * u[0] - 51 * u[1] - 9 * cos(t) + (1.0 / 3.0) * sin(t);
    return du;
}

void rungeKutta(Eigen::VectorXd& u, double& t, double& h, double tol) {
    Eigen::VectorXd k1, k2, k3, k4, u_temp(2), u_new(2);
    double diff;
    do {
        k1 = f(u, t);
        u_temp = u + 0.5 * h * k1;
        k2 = f(u_temp, t + 0.5 * h);
        u_temp = u + 0.5 * h * k2;
        k3 = f(u_temp, t + 0.5 * h);
        u_temp = u + h * k3;
        k4 = f(u_temp, t + h);
        u_new = u + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
        diff = (u_new - u).cwiseAbs().maxCoeff();
        if (diff > tol) {
            h *= 0.5;
        } else if (diff < tol / 10) {
            h *= 2;
        }
    } while (diff > tol);
    u = u_new;
    t += h;
}

// для сравнения
double u1_analytic(double t) {
    return 2 * exp(-3 * t) - exp(-39 * t) + (1.0 / 3.0) * cos(t);
}

double u2_analytic(double t) {
    return -exp(-3 * t) + 2 * exp(-39 * t) - (1.0 / 3.0) * cos(t);
}

int main() {
    // первая задача
    {
        double t_end = 10;
        double y_initial = 1;

        double h = 0.1; // начальный шаг
        double tol = 0.001; // допустимая ошибка
        double max_discrepancy = 0; // максимальное рассогласование

        double t = 0.0;
        double y = y_initial;

        while (t < t_end) {
            double u = rk4(t, y, h, tol);
            double discrepancy = std::abs(u - y);
            if (discrepancy > max_discrepancy) {
                max_discrepancy = discrepancy;}
            std::cout << "u(" << t << ") = " << u << std::endl;
            // переход к следующему моменту времени
            t += h;
            // обновление y
            y = u;
        }
        std::cout << "Шаг, при котором рассогласование не превышает 0.001: " << h << std::endl;
        std::cout << "Максимальное рассогласование: " << max_discrepancy << std::endl;
    }

    // вторая часть
    {
        double t0 = 0.0; // начальное время
        double tf = 1.0; // конечное время
        double h = 0.001; // шаг интегрирования
        double tol = 0.001; // допустимая ошибка

        Eigen::VectorXd u(2);
        u << 4/3, 2/3;

        double t = t0;
        while (t <= tf) {
            rungeKutta(u, t, h, tol);
            double u1_analyt = u1_analytic(t);
            double u2_analyt = u2_analytic(t);
            std::cout << "t = " << t << " Численным методом u1(t) = " << u[0] << ", аналитическим u1(t) = " << u1_analyt << std::endl;
            std::cout << "t = " << t << " Численным методом u2(t) = " << u[1] << ", аналитическим u2(t) = " << u2_analyt << std::endl;
        }
        return 0;
    }
}

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>

double f(double x) {
    return 0.1 * (x - 4.0) * std::cos(x) + 0.5 * x;
}

double K(double z) {
    return std::abs(z) <= 1.0 ? 0.75 * (1.0 - z * z) : 0.0;
}

double mean(const std::vector<double>& y) {
    double s = 0.0;
    for (double v : y) s += v;
    return s / y.size();
}

double get_delta(const std::vector<double>& x) {
    std::vector<double> t = x;
    std::sort(t.begin(), t.end());
    double d = 0.0;
    for (int i = 1; i < (int)t.size(); ++i)
        if (t[i] - t[i - 1] > d) d = t[i] - t[i - 1];
    return d;
}

double predict(const std::vector<double>& x,
               const std::vector<double>& y,
               double x0, double h) {
    double a = 0.0, b = 0.0;
    for (int i = 0; i < (int)x.size(); ++i) {
        double w = K((x0 - x[i]) / h);
        a += w * y[i];
        b += w;
    }
    return b == 0.0 ? mean(y) : a / b;
}

double loocv(const std::vector<double>& x,
             const std::vector<double>& y,
             double beta, double delta) {
    double h = delta / beta;
    double mse = 0.0;
    int n = x.size();

    for (int i = 0; i < n; ++i) {
        std::vector<double> xt, yt;
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                xt.push_back(x[j]);
                yt.push_back(y[j]);
            }
        }
        double yp = predict(xt, yt, x[i], h);
        mse += (y[i] - yp) * (y[i] - yp);
    }

    return mse / n;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    int n = 80;
    std::vector<double> x(n), y(n);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> ux(-3.0, 7.0);
    std::normal_distribution<double> un(0.0, 0.3);

    for (int i = 0; i < n; ++i) {
        x[i] = ux(gen);
        y[i] = f(x[i]) + un(gen);
    }

    double delta = get_delta(x);
    double best_beta = 0.1;
    double best_mse = 1e100;

    for (int k = 1; k <= 20; ++k) {
        double beta = k / 10.0;
        double mse = loocv(x, y, beta, delta);
        if (mse < best_mse) {
            best_mse = mse;
            best_beta = beta;
        }
    }

    double h = delta / best_beta;

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t = finish - start;

    std::cout << "Лучшее beta: " << best_beta << "\n";
    std::cout << "LOO-MSE = " << best_mse << "\n";
    std::cout << "Время работы: " << t.count() << " сек.\n";

    return 0;
}
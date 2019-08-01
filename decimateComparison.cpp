#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

constexpr int SIZE = 25;
constexpr int FILTERSIZE = 5;
const std::array<float, FILTERSIZE> filter{0.1, 1.1, 2.1, 3.1, 4.1};
std::array<float, FILTERSIZE> filter_op = filter;
std::array<float, 2 * (FILTERSIZE - 1)> bank = {};
std::array<float, 2 * (FILTERSIZE - 1)> bank_for_optim = {};

void FIR(float* data, float* output) {
    int k;
    for (int i = 0; i < 2 * FILTERSIZE; i += 2) {
        float temp1 = 0, temp2 = 0;
        k = 0;
        for (int j = 0; j < FILTERSIZE; j++, k += 2) {
            if (i - k < 0) break;
            temp1 += filter[j] * data[i - k];
            temp2 += filter[j] * data[i - k + 1];
        }
        output[i] = temp1;
        output[i + 1] = temp2;
    }
    for (int i = 0; i < 2 * (FILTERSIZE - 1); i += 2) {
        output[i] += bank[i];
        output[i + 1] += bank[i + 1];
    }
    for (int i = 2 * FILTERSIZE; i < SIZE; i += 2) {
        float temp1 = 0, temp2 = 0;
        k = 0;
        for (int j = 0; j < FILTERSIZE; j++, k += 2) {
            temp1 += filter[j] * data[i - k];
            temp2 += filter[j] * data[i - k + 1];
        }
        output[i] = temp1;
        output[i + 1] = temp2;
    }
    int idx;
    for (int i = 0; i < 2 * (FILTERSIZE - 1); i += 2) {
        k = 0;
        for (int j = 0; j < FILTERSIZE - 1; j++, k += 2) {
            if (SIZE - 2 + i - k > SIZE - 1) continue;
            idx = SIZE - 2 + i - k;
            bank[i] += filter[j + 1] * data[idx];
            bank[i + 1] += filter[j + 1] * data[idx + 1];
        }
    }
}

// the first idea is to compute cross-convolution of the
// reverse filter.
void FIR_optim(float* data, float* output) {
    int k, b;
    for (int i = 0; i < 2 * FILTERSIZE; i += 2) {
        float temp1 = 0, temp2 = 0;
        k = 0;
        b = i - 2 * FILTERSIZE + 2;
        for (int j = 0; j < FILTERSIZE; j++, k += 2) {
            if (b + k < 0) continue;
            temp1 += filter_op[j] * data[b + k];
            temp2 += filter_op[j] * data[b + k + 1];
        }
        output[i] = temp1;
        output[i + 1] = temp2;
    }
    for (int i = 0; i < 2 * (FILTERSIZE - 1); i += 2) {
        output[i] += bank_for_optim[i];
        output[i + 1] += bank_for_optim[i + 1];
    }
    for (int i = 2 * FILTERSIZE; i < SIZE; i += 2) {
        float temp1 = 0, temp2 = 0;
        k = 0;
        b = i - 2 * FILTERSIZE + 2;
        for (int j = 0; j < FILTERSIZE; j++, k += 2) {
            temp1 += filter_op[j] * data[b + k];
            temp2 += filter_op[j] * data[b + k + 1];
        }
        output[i] = temp1;
        output[i + 1] = temp2;
    }
    for (int i = 0; i < 2 * (FILTERSIZE - 1); i += 2) {
        k = 0;
        b = i - 2 * FILTERSIZE + 2;
        for (int j = 0; j < FILTERSIZE - 1; j++, k += 2) {
            if (SIZE + b + k > SIZE - 1) break;
            int idx = SIZE + b + k;
            bank_for_optim[i] += filter_op[j] * data[idx];
            bank_for_optim[i + 1] += filter_op[j] * data[idx + 1];
        }
    }
}

int main() {
    std::cout << "start test" << std::endl;
    float* data = new float[SIZE];
    float* out = new float[SIZE];
    float* outOptimised = new float[SIZE];
    for (int i = 0; i < SIZE; i++) {
        data[i] = i + 1;
    }
    FIR(data, out);
    // reverse filter first;
    std::cout << std::endl;
    std::reverse(filter_op.data(), filter_op.data() + FILTERSIZE);
    FIR_optim(data, outOptimised);
    /*
    std::cout << "output" << std::endl;
    for (int i = 0; i < SIZE; i += 2) {
        std::cout << i / 2 << std::endl;
        std::cout << "normal: " << out[i] << " " << out[i + 1] << std::endl;
        std::cout << "optmised: " << outOptimised[i] << " "
                  << outOptimised[i + 1] << std::endl;
        std::cout << std::endl;
    }
    std::cout << std::endl;
    for (int i = 0; i < 2 * (FILTERSIZE - 1); i++) {
        std::cout << bank[i] << " " << bank_for_optim[i] << std::endl;
    }
    std::cout << "now we need to rerun the decimation" << std::endl;
    */

    for (int i = 0; i < 100; i++) {
        FIR(data, out);
        FIR_optim(data, outOptimised);
    }

    /*
    for (int i = 0; i < SIZE; i += 2) {
        std::cout << i / 2 << std::endl;
        std::cout << "normal: " << out[i] << " " << out[i + 1] << std::endl;
        std::cout << "optmised: " << outOptimised[i] << " "
                  << outOptimised[i + 1] << std::endl;
        std::cout << std::endl;
    }
    */
}


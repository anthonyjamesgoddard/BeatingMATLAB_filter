#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

constexpr int SIZE = 20;
constexpr int FILTERSIZE = 10;
const std::array<float, FILTERSIZE> filter{0.1, 1.1, 2.1, 3.1, 4.1,
                                           5.1, 6.1, 7.1, 8.1, 9.1};
std::array<float, FILTERSIZE> filter_op = filter;
std::array<float, 2 * (FILTERSIZE - 1)> bank = {};
std::array<float, 2 * (FILTERSIZE - 1)> bank_for_optim = {};

void FIR(float* data, float* output) {
    int k;
    // the first chunk of the filter O(FILTERSIZE*FILTERSIZE)
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
    // update the bank O(FILTERSIZE*FILTERSIZE)
    for (int i = 0; i < 2 * (FILTERSIZE - 1); i += 2) {
        k = 0;
        for (int j = 0; j < FILTERSIZE - 1; j++, k += 2) {
            if (SIZE - 2 + i - k > SIZE - 1) continue;
            bank[i] += filter[j + 1] * data[SIZE - 2 + i - k];
            bank[i + 1] += filter[j + 1] * data[SIZE - 2 + i - k + 1];
        }
    }
}
void FIR_optim(float* data, float* output) {
    int k;
    // the first chunk of the filter O(FILTERSIZE*FILTERSIZE)
    for (int i = 0; i < 2 * FILTERSIZE; i += 2) {
        float temp1 = 0, temp2 = 0;
        k = 0;
        int b = i - 2 * FILTERSIZE;
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
        int b = i - 2*FILTERSIZE;
        for (int j = 0; j < FILTERSIZE; j++, k += 2) {
            temp1 += filter_op[j] * data[b+k];
            temp2 += filter_op[j] * data[b+k + 1];
        }
        output[i] = temp1;
        output[i + 1] = temp2;
    }
    // update the bank O(FILTERSIZE*FILTERSIZE)
    for (int i = 0; i < 2 * (FILTERSIZE - 1); i += 2) {
        k = 0;
        for (int j = 0; j < FILTERSIZE - 1; j++, k += 2) {
            if (SIZE - 2 + i - k > SIZE - 1) continue;
            bank_for_optim[i] += filter_op[j + 1] * data[SIZE - 2 + i - k];
            bank_for_optim[i + 1] +=
                filter_op[j + 1] * data[SIZE - 2 + i - k + 1];
        }
    }
}
int main() {
    std::cout << "start test" << std::endl;
    float* data = new float[SIZE];
    float* out = new float[SIZE];
    float* outOptimised = new float[SIZE];
    int j = 0;
    for (int i = 0; i < SIZE / 2; i++) {
        data[j++] = i + 1;
        data[j++] = i + 1;
    }
    FIR(data, out);
    // reverse filter first;A
    std::reverse(filter_op.data(), filter_op.data() + FILTERSIZE);
    FIR_optim(data, outOptimised);
    std::cout << "output" << std::endl;
    for (int i = 0; i < SIZE; i += 2) {
        std::cout << i << " ";
        if (out[i] != outOptimised[i]) {
            std::cout << "difference encountered" << std::endl;
        }
        if (out[i + 1] != outOptimised[i + 1]) {
            std::cout << "difference encountered" << std::endl;
        }
    }
    std::cout << "now we need to rerun the decimation" << std::endl;
    FIR(data, out);
    FIR_optim(data, outOptimised);
    for (int i = 0; i < SIZE; i += 2) {
        if (out[i] != outOptimised[i]) {
            std::cout << "difference encountered" << std::endl;
        }
        if (out[i + 1] != outOptimised[i + 1]) {
            std::cout << "difference encountered" << std::endl;
        }
    }
}


#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <vector>

constexpr int SIZE = 80000;
constexpr int FILTERSIZE = 5;
const std::array<float, FILTERSIZE> filter{0.1, 1.1, 2.1, 3.1, 4.1};
std::array<float, FILTERSIZE> filter_op = filter;
std::array<float, 2 * (FILTERSIZE - 1)> bank = {};
std::array<float, (FILTERSIZE - 1)> bankr = {};
std::array<float, (FILTERSIZE - 1)> bankc = {};

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

#include <immintrin.h>
/*
 * 1)  Convert the problem into a cross-correlation
 *     (basicallly dot products) vs a convolution
 * 2)  AVX-ify it!
 */

void FIR_optim(float* datar, float* datac, float* outputr, float* outputc) {
    // ----------------------------------------
    __m256 revKernel[FILTERSIZE];
    for (size_t i = 0; i < FILTERSIZE; i++)
        revKernel[i] = _mm256_set1_ps(filter[FILTERSIZE - i - 1]);
    // ----------------------------------------

    int b;
    for (int i = 0; i < FILTERSIZE; i++) {
        float temp1 = 0, temp2 = 0;
        b = i - FILTERSIZE + 1;
        for (int j = 0; j < FILTERSIZE; j++) {
            if (b + j < 0) continue;
            temp1 += filter_op[j] * datar[b + j];
            temp2 += filter_op[j] * datac[b + j];
        }
        outputr[i] = temp1;
        outputc[i] = temp2;
    }
    for (int i = 0; i < (FILTERSIZE - 1); i++) {
        outputr[i] += bankr[i];
        outputc[i] += bankc[i];
    }
    for (int i = FILTERSIZE; i + 16 < SIZE / 2; i += 8) {
        b = i - FILTERSIZE + 1;
        __m256 res1 = _mm256_setzero_ps();
        __m256 res2 = _mm256_setzero_ps();
        for (int j = 0; j < FILTERSIZE; j++) {
            __m256 floats1 = _mm256_load_ps(&datar[b + j]);
            __m256 floats2 = _mm256_load_ps(&datac[b + j]);
            res1 = _mm256_fmadd_ps(revKernel[j], floats1, res1);
            res2 = _mm256_fmadd_ps(revKernel[j], floats2, res2);
        }
        _mm256_storeu_ps(&outputr[i], res1);
        _mm256_storeu_ps(&outputc[i], res2);
    }
    int idx;
    for (int i = 0; i < (FILTERSIZE - 1); i++) {
        b = i - FILTERSIZE + 2;
        for (int j = 0; j < FILTERSIZE - 1; j++) {
            if (SIZE + b + j > SIZE - 1) break;
            idx = SIZE + b + j;
            bankr[i] += filter_op[j] * datar[idx];
            bankc[i] += filter_op[j] * datac[idx];
        }
    }
}

int main() {
    std::cout << "start test" << std::endl;
    float* data = new float[SIZE];
    float* datar = new float[SIZE / 2];
    float* datac = new float[SIZE / 2];
    float* out = new float[SIZE];
    float* outOptimised = new float[SIZE];
    float* outOptimisedr = new float[SIZE / 2];
    float* outOptimisedc = new float[SIZE / 2];
    for (int i = 0; i < SIZE; i++) {
        data[i] = i + 1;
    }
    int j = 0;
    for (int i = 0; i < SIZE / 2; i++) {
        datar[i] = data[j++];
        datac[i] = data[j++];
    }
    FIR(data, out);
    // reverse filter first;
    std::cout << std::endl;
    std::reverse(filter_op.data(), filter_op.data() + FILTERSIZE);
    FIR_optim(datar, datac, outOptimisedr, outOptimisedc);

    std::cout << "output" << std::endl;
    /*
    j = 0;
    for (int i = 0; i < SIZE/2; i++) {
        std::cout << i << std::endl;
        std::cout << "normal: " << out[j] << " " << out[j + 1] << std::endl;
        std::cout << "optmised: " << outOptimisedr[i] << " "
                  << outOptimisedc[i] << std::endl;
        std::cout << std::endl;
        j += 2;
    }
    std::cout << std::endl;
    */
    /*
    for (int i = 0; i < 2 * (FILTERSIZE - 1); i++) {
        std::cout << bank[i] << " " << bank_for_optim[i] << std::endl;
    }
    */

    std::cout << "now we need to rerun the decimation" << std::endl;

    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    std::cout << "optimised version" << std::endl;

    for (int i = 0; i < 1000; i++) {
        FIR_optim(datar, datac, outOptimisedr, outOptimisedc);
    }

    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    std::cout << "Time difference = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       begin)
                     .count()
              << "[ms]" << std::endl;

    begin = std::chrono::steady_clock::now();
    std::cout << "vanilla version" << std::endl;

    for (int i = 0; i < 1000; i++) {
        FIR(data, out);
    }

    end = std::chrono::steady_clock::now();

    std::cout << "Time difference = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       begin)
                     .count()
              << "[ms]" << std::endl;

    /*
        for (int i = 0; i < SIZE; i += 2) {
            std::cout << "normal: " << out[i] << " " << out[i + 1] << std::endl;
            std::cout << "optmised: " << outOptimised[i] << " "
                      << outOptimised[i + 1] << std::endl;
            std::cout << std::endl;
        }
        */
}


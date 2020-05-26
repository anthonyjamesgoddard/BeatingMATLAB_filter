#include <immintrin.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <vector>

constexpr int SIZE = 200000;
constexpr int FILTERSIZE = 10;
const std::array<float, FILTERSIZE> filter{0.1, 1.1, 2.1, 3.1, 4.1,
                                           0.1, 1.1, 2.1, 3.1, 4.1};
std::array<float, FILTERSIZE> filter_op = filter;
std::array<float, 2 * (FILTERSIZE - 1)> bank = {};
std::array<float, 2 * (FILTERSIZE - 1)> bank_r = {};
std::array<float, (FILTERSIZE - 1)> bankr = {};
std::array<float, (FILTERSIZE - 1)> bankc = {};

#define EXECUTE(j)                                                        \
    {                                                                     \
        __m256 floats1 = _mm256_load_ps(&datar[b + j]);                   \
        __m256 floats2 = _mm256_load_ps(&datac[b + j]);                   \
        res1 = _mm256_add_ps(res1, _mm256_mul_ps(revKernel[j], floats1)); \
        res2 = _mm256_add_ps(res2, _mm256_mul_ps(revKernel[j], floats2)); \
    }

namespace {
__m256 revKernel[FILTERSIZE];
}

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
            idx = SIZE - 2 + i - k;
            if (idx > SIZE - 1) continue;
            bank[i] += filter[j + 1] * data[idx];
            bank[i + 1] += filter[j + 1] * data[idx + 1];
        }
    }
}

void FIR_rev(float* data, float* output) {
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
        output[i] += bank_r[i];
        output[i + 1] += bank_r[i + 1];
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
            bank_r[i] += filter_op[j] * data[idx];
            bank_r[i + 1] += filter_op[j] * data[idx + 1];
        }
    }
}

void FIR_optim(float* datar, float* datac, float* outputr, float* outputc) {
    int b;
    float temp1, temp2;
    for (int i = 0; i < FILTERSIZE; i++) {
        temp1 = 0;
        temp2 = 0;
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
    for (int i = FILTERSIZE; i < SIZE / 2; i++) {
        b = i - FILTERSIZE + 1;
        temp1 = 0;
        temp2 = 0;
        for (int j = 0; j < FILTERSIZE; j++) {
            if (b + j < 0) continue;
            temp1 += filter_op[j] * datar[b + j];
            temp2 += filter_op[j] * datac[b + j];
        }
        outputr[i] = temp1;
        outputc[i] = temp2;
    }
    int idx;
    for (int i = 0; i < (FILTERSIZE - 1); i++) {
        b = i - FILTERSIZE + 1;
        for (int j = 0; j < FILTERSIZE - 1; j++) {
            idx = SIZE / 2 + b + j;
            if (idx > SIZE / 2 - 1) break;
            bankr[i] += filter_op[j] * datar[idx];
            bankc[i] += filter_op[j] * datac[idx];
        }
    }
}

void FIR_optim_avx(float* datar, float* datac, float* outputr, float* outputc) {
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
    int i = FILTERSIZE;
    for (; i + 16 < SIZE / 2; i += 8) {
        b = i - FILTERSIZE + 1;
        __m256 res1 = _mm256_setzero_ps();
        __m256 res2 = _mm256_setzero_ps();
        EXECUTE(0);
        EXECUTE(1);
        EXECUTE(2);
        EXECUTE(3);
        EXECUTE(4);
        EXECUTE(5);
        EXECUTE(6);
        EXECUTE(7);
        EXECUTE(8);
        EXECUTE(9);
        _mm256_storeu_ps(&outputr[i], res1);
        _mm256_storeu_ps(&outputc[i], res2);
    }
    for (; i < SIZE / 2; i++) {
        b = i - FILTERSIZE + 1;
        float temp1 = 0, temp2 = 0;
        for (int j = 0; j < FILTERSIZE; j++) {
            temp1 += filter_op[j] * datar[b + j];
            temp2 += filter_op[j] * datac[b + j];
        }
        outputr[i] = temp1;
        outputc[i] = temp2;
    }
    int idx;
    for (int i = 0; i < (FILTERSIZE - 1); i++) {
        b = i - FILTERSIZE + 1;
        for (int j = 0; j < FILTERSIZE - 1; j++) {
            idx = SIZE / 2 + b + j;
            if (idx > SIZE / 2 - 1) break;
            bankr[i] += filter_op[j] * datar[idx];
            bankc[i] += filter_op[j] * datac[idx];
        }
    }
}

int main() {
    std::cout << "start test" << std::endl;
    float* data = new float[SIZE];
    float* data_r = new float[SIZE];
    float* datasplitr = new float[SIZE / 2];
    float* datasplitc = new float[SIZE / 2];
    float* datar = new float[SIZE / 2];
    float* datac = new float[SIZE / 2];
    float* out = new float[SIZE];
    float* out_r = new float[SIZE];
    float* outOptimised = new float[SIZE];
    float* outSplitr = new float[SIZE / 2];
    float* outSplitc = new float[SIZE / 2];
    float* outOptimisedr = new float[SIZE / 2];
    float* outOptimisedc = new float[SIZE / 2];

    for (int i = 0; i < SIZE; i++) {
        data[i] = i + 1;
        data_r[i] = SIZE - i;
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
    for (size_t i = 0; i < FILTERSIZE; i++)
        revKernel[i] = _mm256_set1_ps(filter[FILTERSIZE - i - 1]);
    FIR_rev(data_r, out_r);
    FIR_optim(datar, datac, outOptimisedr, outOptimisedc);
    FIR_optim_avx(datasplitr, datasplitc, outSplitr, outSplitc);
    //   std::cout << "output" << std::endl;

    j = 0;
    /*
    for (int i = 0; i < SIZE / 2; i++) {
        std::cout << i << std::endl;
        std::cout << "normal: " << out[j] << " " << out[j + 1] << std::endl;
        std::cout << "optmised: " << outOptimisedr[i] << " " << outOptimisedc[i]
                  << std::endl;
        std::cout << std::endl;
        if (out[j] != outOptimisedr[i]) {
            std::cout << i << std::endl;
            std::cout << "difference " << out[j] - outOptimisedr[i]
                      << std::endl;
        }
        if (out[j + 1] != outOptimisedc[i]) {
            std::cout << i << std::endl;
            std::cout << "difference " << out[j] - outOptimisedr[i]
                      << std::endl;
        }
        j += 2;
    }
    std::cout << std::endl;
*/
    /*
    std::cout << " bank for normal " << std::endl;
    for (int i = 0; i < 2 * (FILTERSIZE - 1); i += 2) {
        // std::cout << bank[i] << " " << bank[i + 1] << std::endl;
    }
    std::cout << " bank for optmised " << std::endl;
    for (int i = 0; i < (FILTERSIZE - 1); i++) {
        // std::cout << bankr[i] << " " << bankc[i] << std::endl;
    }
*/
    // std::cout << "now we need to rerun the decimation" << std::endl;
    // TIMING ---------------
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    std::cout << "avx version" << std::endl;

    for (int i = 0; i < 1000; i++) {
        FIR_optim_avx(datasplitr, datasplitc, outSplitr, outSplitc);
    }

    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();

    std::cout << "Time difference = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       begin)
                     .count()
              << "[ms]" << std::endl;
    // -----------------------
    begin = std::chrono::steady_clock::now();
    std::cout << "vanilla-reverse real and imaginary version" << std::endl;

    for (int i = 0; i < 1000; i++) {
        FIR_optim(datar, datac, outOptimisedr, outOptimisedc);
    }

    end = std::chrono::steady_clock::now();

    std::cout << "Time difference = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       begin)
                     .count()
              << "[ms]" << std::endl;
    // -----------------------
    begin = std::chrono::steady_clock::now();
    std::cout << "vanilla-reverse version" << std::endl;

    for (int i = 0; i < 1000; i++) {
        FIR_rev(data_r, out_r);
    }

    end = std::chrono::steady_clock::now();

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
    j = 0;
    for (int i = 0; i < SIZE / 2; i++) {
        /*
        std::cout << i << std::endl;
        std::cout << "normal: " << out[j] << " " << out[j + 1] << std::endl;
        std::cout << "optmised: " << outOptimisedr[i] << " " << outOptimisedc[i]
                  << std::endl;
        std::cout << std::endl;

        if (out[j] != outOptimisedr[i]) {
            std::cout << i << std::endl;
            std::cout << "difference " << out[j] - outOptimisedr[i]
                      << std::endl;
        }
        if (out[j + 1] != outOptimisedc[i]) {
            std::cout << i << std::endl;
            std::cout << "difference " << out[j] - outOptimisedr[i]
                      << std::endl;
        }
        j += 2;
        */
    }
}


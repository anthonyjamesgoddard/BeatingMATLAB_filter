#include <array>
#include <iostream>
#include <vector>
constexpr int SIZE = 20;
constexpr int FILTERSIZE = 5;
const std::array<float, FILTERSIZE> filter{0.1, 1.1, 2.1, 3.1, 4.1};
std::array<float, 2 * (FILTERSIZE - 1)> bank = {};

void decimate(int8_t* data, int8_t* out) {
    std::vector<float> output(SIZE);
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
    for (int i = 0; i < 2 * (FILTERSIZE-1); i += 2) {
        output[i] += bank[i];
        output[i + 1] += bank[i+1];
    }
    // this is the AVX-able part
    // the main part O(FILTERSIZE*SIZE)
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
    size_t j = 0;
    for (size_t i = 0; i < SIZE; i += 2) {
        out[j++] = output[i]+0.5;
        out[j++] = -1 * (output[i + 1]+0.5);
    }
}

int main() {
    std::cout << "start test" << std::endl;
    int8_t* data = new int8_t[SIZE];
    int8_t* out = new int8_t[SIZE];
    int j = 0;
    for (int i = 0; i < SIZE / 2; i++) {
        data[j++] = i + 1;
        data[j++] = i + 1;
    }
    decimate(data, out);
    std::cout << "bank after decimation" << std::endl;
    for (int i = 0; i < 2 * (FILTERSIZE - 1); i += 2) {
        std::cout << bank[i] << " " << bank[i + 1] << std::endl;
    }
    std::cout << "output" << std::endl;
    for (int i = 0; i < SIZE; i += 2) {
        std::cout << static_cast<int>(out[i]) << " "
                  << static_cast<int>(out[i + 1]) << std::endl;
    }
    std::cout << "now we need to rerun the decimation" << std::endl;
    decimate(data, out);
    for (int i = 0; i < SIZE; i += 2) {
        std::cout << static_cast<int>(out[i]) << " "
                  << static_cast<int>(out[i + 1]) << std::endl;
    }
}


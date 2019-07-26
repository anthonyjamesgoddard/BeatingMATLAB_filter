#include <array>
#include <iostream>
#include <vector>
constexpr int SIZE = 20;
constexpr int FILTERSIZE = 5;
const std::array<float, FILTERSIZE> filter{0.1, 1.1, 2.1, 3.1, 4.1};
std::array<float, 2*(FILTERSIZE - 1)> bank = {};

void decimate_init(int8_t* data, int8_t* out) {
    std::vector<float> output(SIZE);
    int k = 0;
    // the first chunk of the filter O(FILTERSIZE*FILTERSIZE)
    for (int i = 0; i < FILTERSIZE; i += 2) {
        float temp1 = 0, temp2 = 0;
        for (int j = 0; j < FILTERSIZE; j++, k = j + 1) {
            if (i - k < 0) break;
            temp1 += filter[j] * data[i - k];
            temp2 += filter[j] * data[i - k + 1];
        }
        output[i] = temp1;
        output[i + 1] = temp2;
    }
    // this is the AVX-able part
    // the main part O(FILTERSIZE*SIZE)
    k = 0;
    for (int i = FILTERSIZE; i < SIZE; i+=2) {
        float temp1 = 0, temp2 = 0;
        for (int j = 0; j < FILTERSIZE; j++, k = j + 1) {
            temp1 += filter[j] * data[i - k];
            temp2 += filter[j] * data[i - k + 1];
        }
        output[i] = temp1;
        output[i + 1] = temp2;
    }
    // update the bank O(FILTERSIZE*FILTERSIZE)
    k = 0;
    int offset = SIZE - FILTERSIZE;
    for (int i = 0; i < 2*(FILTERSIZE - 1); i+=2) {
        for (int j = 0; j < FILTERSIZE; j++, k = j + 1) {
            std::cout << i << " "  << offset + i - k << std::endl;
            bank[i] += filter[j] * data[offset + i - k];
            bank[i + 1] += filter[j] * data[offset + i - k + 1];
        }
    }
    size_t j = 0;
    for (size_t i = 0; i < SIZE; i += 2) {
        out[j++] = output[i];
        out[j++] = output[i + 1];
    }
}

void decimate(int8_t* data, int8_t* out) {
    std::vector<float> output(SIZE);
    int k = 0;
    for (int i = 0; i < FILTERSIZE; i += 2) {
        float temp1 = 0, temp2 = 0;
        for (int j = 0; j < FILTERSIZE; j++, k = j + 1) {
            if (i - k < 0) break;
            temp1 += filter[j] * data[i - k];
            temp2 += filter[j] * data[i - k + 1];
        }
        output[i] = temp1;
        output[i + 1] = temp2;
    }
    k = 0;
    for (int i = FILTERSIZE; i < SIZE; i++) {
        float temp1 = 0, temp2 = 0;
        for (int j = 0; j < 5; j++, k = j + 1) {
            temp1 += filter[j] * data[i - k];
            temp2 += filter[j] * data[i - k + 1];
        }
        output[i] = temp1;
        output[i + 1] = temp2;
    }
    size_t j = 0;
    for (size_t i = 0; i < SIZE; i += 2) {
        out[j++] = output[i];
        out[j++] = output[i + 1];
    }
}

int main() {
    std::cout << "start test" << std::endl;
    int8_t* data = new int8_t[20];
    int8_t* out = new int8_t[20];
    for (int i = 0; i < SIZE; i++) {
        data[i] = 1;
    }
    decimate_init(data, out);
    std::cout << "bank after decimation" << std::endl;
    for(int i = 0; i < 2*(FILTERSIZE - 1); i+=2) {
        std::cout << bank[i] << " " << bank[i+1] << std::endl;
    }
    std::cout << "output" << std::endl;
    for (int i = 0; i < SIZE; i++) {
        std::cout << static_cast<int>(out[i]) << std::endl;
    }
}


// SPDX-License-Identifier: GPL-3.0-or-later
#include "../src/reactiveTransport/reactiveWale.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct Case {
    double gradient[9];
    double delta;
    double coefficient;
};

struct Output {
    double nut;
    int valid;
};

[[noreturn]] void fail(const std::string& message)
{
    std::cerr << "WALE CUDA test failed: " << message << '\n';
    std::exit(1);
}

void cudaCheck(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) fail(std::string(operation) + ": " + cudaGetErrorString(status));
}

__global__ void evaluateKernel(const Case* cases, Output* output, int count)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) return;
    double nut = 0.0;
    const bool valid = ReactiveWale::eddyViscosity(
        cases[index].gradient, cases[index].delta, cases[index].coefficient, nut);
    output[index] = {nut, valid ? 1 : 0};
}

std::vector<Case> fixtures()
{
    std::vector<Case> result;
    result.push_back({{}, 1.0, 0.5});
    Case shear{{}, 1.0, 0.5};
    shear.gradient[1] = 3.0;
    result.push_back(shear);
    Case wall{{}, 1.0, 0.5};
    wall.gradient[1] = 1.0;
    wall.gradient[0] = 1e-5;
    wall.gradient[8] = -1e-5;
    result.push_back(wall);
    Case huge{{1.0, -0.5, 0.25, 0.75, -0.125, 0.5, -0.25, 0.375, 0.625},
        std::ldexp(1.0, -550), 0.5};
    for (double& value : huge.gradient) value = std::ldexp(value, 1000);
    result.push_back(huge);
    Case tiny{{1.0, -0.5, 0.25, 0.75, -0.125, 0.5, -0.25, 0.375, 0.625},
        std::ldexp(1.0, 550), 0.5};
    for (double& value : tiny.gradient) value = std::ldexp(value, -1000);
    result.push_back(tiny);

    unsigned state = 0x8d12e5a7u;
    for (int sample = 0; sample < 27; ++sample) {
        Case item{{}, 0.02 * (sample + 1), 0.325 + 0.01 * (sample % 5)};
        for (double& value : item.gradient) {
            state = state * 1664525u + 1013904223u;
            value = (static_cast<double>(state >> 8) / 8388608.0 - 1.0)
                * std::ldexp(1.0, (sample % 7) * 80 - 240);
        }
        result.push_back(item);
    }
    Case invalid = result.back();
    invalid.gradient[2] = std::numeric_limits<double>::infinity();
    result.push_back(invalid);
    invalid = result.back();
    invalid.gradient[2] = 1.0;
    invalid.delta = -1.0;
    result.push_back(invalid);
    return result;
}

} // namespace

int main()
{
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount < 1) fail("no CUDA device is available");
    cudaDeviceProp properties{};
    cudaCheck(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");

    const std::vector<Case> input = fixtures();
    std::vector<Output> host(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        host[i].valid = ReactiveWale::eddyViscosity(
            input[i].gradient, input[i].delta, input[i].coefficient, host[i].nut) ? 1 : 0;
    }

    Case* deviceInput = nullptr;
    Output* deviceOutput = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceInput), input.size() * sizeof(Case)),
        "cudaMalloc input");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&deviceOutput), input.size() * sizeof(Output)),
        "cudaMalloc output");
    cudaCheck(cudaMemcpy(deviceInput, input.data(), input.size() * sizeof(Case), cudaMemcpyHostToDevice),
        "cudaMemcpy input");
    evaluateKernel<<<1, 64>>>(deviceInput, deviceOutput, static_cast<int>(input.size()));
    cudaCheck(cudaGetLastError(), "evaluateKernel launch");
    cudaCheck(cudaDeviceSynchronize(), "evaluateKernel execution");
    std::vector<Output> device(input.size());
    cudaCheck(cudaMemcpy(device.data(), deviceOutput, device.size() * sizeof(Output), cudaMemcpyDeviceToHost),
        "cudaMemcpy output");
    cudaCheck(cudaFree(deviceOutput), "cudaFree output");
    cudaCheck(cudaFree(deviceInput), "cudaFree input");

    for (std::size_t i = 0; i < input.size(); ++i) {
        if (device[i].valid != host[i].valid) {
            fail("status mismatch at fixture " + std::to_string(i));
        }
        if (!host[i].valid) {
            if (!std::isnan(device[i].nut)) fail("invalid device result is not NaN");
            continue;
        }
        const double scale = std::max({1e-300, std::abs(host[i].nut), std::abs(device[i].nut)});
        if (!std::isfinite(device[i].nut)
            || std::abs(device[i].nut - host[i].nut) > 3e-14 * scale) {
            fail("numeric mismatch at fixture " + std::to_string(i));
        }
    }

    std::cout << "WALE CUDA parity tests passed on " << properties.name
              << " for " << input.size() << " bounded fixtures\n";
    return 0;
}

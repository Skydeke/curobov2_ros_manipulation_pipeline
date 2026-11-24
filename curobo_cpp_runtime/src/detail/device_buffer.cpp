// SPDX-License-Identifier: Apache-2.0
#include "detail/device_buffer.hpp"

#include <stdexcept>
#include <string>

#include "detail/errors.hpp"

namespace curobo_cpp_runtime
{
namespace detail
{

void cuda_check(const char * method, cudaError_t status)
{
  if (status != cudaSuccess) {
    fail(method, std::string("CUDA error: ") + cudaGetErrorString(status));
  }
}

DeviceBuffer::~DeviceBuffer()
{
  if (data_ != nullptr) {
    cudaFree(data_);
  }
}

void DeviceBuffer::allocate(std::size_t bytes)
{
  if (bytes == 0) {
    data_ = nullptr;
    bytes_ = 0;
    return;
  }
  if (data_ != nullptr) {
    cudaFree(data_);
  }
  cuda_check("cudaMalloc", cudaMalloc(&data_, bytes));
  bytes_ = bytes;
}

void DeviceBuffer::upload(
  const void * source, std::size_t bytes, cudaMemcpyKind kind)
{
  if (bytes == 0) {
    return;
  }
  if (data_ == nullptr || bytes > bytes_) {
    allocate(bytes);
  }
  cuda_check("cudaMemcpy", cudaMemcpy(data_, source, bytes, kind));
}

void DeviceBuffer::download(void * destination, std::size_t bytes) const
{
  if (bytes == 0) {
    return;
  }
  if (data_ == nullptr || bytes > bytes_) {
    throw std::runtime_error("DeviceBuffer::download: buffer is smaller than request");
  }
  cuda_check("cudaMemcpy", cudaMemcpy(destination, data_, bytes, cudaMemcpyDeviceToHost));
}

}  // namespace detail
}  // namespace curobo_cpp_runtime

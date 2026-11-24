// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <vector>

namespace curobo_cpp_runtime
{
namespace detail
{

void cuda_check(const char * method, cudaError_t status);

class DeviceBuffer
{
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer & operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer();

  void allocate(std::size_t bytes);

  void upload(
    const void * source, std::size_t bytes,
    cudaMemcpyKind kind = cudaMemcpyHostToDevice);

  template<typename T>
  void upload(const std::vector<T> & values)
  {
    upload(values.data(), values.size() * sizeof(T));
  }

  void download(void * destination, std::size_t bytes) const;

  template<typename T>
  void download(std::vector<T> & values) const
  {
    values.resize(bytes_ / sizeof(T));
    download(values.data(), bytes_);
  }

  void * get() {return data_;}
  const void * get() const {return data_;}
  std::size_t bytes() const {return bytes_;}

private:
  void * data_{nullptr};
  std::size_t bytes_{0};
};

}  // namespace detail
}  // namespace curobo_cpp_runtime

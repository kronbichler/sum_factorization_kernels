// This file is free software. You can use it, redistribute it, and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or (at
// your option) any later version.
//
// Small test case to check efficiency of operations
//
// Author: Martin Kronbichler, March 2021

#include <chrono>
#include <cmath>
#include <iomanip>

#include "utilities.h"
#include "vectorization.h"

//

#include "aligned_vector.h"
#include "matrix_vector_kernel.h"


template <typename Number, typename Number2 = Number>
void
test_abstract(const Number2 *__restrict matrix, const Number *input, Number *output)
{
  constexpr unsigned int size = 6;
  for (unsigned int l = 0; l < size; ++l)
    {
      // emulate operation in x direction in 3D
      for (unsigned int k = 0; k < size; ++k)
        apply_1d_matvec_kernel<size, 1, 0, true, false, Number>(
          matrix, input + l * size * size + k * size, output + l * size * size + k * size);
      // emulate operation in y direction in 3D
      for (unsigned int k = 0; k < size; ++k)
        apply_1d_matvec_kernel<size, size, 0, true, false, Number>(matrix,
                                                                   output + l * size * size + k,
                                                                   output + l * size * size + k);
    }
}


template <typename Number, typename Number2 = Number>
void
test_low_level(const Number2 *__restrict matrix, const Number *input, Number *output)
{
#if defined(__ARM_FEATURE_SVE)
  constexpr unsigned int size = 6;
  for (unsigned int l = 0; l < size; ++l)
    {
      // emulate operation in x direction in 3D
      for (unsigned int k = 0; k < size; ++k)
        {
          const Number *         in     = input + l * size * size + k * size;
          Number *               out    = output + l * size * size + k * size;
          constexpr unsigned int stride = 1;

          svfloat64_t t0  = svld1_f64(svptrue_b64(), in[0].data);
          svfloat64_t t1  = svld1_f64(svptrue_b64(), in[stride * 5].data);
          svfloat64_t xp0 = svadd_f64_z(svptrue_b64(), t0, t1);
          svfloat64_t xm0 = svsub_f64_z(svptrue_b64(), t0, t1);
          t0              = svld1_f64(svptrue_b64(), in[stride].data);
          t1              = svld1_f64(svptrue_b64(), in[stride * 4].data);
          svfloat64_t xp1 = svadd_f64_z(svptrue_b64(), t0, t1);
          svfloat64_t xm1 = svsub_f64_z(svptrue_b64(), t0, t1);
          t0              = svld1_f64(svptrue_b64(), in[stride * 2].data);
          t1              = svld1_f64(svptrue_b64(), in[stride * 3].data);
          svfloat64_t xp2 = svadd_f64_z(svptrue_b64(), t0, t1);
          svfloat64_t xm2 = svsub_f64_z(svptrue_b64(), t0, t1);
          svfloat64_t c00 = svdup_n_f64(matrix[0]);
          svfloat64_t c01 = svdup_n_f64(matrix[1]);
          svfloat64_t c02 = svdup_n_f64(matrix[2]);
          svfloat64_t c03 = svdup_n_f64(matrix[3]);
          svfloat64_t c04 = svdup_n_f64(matrix[4]);
          svfloat64_t c05 = svdup_n_f64(matrix[5]);
          svfloat64_t c06 = svdup_n_f64(matrix[6]);
          svfloat64_t c07 = svdup_n_f64(matrix[7]);
          svfloat64_t c08 = svdup_n_f64(matrix[8]);
          svfloat64_t c09 = svdup_n_f64(matrix[9]);
          svfloat64_t c10 = svdup_n_f64(matrix[10]);
          svfloat64_t c11 = svdup_n_f64(matrix[11]);
          svfloat64_t c12 = svdup_n_f64(matrix[12]);
          svfloat64_t c13 = svdup_n_f64(matrix[13]);
          svfloat64_t c14 = svdup_n_f64(matrix[14]);
          svfloat64_t c15 = svdup_n_f64(matrix[15]);
          svfloat64_t c16 = svdup_n_f64(matrix[16]);
          svfloat64_t c17 = svdup_n_f64(matrix[17]);

          svfloat64_t t2, t3, t4, t5;

          t0 = svmul_f64_z(svptrue_b64(), xp0, c00);
          t1 = svmul_f64_z(svptrue_b64(), xm0, c15);
          t2 = svmul_f64_z(svptrue_b64(), xp0, c01);
          t3 = svmul_f64_z(svptrue_b64(), xm0, c16);
          t4 = svmul_f64_z(svptrue_b64(), xp0, c02);
          t5 = svmul_f64_z(svptrue_b64(), xm0, c17);
          t0 = svmla_f64_z(svptrue_b64(), t0, xp1, c03);
          t1 = svmla_f64_z(svptrue_b64(), t1, xm1, c12);
          t2 = svmla_f64_z(svptrue_b64(), t2, xp1, c04);
          t3 = svmla_f64_z(svptrue_b64(), t3, xm1, c13);
          t4 = svmla_f64_z(svptrue_b64(), t4, xp1, c05);
          t5 = svmla_f64_z(svptrue_b64(), t5, xm1, c14);
          t0 = svmla_f64_z(svptrue_b64(), t0, xp2, c06);
          t1 = svmla_f64_z(svptrue_b64(), t1, xm2, c09);
          t2 = svmla_f64_z(svptrue_b64(), t2, xp2, c07);
          t3 = svmla_f64_z(svptrue_b64(), t3, xm2, c10);
          t4 = svmla_f64_z(svptrue_b64(), t4, xp2, c08);
          t5 = svmla_f64_z(svptrue_b64(), t5, xm2, c11);
          svst1_f64(svptrue_b64(), out[0 * stride].data, svadd_f64_z(svptrue_b64(), t0, t1));
          svst1_f64(svptrue_b64(), out[5 * stride].data, svsub_f64_z(svptrue_b64(), t0, t1));
          svst1_f64(svptrue_b64(), out[1 * stride].data, svadd_f64_z(svptrue_b64(), t2, t3));
          svst1_f64(svptrue_b64(), out[4 * stride].data, svsub_f64_z(svptrue_b64(), t2, t3));
          svst1_f64(svptrue_b64(), out[2 * stride].data, svadd_f64_z(svptrue_b64(), t4, t5));
          svst1_f64(svptrue_b64(), out[3 * stride].data, svsub_f64_z(svptrue_b64(), t4, t5));
        }

      // emulate operation in y direction in 3D
      for (unsigned int k = 0; k < size; ++k)
        {
          const Number *         in     = output + l * size * size + k;
          Number *               out    = output + l * size * size + k;
          constexpr unsigned int stride = size;

          svfloat64_t t0  = svld1_f64(svptrue_b64(), in[0].data);
          svfloat64_t t1  = svld1_f64(svptrue_b64(), in[stride * 5].data);
          svfloat64_t xp0 = svadd_f64_z(svptrue_b64(), t0, t1);
          svfloat64_t xm0 = svsub_f64_z(svptrue_b64(), t0, t1);
          t0              = svld1_f64(svptrue_b64(), in[stride].data);
          t1              = svld1_f64(svptrue_b64(), in[stride * 4].data);
          svfloat64_t xp1 = svadd_f64_z(svptrue_b64(), t0, t1);
          svfloat64_t xm1 = svsub_f64_z(svptrue_b64(), t0, t1);
          t0              = svld1_f64(svptrue_b64(), in[stride * 2].data);
          t1              = svld1_f64(svptrue_b64(), in[stride * 3].data);
          svfloat64_t xp2 = svadd_f64_z(svptrue_b64(), t0, t1);
          svfloat64_t xm2 = svsub_f64_z(svptrue_b64(), t0, t1);
          svfloat64_t c00 = svdup_n_f64(matrix[0]);
          svfloat64_t c01 = svdup_n_f64(matrix[1]);
          svfloat64_t c02 = svdup_n_f64(matrix[2]);
          svfloat64_t c03 = svdup_n_f64(matrix[3]);
          svfloat64_t c04 = svdup_n_f64(matrix[4]);
          svfloat64_t c05 = svdup_n_f64(matrix[5]);
          svfloat64_t c06 = svdup_n_f64(matrix[6]);
          svfloat64_t c07 = svdup_n_f64(matrix[7]);
          svfloat64_t c08 = svdup_n_f64(matrix[8]);
          svfloat64_t c09 = svdup_n_f64(matrix[9]);
          svfloat64_t c10 = svdup_n_f64(matrix[10]);
          svfloat64_t c11 = svdup_n_f64(matrix[11]);
          svfloat64_t c12 = svdup_n_f64(matrix[12]);
          svfloat64_t c13 = svdup_n_f64(matrix[13]);
          svfloat64_t c14 = svdup_n_f64(matrix[14]);
          svfloat64_t c15 = svdup_n_f64(matrix[15]);
          svfloat64_t c16 = svdup_n_f64(matrix[16]);
          svfloat64_t c17 = svdup_n_f64(matrix[17]);

          svfloat64_t t2, t3, t4, t5;

          t0 = svmul_f64_z(svptrue_b64(), xp0, c00);
          t1 = svmul_f64_z(svptrue_b64(), xm0, c15);
          t2 = svmul_f64_z(svptrue_b64(), xp0, c01);
          t3 = svmul_f64_z(svptrue_b64(), xm0, c16);
          t4 = svmul_f64_z(svptrue_b64(), xp0, c02);
          t5 = svmul_f64_z(svptrue_b64(), xm0, c17);
          t0 = svmla_f64_z(svptrue_b64(), t0, xp1, c03);
          t1 = svmla_f64_z(svptrue_b64(), t1, xm1, c12);
          t2 = svmla_f64_z(svptrue_b64(), t2, xp1, c04);
          t3 = svmla_f64_z(svptrue_b64(), t3, xm1, c13);
          t4 = svmla_f64_z(svptrue_b64(), t4, xp1, c05);
          t5 = svmla_f64_z(svptrue_b64(), t5, xm1, c14);
          t0 = svmla_f64_z(svptrue_b64(), t0, xp2, c06);
          t1 = svmla_f64_z(svptrue_b64(), t1, xm2, c09);
          t2 = svmla_f64_z(svptrue_b64(), t2, xp2, c07);
          t3 = svmla_f64_z(svptrue_b64(), t3, xm2, c10);
          t4 = svmla_f64_z(svptrue_b64(), t4, xp2, c08);
          t5 = svmla_f64_z(svptrue_b64(), t5, xm2, c11);
          svst1_f64(svptrue_b64(), out[0 * stride].data, svadd_f64_z(svptrue_b64(), t0, t1));
          svst1_f64(svptrue_b64(), out[5 * stride].data, svsub_f64_z(svptrue_b64(), t0, t1));
          svst1_f64(svptrue_b64(), out[1 * stride].data, svadd_f64_z(svptrue_b64(), t2, t3));
          svst1_f64(svptrue_b64(), out[4 * stride].data, svsub_f64_z(svptrue_b64(), t2, t3));
          svst1_f64(svptrue_b64(), out[2 * stride].data, svadd_f64_z(svptrue_b64(), t4, t5));
          svst1_f64(svptrue_b64(), out[3 * stride].data, svsub_f64_z(svptrue_b64(), t4, t5));
        }
    }
#endif
}


int
main(int argc, char **argv)
{
  unsigned int n_tests = 100;
  if (argc > 1)
    n_tests = std::atoi(argv[1]);

  AlignedVector<double> matrix(18);
  for (unsigned int i = 0; i < matrix.size(); ++i)
    matrix[i] = i + 1;

  AlignedVector<double> input(6 * 6 * 6);

  for (unsigned int i = 0; i < input.size(); ++i)
    input[i] = i + 1;

  AlignedVector<double> output(6 * 6 * 6);

  {
    double tmin = 1e10, tmax = 0, tavg = 0, variance = 0;
    for (unsigned int test = 0; test < 100; ++test)
      {
        auto t1 = std::chrono::system_clock::now();
        for (unsigned int t = 0; t < n_tests; ++t)
          test_abstract(matrix.begin(), input.begin(), output.begin());

        double time = std::chrono::duration<double>(std::chrono::system_clock::now() - t1).count();
        tmin        = std::min(tmin, time);
        tmax        = std::max(tmax, time);
        variance    = (test > 0 ? (double)(test - 1) / (test)*variance : 0) +
                   (time - tavg) * (time - tavg) / (test + 1);
        tavg = tavg + (time - tavg) / (test + 1);
      }
    std::cout << "Time no vectorize:   " << std::setw(12) << tavg << "  dev " << std::setw(12)
              << std::sqrt(variance) << "  min " << std::setw(12) << tmin << "  max "
              << std::setw(12) << tmax << "  Gflop/s: " << std::setw(12)
              << 1e-9 * n_tests * 2 * 6 * 6 * (6 + 6 + 12 * 2 + 6) / tmin << std::endl;
  }

  AlignedVector<VectorizedArray<double>> input_vec(input.size());
  for (unsigned int i = 0; i < input.size(); ++i)
    input_vec[i] = input[i];

  AlignedVector<VectorizedArray<double>> output_vec(output.size());

  {
    double tmin = 1e10, tmax = 0, tavg = 0, variance = 0;
    for (unsigned int test = 0; test < 100; ++test)
      {
        auto t1 = std::chrono::system_clock::now();
        for (unsigned int t = 0; t < n_tests; ++t)
          test_abstract(matrix.begin(), input_vec.begin(), output_vec.begin());

        double time = std::chrono::duration<double>(std::chrono::system_clock::now() - t1).count();
        tmin        = std::min(tmin, time);
        tmax        = std::max(tmax, time);
        variance    = (test > 0 ? (double)(test - 1) / (test)*variance : 0) +
                   (time - tavg) * (time - tavg) / (test + 1);
        tavg = tavg + (time - tavg) / (test + 1);
      }
    std::cout << "Time vectorize high: " << std::setw(12) << tavg << "  dev " << std::setw(12)
              << std::sqrt(variance) << "  min " << std::setw(12) << tmin << "  max "
              << std::setw(12) << tmax << "  Gflop/s: " << std::setw(12)
              << 1e-9 * n_tests * 2 * 6 * 6 * (6 + 6 + 12 * 2 + 6) *
                   VectorizedArray<double>::n_array_elements / tmin
              << std::endl;
    double error = 0.;
    for (unsigned int i = 0; i < output.size(); ++i)
      for (unsigned int v = 0; v < VectorizedArray<double>::n_array_elements; ++v)
        error = std::max(std::abs(output[i] - output_vec[i][v]), error);
    std::cout << "Error vectorize high: " << error << std::endl;
  }

  for (unsigned int i = 0; i < output.size(); ++i)
    output[i] = 0;
  {
    double tmin = 1e10, tmax = 0, tavg = 0, variance = 0;
    for (unsigned int test = 0; test < 100; ++test)
      {
        auto t1 = std::chrono::system_clock::now();
        for (unsigned int t = 0; t < n_tests; ++t)
          test_low_level(matrix.begin(), input_vec.begin(), output_vec.begin());

        double time = std::chrono::duration<double>(std::chrono::system_clock::now() - t1).count();
        tmin        = std::min(tmin, time);
        tmax        = std::max(tmax, time);
        variance    = (test > 0 ? (double)(test - 1) / (test)*variance : 0) +
                   (time - tavg) * (time - tavg) / (test + 1);
        tavg = tavg + (time - tavg) / (test + 1);
      }
    std::cout << "Time vectorize low:  " << std::setw(12) << tavg << "  dev " << std::setw(12)
              << std::sqrt(variance) << "  min " << std::setw(12) << tmin << "  max "
              << std::setw(12) << tmax << "  Gflop/s: " << std::setw(12)
              << 1e-9 * n_tests * 2 * 6 * 6 * (6 + 6 + 12 * 2 + 6) *
                   VectorizedArray<double>::n_array_elements / tmin
              << std::endl;
    double error = 0.;
    for (unsigned int i = 0; i < output.size(); ++i)
      for (unsigned int v = 0; v < VectorizedArray<double>::n_array_elements; ++v)
        error = std::max(std::abs(output[i] - output_vec[i][v]), error);
    std::cout << "Error vectorize low:  " << error << std::endl;
  }
}

// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Matteo Perotti <mperotti@iis.ee.ethz.ch>

#include <stdint.h>
#include <string.h>

#include "kernel/fft.h"
#include "printf.h"
#include "runtime.h"

#define MAX_NFFT 256

#define DEBUG
#undef DEBUG

extern unsigned long int NFFT;

extern dtype twiddle[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern cmplxtype twiddle_vec[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern dtype twiddle_reim[MAX_NFFT] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern cmplxtype samples[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
cmplxtype samples_copy[MAX_NFFT] __attribute__((aligned(32 * NR_LANES), section(".l2")));
cmplxtype samples_vec[MAX_NFFT] __attribute__((aligned(32 * NR_LANES), section(".l2")));
dtype buf[MAX_NFFT] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern cmplxtype gold_out[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
signed short SwapTable[MAX_NFFT] __attribute__((aligned(32 * NR_LANES), section(".l2")));

  uint8_t mask_addr_0[32] __attribute__((aligned(32 * NR_LANES), section(".l2"))) = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
  uint8_t mask_addr_1[32] __attribute__((aligned(32 * NR_LANES), section(".l2"))) = {0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF};
  uint8_t mask_addr_2[32] __attribute__((aligned(32 * NR_LANES), section(".l2"))) = {0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF};
  uint8_t mask_addr_3[32] __attribute__((aligned(32 * NR_LANES), section(".l2"))) = {0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF};
  uint8_t mask_addr_4[32] __attribute__((aligned(32 * NR_LANES), section(".l2"))) = {0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0};
  uint8_t mask_addr_5[32] __attribute__((aligned(32 * NR_LANES), section(".l2"))) = {0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC};
  uint8_t mask_addr_6[32] __attribute__((aligned(32 * NR_LANES), section(".l2"))) = {0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA};
  const uint8_t* mask_addr_vec[7] = {mask_addr_0, mask_addr_1, mask_addr_2, mask_addr_3, mask_addr_4, mask_addr_5, mask_addr_6};

  uint32_t index_ptr[128] __attribute__((aligned(32 * NR_LANES), section(".l2"))) = {0, 64, 32, 96, 16, 80, 48, 112, 8, 72, 40, 104, 24, 88, 56, 120, 4, 68, 36, 100, 20, 84, 52, 116, 12, 76, 44, 108, 28, 92, 60, 124, 2, 66, 34, 98, 18, 82, 50, 114, 10, 74, 42, 106, 26, 90, 58, 122, 6, 70, 38, 102, 22, 86, 54, 118, 14, 78, 46, 110, 30, 94, 62, 126, 1, 65, 33, 97, 17, 81, 49, 113, 9, 73, 41, 105, 25, 89, 57, 121, 5, 69, 37, 101, 21, 85, 53, 117, 13, 77, 45, 109, 29, 93, 61, 125, 3, 67, 35, 99, 19, 83, 51, 115, 11, 75, 43, 107, 27, 91, 59, 123, 7, 71, 39, 103, 23, 87, 55, 119, 15, 79, 47, 111, 31, 95, 63, 127};


// Threshold for FP numbers comparison during the final check
#define THRESHOLD 0.1
#define FABS(x) ((x < 0) ? -x : x)

int similarity_check(double a, double b, double threshold) {
  double diff = a - b;
  if (FABS(diff) > threshold)
    return 0;
  else
    return 1;
}

int main() {
  printf("\n");
  printf("=========\n");
  printf("=  FFT  =\n");
  printf("=========\n");
  printf("\n");
  printf("\n");

  printf("\n");
  printf("------------------------------------------------------------\n");
  printf("------------------------------------------------------------\n");
  printf("\n");

  printf("Radix 2 DIF FFT on %d points\n", NFFT);

  ////////////////////
  // INITIALIZATION //
  ////////////////////

  // Initialize Twiddle factors
  printf("Initializing Twiddle Factors\n");
  //SetupTwiddlesLUT(twiddle, NFFT, 0);
  // Initialize Swap Table
  printf("Initializing Swap Table\n");
  SetupR2SwapTable(SwapTable, NFFT);
  //for (int i = 0; i < NFFT; i+=2) printf("%d, ", SwapTable[i]);
  //printf("\n");
  // Initialize Inputs
  printf("Initializing Inputs for DIT\n");
  //SetupInput(In_DIT, NFFT, FFT2_SAMPLE_DYN);
  memcpy((void*) samples_copy, (void*) samples, 2 * NFFT * sizeof(dtype));
  memcpy((void*) samples_vec, (void*) samples, 2 * NFFT * sizeof(dtype));

  // Print the input
#ifdef DEBUG
  for (unsigned int i = 0; i < NFFT; ++i) {
    printf("In_DIT[%d] == %f + (%f)j\n", i, samples[i][0], samples[i][1]);
  }
  printf("\n");
#endif
/*
  for (unsigned int i = 0; i < NFFT; ++i) {
    printf("In_DIF[%d] == %f + (%f)j\n", i, samples_copy[i][0], samples_copy[i][1]);
  }
  printf("\n");
*/

  /////////////
  // DIF FFT //
  /////////////

#ifdef DEBUG
  for (int k = 0; k < 7; ++k) {
    // Run a partial bmark
    Radix2FFT_DIF_float((dtype*) samples_copy, twiddle, NFFT, k);
    // Print the partial res
    for (unsigned int i = 0; i < NFFT; ++i) {
      printf("Out_DIF_%d[%d] == %f + (%f)j\n", k, i, samples_copy[i][0], samples_copy[i][1]);
    }
    // Reset to initial situation
    memcpy((void*) samples_copy, (void*) samples, 2 * NFFT * sizeof(dtype));
  }
#else
  start_timer();
  Radix2FFT_DIF_float((dtype*) samples_copy, twiddle, NFFT, 10);
  stop_timer();
  SwapSamples(samples_copy, SwapTable, NFFT);
#endif

  printf("Initializing Inputs for DIF\n");

  // Performance metrics
  int64_t runtime = get_timer();
  printf("The DIF execution took %d cycles.\n", runtime);

  /////////////
  // DIT FFT //
  /////////////

  // Swap samples for DIT FFT
  SwapSamples(samples, SwapTable, NFFT);

/*
  printf("Swapping samples for DIT:\n\n");
  for (unsigned int i = 0; i < NFFT; ++i) {
    printf("In_DIT[%d] == %f + (%f)j\n", i, samples[i][0], samples[i][1]);
  }
*/

  start_timer();
  Radix2FFT_DIT_float((dtype*) samples, twiddle, NFFT);
  stop_timer();

  // Performance metrics
  runtime = get_timer();
  printf("The DIT execution took %d cycles.\n", runtime);

  ////////////////////
  // Vector DIF FFT //
  ////////////////////
  // Example for 16 Samples
//  uint8_t mask_addr_0[8] = {0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0};

  float* samples_reim = cmplx2reim(samples_vec, buf, NFFT);
/*
  // Print the twiddles
  for (unsigned int i = 0; i < (unsigned int) ((NFFT >> 1) * (31 - __builtin_clz(NFFT))); ++i) {
    printf("twiddle_vec[%d] == %f + j%f\n", i, (twiddle_vec[i])[0],  (twiddle_vec[i])[1]);
  }
*/
  float* twiddle_reim = cmplx2reim(twiddle_vec, buf, ((NFFT >> 1) * (31 - __builtin_clz(NFFT))));
/*
  // Print the twiddles
  for (unsigned int i = 0; i < (unsigned int) ((NFFT >> 1) * (31 - __builtin_clz(NFFT))); ++i) {
    printf("twiddle_re[%d] == %f\n", i, twiddle_reim[i]);
  }
  for (unsigned int i = 0; i < (unsigned int) ((NFFT >> 1) * (31 - __builtin_clz(NFFT))); ++i) {
    printf("twiddle_im[%d] == %f\n", i, twiddle_reim[i + ((NFFT >> 1) * (31 - __builtin_clz(NFFT)))]);
  }
*/
  fft_r2dif_vec(samples_reim, samples_reim + NFFT,
                twiddle_reim, twiddle_reim + ((NFFT >> 1) * (31 - __builtin_clz(NFFT))),
                mask_addr_vec, index_ptr, NFFT);

/*
  // Print the results
  for (unsigned int i = 0; i < NFFT; ++i) {
    printf("Out_DIT[%d] == %f + (%f)j\n", i, samples[i][0], samples[i][1]);
  }
*/

  printf("\n");
  for (unsigned int i = 0; i < NFFT; ++i) {
    printf("Out_DIF[%d] == %f + (%f)j\n", i, samples_copy[i][0], samples_copy[i][1]);
  }

  printf("\n");
  // Print the results
  for (unsigned int i = 0; i < NFFT; ++i) {
    printf("Out_vec_DIF[%d] == %f + (%f)j\n", i, samples_reim[i], samples_reim[i+NFFT]);
  }
/*
  printf("\n");
  for (unsigned int i = 0; i < NFFT; ++i) {
    printf("gold_out[%d] == %f + (%f)j\n", i , gold_out[i][0], gold_out[i][1]);
  }
*/

  for (unsigned int i = 0; i < NFFT; ++i) {
    if (!similarity_check(samples_reim[i], gold_out[i][0], THRESHOLD)) printf("Real part error at index %d\n", i);
    if (!similarity_check(samples_reim[i], gold_out[i][0], THRESHOLD)) printf("Img part error at index %d\n", i);
  }

  return 0;
}

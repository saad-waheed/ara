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
//#undef DEBUG

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
  // Initialize Inputs
  printf("Initializing Inputs for DIT\n");
  //SetupInput(In_DIT, NFFT, FFT2_SAMPLE_DYN);
  memcpy((void*) samples_copy, (void*) samples, 2 * NFFT * sizeof(dtype));
  memcpy((void*) samples_vec, (void*) samples, 2 * NFFT * sizeof(dtype));

  // Print the input
  for (unsigned int i = 0; i < NFFT; ++i) {
    printf("In_DIT[%d] == %f + (%f)j\n", i, samples[i][0], samples[i][1]);
  }
  printf("\n");
/*
  for (unsigned int i = 0; i < NFFT; ++i) {
    printf("In_DIF[%d] == %f + (%f)j\n", i, samples_copy[i][0], samples_copy[i][1]);
  }
  printf("\n");
*/
  // Swap samples for DIT FFT
  SwapSamples(samples, SwapTable, NFFT);

/*
  printf("Swapping samples for DIT:\n\n");
  for (unsigned int i = 0; i < NFFT; ++i) {
    printf("In_DIT[%d] == %f + (%f)j\n", i, samples[i][0], samples[i][1]);
  }
*/

  /////////////
  // DIT FFT //
  /////////////

  start_timer();
  Radix2FFT_DIT_float((dtype*) samples, twiddle, NFFT);
  stop_timer();

  // Performance metrics
  int64_t runtime = get_timer();
  printf("The DIT execution took %d cycles.\n", runtime);

  /////////////
  // DIF FFT //
  /////////////

  start_timer();
  Radix2FFT_DIF_float((dtype*) samples_copy, twiddle, NFFT, 2);
  stop_timer();
#ifndef DEBUG
  SwapSamples(samples_copy, SwapTable, NFFT);
#endif

  printf("Initializing Inputs for DIF\n");

  // Performance metrics
  runtime = get_timer();
  printf("The DIF execution took %d cycles.\n", runtime);

  ////////////////////
  // Vector DIF FFT //
  ////////////////////
  // Example for 16 Samples
//  uint8_t mask_addr_0[8] = {0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0};
  uint8_t mask_addr_0[8] = {0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0};
  uint8_t mask_addr_1[8] = {0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC};
  uint8_t mask_addr_2[8] = {0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA};
  const uint8_t* mask_addr_vec[3] = {mask_addr_0, mask_addr_1, mask_addr_2};

  float* samples_reim = cmplx2reim(samples_vec, buf, NFFT);
  // Print the twiddles
  for (unsigned int i = 0; i < (unsigned int) ((NFFT >> 1) * (31 - __builtin_clz(NFFT))); ++i) {
    printf("twiddle_vec[%d] == %f + j%f\n", i, (twiddle_vec[i])[0],  (twiddle_vec[i])[1]);
  }
  float* twiddle_reim = cmplx2reim(twiddle_vec, buf, ((NFFT >> 1) * (31 - __builtin_clz(NFFT))));
  // Print the twiddles
  for (unsigned int i = 0; i < (unsigned int) ((NFFT >> 1) * (31 - __builtin_clz(NFFT))); ++i) {
    printf("twiddle_re[%d] == %f\n", i, twiddle_reim[i]);
  }
  for (unsigned int i = 0; i < (unsigned int) ((NFFT >> 1) * (31 - __builtin_clz(NFFT))); ++i) {
    printf("twiddle_im[%d] == %f\n", i, twiddle_reim[i + ((NFFT >> 1) * (31 - __builtin_clz(NFFT)))]);
  }
  fft_r2dif_vec(samples_reim, samples_reim + NFFT,
                twiddle_reim, twiddle_reim + ((NFFT >> 1) * (31 - __builtin_clz(NFFT))),
                mask_addr_vec, NFFT);

/*
  // Print the results
  for (unsigned int i = 0; i < NFFT; ++i) {
    printf("Out_DIT[%d] == %f + (%f)j\n", i, samples[i][0], samples[i][1]);
  }
*/
///*
  printf("\n");
  for (unsigned int i = 0; i < NFFT; ++i) {
    printf("Out_DIF[%d] == %f + (%f)j\n", i, samples_copy[i][0], samples_copy[i][1]);
  }
//*/
  printf("\n");
  // Print the results
  for (unsigned int i = 0; i < NFFT; ++i) {
    printf("Out_vec_DIF[%d] == %f + (%f)j\n", i, samples_reim[i], samples_reim[i+NFFT]);
  }
  printf("\n");
  for (unsigned int i = 0; i < NFFT; ++i) {
    printf("gold_out[%d] == %f + (%f)j\n", i , gold_out[i][0], gold_out[i][1]);
  }

  return 0;
}

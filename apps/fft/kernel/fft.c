// Copyright 2022 ETH Zurich and University of Bologna.
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
//
// Author: Giuseppe Tagliavini

#include <stdlib.h>
#include <math.h>

#include "fft.h"

/////////////////
// FIXED-POINT //
/////////////////

/*
   Radix-2 Decimated in Time FFT. Input have to be digitally-reversed, output is naturally ordered.
   First stage uses the fact that twiddles are all (1, 0)
*/
void Radix2FFT_DIT(signed short *__restrict__ Data, signed short *__restrict__ Twiddles, int N_FFT2)

{
  // int iLog2N  = log2(N_FFT2);
  int iLog2N  = 31 - __builtin_clz(N_FFT2);
  int iCnt1, iCnt2, iCnt3,
            iQ,    iL,    iM,
            iA,    iB;
  v2s *CoeffV = (v2s *) Twiddles;
  v2s *DataV  = (v2s *) Data;

      iL = N_FFT2 >> 1; iM = 1; iA = 0;
  /* First Layer: W = (1, 0) */
        for (iCnt3 = 0; iCnt3 < (N_FFT2>>1); iCnt3++) {
    v2s Tmp;
    iB = iA + iM;
    Tmp = DataV[iB];
    DataV[iB] = (DataV[iA] - Tmp); //  >> (v2s) {FFT2_SCALEDOWN, FFT2_SCALEDOWN};
    DataV[iA] = (DataV[iA] + Tmp); //  >> (v2s) {FFT2_SCALEDOWN, FFT2_SCALEDOWN};
                 iA = iA + 2;
  }
      iL >>= 1; iM <<= 1;

      for (iCnt1 = 1; iCnt1 < iLog2N; ++iCnt1) {
          iQ = 0;
          for (iCnt2 = 0; iCnt2 < iM; ++iCnt2) {
                v2s W = CoeffV[iQ];
                iA = iCnt2;
                for (iCnt3 = 0; iCnt3 < iL; iCnt3++) {
                    v2s Tmp, Tmp1;
                    iB = iA + iM;
                    Tmp = cplxmuls(DataV[iB], W);
                    Tmp1 = DataV[iA];

                    DataV[iB] = (Tmp1 - Tmp) >> (v2s) {FFT2_SCALEDOWN, FFT2_SCALEDOWN};
                    DataV[iA] = (Tmp1 + Tmp) >> (v2s) {FFT2_SCALEDOWN, FFT2_SCALEDOWN};
                    iA = iA + 2 * iM;
                }
                iQ += iL;
          }
          iL >>= 1;
          iM <<= 1;
      }
}

void Radix2FFT_DIF(signed short *__restrict__ Data, signed short *__restrict__ Twiddles, int N_FFT2)
{
  int iLog2N  = 31 - __builtin_clz(N_FFT2);
  int iCnt1, iCnt2, iCnt3,
            iQ,    iL,    iM,
            iA,    iB;
  v2s *CoeffV = (v2s *) Twiddles;
  v2s *DataV  = (v2s *) Data;

  iL = 1;
  iM = N_FFT2 / 2;

    for (iCnt1 = 0; iCnt1 < (iLog2N-1); iCnt1++) {
      iQ = 0;
      for (iCnt2 = 0; iCnt2 < iM; iCnt2++) {
        v2s W = CoeffV[iQ];
        iA = iCnt2;
        for (iCnt3 = 0; iCnt3 < iL; iCnt3++) {
          v2s Tmp;
          iB = iA + iM;
          Tmp = DataV[iA] - DataV[iB];
          DataV[iA] = (DataV[iA] + DataV[iB]) >> (v2s) {FFT2_SCALEDOWN, FFT2_SCALEDOWN};
          DataV[iB] = cplxmulsdiv2(Tmp, W);
          iA = iA + 2 * iM;
        }
        iQ += iL;
      }
      iL <<= 1;
      iM >>= 1;
    }
    iA = 0;

    /* Last Layer: W = (1, 0) */
    for (iCnt3 = 0; iCnt3 < (N_FFT2>>1); iCnt3++) {
      v2s Tmp;
      iB = iA + 1;
      Tmp = (DataV[iA] - DataV[iB]);
      DataV[iA] = (DataV[iA] + DataV[iB]);
      DataV[iB] = Tmp;
      iA = iA + 2;
    }
}

static inline v2s cplxmuls(v2s x, v2s	y)
{
   return (v2s) {(signed short) ((((int) (x)[0]*(int) (y)[0]) - ((int) (x)[1]*(int) (y)[1]))>>15), (signed short) ((((int) (x)[0]*(int) (y)[1]) + ((int) (x)[1]*(int) (y)[0]))>>15)};
}

static inline v2s cplxmulsdiv2(v2s x, v2s	y)
{
   return (v2s){((signed short) ((((int) (x)[0]*(int) (y)[0]) - ((int) (x)[1]*(int) (y)[1]))>>15))>>1, ((signed short) ((((int) (x)[0]*(int) (y)[1]) + ((int) (x)[1]*(int) (y)[0]))>>15))>>1};
}

/* Setup twiddles factors */
void SetupTwiddlesLUT(signed short *Twiddles, int Nfft, int Inverse)
{
  int i;
  v2s *P_Twid = (v2s *) Twiddles;
  /* Radix 4: 3/4 of the twiddles
     Radix 2: 1/2 of the twiddles
  */
  if (Inverse) {
    float Theta = (2*M_PI)/Nfft;
    for (i=0; i<Nfft; i++) {
      float Phi = Theta*i;
      P_Twid[i] = (v2s) {(short int) (cos(Phi)*((1<<FFT_TWIDDLE_DYN)-1)),
             (short int) (sin(Phi)*((1<<FFT_TWIDDLE_DYN)-1))};
      // Twiddles[2*i  ] = (short int) (cos(Phi)*((1<<FFT_TWIDDLE_DYN)-1));
      // Twiddles[2*i+1] = (short int) (sin(Phi)*((1<<FFT_TWIDDLE_DYN)-1));
    }
  } else {
    float Theta = (2*M_PI)/Nfft;
    for (i=0; i<Nfft; i++) {
      float Phi = Theta*i;
      P_Twid[i] = (v2s) {(short int) (cos(-Phi)*((1<<FFT_TWIDDLE_DYN)-1)),
             (short int) (sin(-Phi)*((1<<FFT_TWIDDLE_DYN)-1))};
      // Twiddles[2*i  ] = (short int) (cos(-Phi)*((1<<FFT_TWIDDLE_DYN)-1));
      // Twiddles[2*i+1] = (short int) (sin(-Phi)*((1<<FFT_TWIDDLE_DYN)-1));
    }
  }
}

/* Reorder from natural indexes to digitally-reversed one. Uses a pre computed LUT */

void SwapSamples (vtype *__restrict__ Data, short *__restrict__ SwapTable, int Ni)
{
  int i;

  for (i = 0; i < Ni; i++) {
    vtype S = Data[i];
    int SwapIndex = SwapTable[i];
    if (i < SwapIndex) {
      Data[i] = Data[SwapIndex]; Data[SwapIndex] = S;
    }
  }
}

void SetupR2SwapTable (short int *SwapTable, int Ni)
{
  int i, j, iL, iM;
  // int Log2N  = log2(Ni);
  int Log2N = 31 - __builtin_clz(Ni);
      iL = Ni / 2;
      iM = 1;
      SwapTable[0] = 0;
  for (i = 0; i < Log2N; ++i) {
          for (j = 0; j < iM; ++j) SwapTable[j + iM] = SwapTable[j] + iL;
          iL >>= 1; iM <<= 1;
      }
}

void  __attribute__ ((__noinline__)) SetupInput(signed short *In, int N, int Dyn)
{
        unsigned int i, j;
/*
        float Freq_Step[] = {
                2*M_PI/18.0,
                2*M_PI/67.0,
                2*M_PI/49.0,
                2*M_PI/32.0
        };
*/
        float Freq_Step[] = {
                2*M_PI/(N/10.0),
        };
        for (i=0; i < (unsigned)N; i++) {
                float sum = 0.0;
                for (j=0; j<sizeof(Freq_Step)/sizeof(float); j++) {
                        sum += sinf(i*Freq_Step[j]);
                }
                In[2*i] = (short) ((sum/(sizeof(Freq_Step)/sizeof(float)))*((1<<Dyn) -1));
                In[2*i+1] = In[2*i];
        }
}


////////////////////
// Floating-Point //
////////////////////

static inline vtype cplxmuls_float(vtype x, vtype y)
{
   return (vtype) {(x)[0] * (y)[0] - (x)[1] * (y)[1], (x)[0] * (y)[1] + (x)[1] * (y)[0]};
}

/*
   Radix-2 Decimated in Time FFT. Input have to be digitally-reversed, output is naturally ordered.
   First stage uses the fact that twiddles are all (1, 0)
*/
void Radix2FFT_DIT_float(dtype *__restrict__ Data, dtype *__restrict__ Twiddles, int N_FFT2)
{
  // int iLog2N  = log2(N_FFT2);
  int iLog2N  = 31 - __builtin_clz(N_FFT2);
  int iCnt1, iCnt2, iCnt3,
            iQ,    iL,    iM,
            iA,    iB;
  vtype *CoeffV = (vtype *) Twiddles;
  vtype *DataV  = (vtype *) Data;

      iL = N_FFT2 >> 1; iM = 1; iA = 0;
  /* First Layer: W = (1, 0) */
        for (iCnt3 = 0; iCnt3 < (N_FFT2>>1); iCnt3++) {
    vtype Tmp;
    iB = iA + iM;
    Tmp = DataV[iB];
    DataV[iB] = (DataV[iA] - Tmp);
    DataV[iA] = (DataV[iA] + Tmp);
                 iA = iA + 2;
  }
      iL >>= 1; iM <<= 1;

      for (iCnt1 = 1; iCnt1 < iLog2N; ++iCnt1) {
          iQ = 0;
          for (iCnt2 = 0; iCnt2 < iM; ++iCnt2) {
                vtype W = CoeffV[iQ];
                iA = iCnt2;
                for (iCnt3 = 0; iCnt3 < iL; iCnt3++) {
                    vtype Tmp, Tmp1;
                    iB = iA + iM;
                    Tmp = cplxmuls_float(DataV[iB], W);
                    Tmp1 = DataV[iA];

                    DataV[iB] = Tmp1 - Tmp;
                    DataV[iA] = Tmp1 + Tmp;
                    iA = iA + 2 * iM;
                }
                iQ += iL;
          }
          iL >>= 1;
          iM <<= 1;
      }
}

void Radix2FFT_DIF_float(dtype *__restrict__ Data, dtype *__restrict__ Twiddles, int N_FFT2)
{
  int iLog2N  = 31 - __builtin_clz(N_FFT2);
  int iCnt1, iCnt2, iCnt3,
            iQ,    iL,    iM,
            iA,    iB;
  vtype *CoeffV = (vtype *) Twiddles;
  vtype *DataV  = (vtype *) Data;

  iL = 1;
  iM = N_FFT2 / 2;

    for (iCnt1 = 0; iCnt1 < (iLog2N-1); iCnt1++) {
      iQ = 0;
      for (iCnt2 = 0; iCnt2 < iM; iCnt2++) {
        vtype W = CoeffV[iQ];
        iA = iCnt2;
        for (iCnt3 = 0; iCnt3 < iL; iCnt3++) {
          vtype Tmp;
          iB = iA + iM;
          Tmp = DataV[iA] - DataV[iB];
          DataV[iA] = DataV[iA] + DataV[iB];
          DataV[iB] = cplxmuls_float(Tmp, W);
          iA = iA + 2 * iM;
        }
        iQ += iL;
      }
      iL <<= 1;
      iM >>= 1;
    }
    iA = 0;

    /* Last Layer: W = (1, 0) */
    for (iCnt3 = 0; iCnt3 < (N_FFT2>>1); iCnt3++) {
      vtype Tmp;
      iB = iA + 1;
      Tmp = (DataV[iA] - DataV[iB]);
      DataV[iA] = (DataV[iA] + DataV[iB]);
      DataV[iB] = Tmp;
      iA = iA + 2;
    }
}

////////////////////
// Vectorial code //
////////////////////

#define vftype vfloat32m1_t
#define ftype float

// First implementation. LMUL == 1
// This implementation works if n_fft < VLMAX for a fixed vsew
void fft_r2dit_vec(const float* samples, const float* twiddles, size_t n_fft) {

  size_t vl = n_fft/2;
  unsigned int log2_nfft= 31 - __builtin_clz(n_fft);
  vftype upper_wing_re, upper_wing_im;
  vftype lower_wing_re, lower_wing_im;
  vftype twiddle_re, twiddle_im;
  vftype vbuf_re, vbuf_im;

  // Use undisturbed policy
  vsetvl_e32m1(vl);

  // Load the values from memory avoiding scatter/gather.
  // These values are not ordered, but the upper/lower wings of
  // each butterfly are aligned.
  // It is more convenient to permutate the output of this stage
  // instead of the inputs, so let's go with the first butterfly!
  // If real/img parts are consecutive in memory, it's possible to
  // load/store segment to divide in two registers.

  upper_wing_re = vle32_v_f32m1(samples_re          , vl);
  upper_wing_im = vle32_v_f32m1(samples_im          , vl);
  lower_wing_re = vle32_v_f32m1(samples_re + n_fft/2, vl);
  lower_wing_im = vle32_v_f32m1(samples_im + n_fft/2, vl);

  // The first twiddle factors are all the same, no need to use a vector for those

  ///////////////////////////
  // First butterfly stage //
  ///////////////////////////

  // 1) Multiply lower wing for the twiddle factor
  vbuf_re       = cmplx_mul_re_vf(lower_wing_re, lower_wing_im, twiddles_re[0], twiddles_im[0]);
  lower_wing_im = cmplx_mul_im_vf(lower_wing_re, lower_wing_im, twiddles_re[0], twiddles_im[0]);
  lower_wing_re = vbuf; // Just for the label. Verify that there is no actual copy of this vector!
  // 2) Get the upper wing output
  vbuf_re = vfadd_vv_f32m1(upper_wing_re, lower_wing_re, vl);
  vbuf_im = vfadd_vv_f32m1(upper_wing_im, lower_wing_im, vl);
  // 3) Get the lower wing output
  lower_wing_re = vfsub_vv_f32m1(upper_wing_re, lower_wing_re, vl);
  lower_wing_im = vfsub_vv_f32m1(upper_wing_im, lower_wing_im, vl);
  // Copy labels
  upper_wing_re = vbuf_re;
  upper_wing_im = vbuf_im;

  /////////////////////////////
  // First permutation stage //
  /////////////////////////////

  // First permutation stage to reorder semi-processed samples
  // Act on vstart and vl + undisturbed policy instead of using slide/mask units
  vslidedown_vi_f32m1 (vbuf_re, upper_wing_re, vl/2, vl/2);
  vslidedown_vi_f32m1 (vbuf_im, upper_wing_im, vl/2, vl/2);
  vslideup_vi_f32m1 (upper_wing_re, lower_wing_re, vl/2, vl/2);
  vslideup_vi_f32m1 (upper_wing_im, lower_wing_im, vl/2, vl/2);
  lower_wing_re = vmv_v_v_f32m1 (vbuf_re, vl/2);
  lower_wing_im = vmv_v_v_f32m1 (vbuf_im, vl/2);

  // From here on, the sample layout follows a normal DIT one

  // Butterfly until the end
  for (int i = 1; i < log2_nfft; ++i) {
    // Bump the twiddle pointers. TODO: MODIFY ME
    twiddles_re++; // Placeholder code
    twiddles_im++; // Placeholder code

    // Load twiddle factors
    twiddle_re = vle32_v_f32m1(twiddles_re, vl);
    twiddle_im = vle32_v_f32m1(twiddles_im, vl);

    // 1) Multiply lower wing for the twiddle factors
    vbuf_re       = cmplx_mul_re_vv(lower_wing_re, lower_wing_im, twiddle_re, twiddle_im);
    lower_wing_im = cmplx_mul_im_vv(lower_wing_re, lower_wing_im, twiddle_re, twiddle_im);
    lower_wing_re = vbuf; // Just for the label. Verify that there is no actual copy of this vector!
    // 2) Get the upper wing output
    vbuf_re = vfadd_vv_f32m1(upper_wing_re, lower_wing_re, vl);
    vbuf_im = vfadd_vv_f32m1(upper_wing_im, lower_wing_im, vl);
    // 3) Get the lower wing output
    lower_wing_re = vfsub_vv_f32m1(upper_wing_re, lower_wing_re, vl);
    lower_wing_im = vfsub_vv_f32m1(upper_wing_im, lower_wing_im, vl);
    // Copy labels
    upper_wing_re = vbuf_re;
    upper_wing_im = vbuf_im;

    // Permutation stage

  }

  // Store the result to memory
  store_output_values();
}


// First implementation. LMUL == 1
// This implementation works if n_fft < VLMAX for a fixed vsew
// Current implementation does not make use of segment memory ops and keeps
// real and img parts in two different separated memory locations
// This will be changed as soon as Ara supports segmented mem ops
void fft_r2dif_vec(const float* samples_re,  const float* samples_im,
                   const float* twiddles_re, const float* twiddles_im, size_t n_fft) {

  // vl of the vectors (each vector contains half of the samples)
  size_t orig_vl = n_fft/2;
  size_t vl = orig_vl;
  unsigned int log2_nfft= 31 - __builtin_clz(n_fft);
  vftype upper_wing_re, upper_wing_im;
  vftype lower_wing_re, lower_wing_im;
  vftype twiddle_re, twiddle_im;
  vftype vbuf_re, vbuf_im;

  // Use undisturbed policy
  vsetvl_e32m1(vl);

  //////////////////////
  // Mask Preparation //
  //////////////////////

  // Prepare the first mask vector to be used in the permutations
  // VLSU and VALU can work separately
  mask_vec = vmv_v_x_i32m1 (0, vl);
  mask_vec = vmv_v_x_i32m1 (0xFFFFFFFF, vl/2);
  mask_vec_buf = vmv_v_x_i32m1 (0, vl);

  ///////////////////////////////
  // LOAD samples and twiddles /
  ///////////////////////////////

  // If real/img parts are consecutive in memory, it's possible to
  // load/store segment to divide in two registers.
  // Ara does not support these instructions now, so we will hypothesize
  // different mem locations
  upper_wing_re = vle32_v_f32m1(samples_re     , vl);
  lower_wing_re = vle32_v_f32m1(samples_re + vl, vl);
  upper_wing_im = vle32_v_f32m1(samples_im     , vl);
  lower_wing_im = vle32_v_f32m1(samples_im + vl, vl);

  // Load twiddle factors
  twiddle_re = vle32_v_f32m1(twiddles_re, vl);
  twiddle_im = vle32_v_f32m1(twiddles_im, vl);

  ///////////////////////////
  // First butterfly stage //
  ///////////////////////////

  // 1) Get the upper wing output
  vbuf_re = vfadd_vv_f32m1(upper_wing_re, lower_wing_re, vl);
  vbuf_im = vfadd_vv_f32m1(upper_wing_im, lower_wing_im, vl);
  // 2) Get the lower wing output
  lower_wing_re = vfsub_vv_f32m1(upper_wing_re, lower_wing_re, vl);
  lower_wing_im = vfsub_vv_f32m1(upper_wing_im, lower_wing_im, vl);
  // Copy labels
  upper_wing_re = vbuf_re;
  upper_wing_im = vbuf_im;
  // 3) Multiply lower wing for the twiddle factor
  vbuf_re       = cmplx_mul_re_vv(lower_wing_re, lower_wing_im, twiddle_re, twiddle_im);
  lower_wing_im = cmplx_mul_im_vv(lower_wing_re, lower_wing_im, twiddle_re, twiddle_im);
  lower_wing_re = vbuf; // Just for the label. Verify that there is no actual copy of this vector!

  /////////////////////////////
  // First permutation stage //
  /////////////////////////////

  // Create the current mask level
  vslideup_vi_f32m1(mask_vec_buf, mask_vec, vl/4, vl);
  mask_vec = vmxor_mm_b1(mask_vec, mask_vec_buf, vl);
  mask_vec_buf = vmnot_mm_b1(mask_buf, mask_vec, vl);

  // Permutate the numbers
  // The first permutation is easier (just halving, no masks needed)
  vslidedown_vi_f32m1(mask_vec_buf, vbuf_re, upper_wing_re, vl/2, vl/2);
  vslidedown_vi_f32m1(mask_vec_buf, vbuf_im, upper_wing_im, vl/2, vl/2);
  vslideup_vi_f32m1(upper_wing_re, lower_wing_re, vl/2, vl/2);
  vslideup_vi_f32m1(upper_wing_im, lower_wing_im, vl/2, vl/2);
  lower_wing_re = vmv_v_v_f32m1(vbuf_re, vl/2);
  lower_wing_im = vmv_v_v_f32m1(vbuf_im, vl/2);

  // Butterfly until the end
  for (int i = 1; i < log2_nfft; ++i) {
    // Bump the twiddle pointers.
    twiddles_re += orig_vl;
    twiddles_im += orig_vl;

    // Load twiddle factors
    twiddle_re = vle32_v_f32m1(twiddles_re, orig_vl);
    twiddle_im = vle32_v_f32m1(twiddles_im, orig_vl);

    // HALVE vl
    vl /= 2;

    // Create the current mask level
    vslideup_vi_f32m1(mask_vec_buf, mask_vec, 0, vl/2);
    mask_vec = vmxor_mm_b1(mask_vec, mask_vec_buf, vl);
    mask_vec_buf = vmnot_mm_b1(mask_buf, mask_vec, vl);

    // 1) Get the upper wing output
    vbuf_re = vfadd_vv_f32m1(upper_wing_re, lower_wing_re, vl);
    vbuf_im = vfadd_vv_f32m1(upper_wing_im, lower_wing_im, vl);
    // 2) Get the lower wing output
    lower_wing_re = vfsub_vv_f32m1(upper_wing_re, lower_wing_re, vl);
    lower_wing_im = vfsub_vv_f32m1(upper_wing_im, lower_wing_im, vl);
    // Copy labels
    upper_wing_re = vbuf_re;
    upper_wing_im = vbuf_im;
    // 3) Multiply lower wing for the twiddle factor
    vbuf_re       = cmplx_mul_re_vv(lower_wing_re, lower_wing_im, twiddle_re, twiddle_im);
    lower_wing_im = cmplx_mul_im_vv(lower_wing_re, lower_wing_im, twiddle_re, twiddle_im);
    lower_wing_re = vbuf; // Just for the label. Verify that there is no actual copy of this vector!

    // Different permutation for the last round
    if (i != log2_nfft - 1) {
      // Permutate the numbers
      vslidedown_vi_f32m1_m(mask_vec_buf, vbuf_re, upper_wing_re, vl/2, vl/2);
      vslidedown_vi_f32m1_m(mask_vec_buf, vbuf_im, upper_wing_im, vl/2, vl/2);
      vslideup_vi_f32m1(upper_wing_re, lower_wing_re, vl/2, vl/2);
      vslideup_vi_f32m1(upper_wing_im, lower_wing_im, vl/2, vl/2);
      lower_wing_re = vmv_v_v_f32m1_m(mask_vec, vbuf_re, vl/2);
      lower_wing_im = vmv_v_v_f32m1_m(mask_vec, vbuf_im, vl/2);
    }
  }

  // Store the result to memory
  // Reorder the results: rotate, mask, mix
  // Last round of permutation
  vslidedown_vi_f32m1(mask_vec_buf, vbuf_re, upper_wing_re, 0, orig_vl/2);
  vslideup_vi_f32m1_m(mask_vec, upper_wing_re, lower_wing_re, vl/2, vl/2);;
  lower_wing_re = vmv_v_v_f32m1_m(mask_vec, vbuf_re, vl/2);

  // Store (segmented if RE and IM are separated!)
  vse32_v_f32m1(output_re, lower_wing_re, orig_vl);
  vse32_v_f32m1(ourput_im, lower_wing_im, orig_vl);
}

// Vector - Scalar
vftype cmplx_mul_re_vf(vftype v0_re, vftype v0_im, ftype f1_re, ftype f1_im) {
  vftype vbuf;

  vbuf = vfmul_vf_f32m1(v0_re, f1_re, vl);
  return vfnmsac_vf_f32m1(vbuf, v0_im, f1_im, vl);
}

vftype cmplx_mul_im_vf(vftype v0_re, vftype v0_im, ftype f1_re, ftype f1_im) {
  vftype vbuf;

  vbuf = vfmul_vf_f32m1(v0_re, f1_im, vl);
  return vfmacc_vv_f32m1(vbuf, v0_im, f1_re, vl);
}

// Vector - Vector
vftype cmplx_mul_re_vv(vftype v0_re, vftype v0_im, vftype f1_re, vftype f1_im) {
  vftype vbuf;

  vbuf = vfmul_vf_f32m1(v0_re, f1_re, vl);
  return vfnmsac_vf_f32m1(vbuf, v0_im, f1_im, vl);
}

vftype cmplx_mul_im_vv(vftype v0_re, vftype v0_im, vftype f1_re, vftype f1_im) {
  vftype vbuf;

  vbuf = vfmul_vf_f32m1(v0_re, f1_im, vl);
  return vfmacc_vv_f32m1(vbuf, v0_im, f1_re, vl);
}

// Ancillary function to divide real and imaginary parts
// sizeof(vtype) == 2*sizeof(dtype)
float* void cmplx2reim(vtype* cmplx, dtype* buf, size_t len) {

  float* cmplx_flat_ptr = (float*) cmplx;

  // Divide the real and img parts
  for (i = 0; i < len; ++i) {
    // Backup the img parts
    buf = cmplx[i][1];
  }
  for (i = 0; i < len; ++i) {
    // Save the real parts
    // No RAW hazards on mem in this way
    cmplx_flat_ptr[i] = cmplx[i][0];
  }

  cmplx_flat_ptr += len;

  for (i = 0; i < len; ++i) {
    // Save the img parts
    cmplx_flat_ptr[i] = buf[i];
  }

  return cmplx_flat_ptr;
}

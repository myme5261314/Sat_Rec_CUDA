/*
 * Self_Utils.h
 *
 *  Created on: 2014-5-25
 *      Author: peng
 */

#ifndef SELF_UTILS_H_
#define SELF_UTILS_H_

#include <assert.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <cula.hpp>
#include <cula_blas.hpp>
#include <cula_blas_device.hpp>
#include <cuda_runtime.h>
#include <cula_device.h>
#include <cublas_v2.h>

//#include "selfabstractMat.h"
#include "utils.h"
#include "selfhostMat.h"
#include "selfdeviceMat.h"

using namespace cv;

bool calMatMultiplication(const self_abstractMat& A, const self_abstractMat& B, const self_abstractMat& C);
bool calMatSVD(const self_abstractMat& Mat, const self_abstractMat& S);

void testMatMultiplication();


#endif /* SELF_UTILS_H_ */

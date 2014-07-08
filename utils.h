/*
 * utils.h
 *
 *  Created on: 2014-6-1
 *      Author: peng
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cula.h>

bool checkCudaError(cudaError_t err);
bool dispCULAStatus(culaStatus &s);


#endif /* UTILS_H_ */

/*
 * utils.h
 *
 *  Created on: 2014-6-1
 *      Author: peng
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <stdio.h>
#include <ctime>
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <cuda_runtime.h>
#include <cula.h>

bool checkCudaError(cudaError_t err);
bool dispCULAStatus(culaStatus &s);
void dispMessage(std::string message, std::clock_t t1, std::clock_t t2);


#endif /* UTILS_H_ */

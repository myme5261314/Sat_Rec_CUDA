/*
 * preProcess.h
 *
 *  Created on: 2014-5-12
 *      Author: peng
 */

#ifndef PREPROCESS_H_
#define PREPROCESS_H_

#include <eigen3/Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <exception>
#include <boost/foreach.hpp>
#include <iostream>
#include <ctime>
#include <omp.h>
#include <cula.h>
#include <cula_blas.h>

#include "Params.h"
#include "Load_Image.h"

using namespace cv;
using namespace boost;

void testpreProcess(const Params *params);

#endif /* LOAD_IMAGE_H_ */

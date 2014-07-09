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

#include "Self_Utils.h"

using namespace cv;
using namespace boost;

cv::Mat XImage2Data(const Params *params, const cv::Mat &ximage);
void calpreMean(const Params *params, const vector<cv::Mat> &inputImage,
		cv::Mat &out_Mat, size_t &num);
void calpreStd(const Params *params, const vector<cv::Mat> &inputImage,
		cv::Mat &out_Mat, size_t &num, const cv::Mat &premu);
void calCov(const Params *params, const vector<cv::Mat> &inputImage,
		cv::Mat &out_Mat, size_t &num, const cv::Mat &premu,
		const cv::Mat &presigma);
void calpostMean(const Params *params, const vector<cv::Mat> &inputImage,
		cv::Mat &out_Mat, size_t &num, const cv::Mat &premu,
		const cv::Mat &presigma, const cv::Mat &Ureduce);
void calpostStd(const Params *params, const vector<cv::Mat> &inputImage,
		cv::Mat &out_Mat, size_t &num, const cv::Mat &premu,
		const cv::Mat &presigma, const cv::Mat &Ureduce, const cv::Mat &postmu);

void testpreProcess(const Params *params);

#endif /* LOAD_IMAGE_H_ */

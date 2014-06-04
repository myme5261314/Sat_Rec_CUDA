/*
 * selfhostMat.cpp
 *
 *  Created on: 2014-6-1
 *      Author: peng
 */

#include "selfhostMat.h"

self_hostMat::self_hostMat(const cv::Mat& mat)
:self_abstractMat(mat.rows, mat.cols, (float*)mat.data, false)
{}

self_hostMat::~self_hostMat() {
	this->data = NULL;
}

float* self_hostMat::getHostCopy() {
	float *t = new float[this->row*this->column];
	memcpy(t, this->data, this->row*this->column*sizeof(float));
	return t;
}
float* self_hostMat::getDeviceCopy() {
	float *t = NULL;
	assert( checkCudaError( cudaMalloc(&t, this->row*this->column*sizeof(float)) ) );
	assert( checkCudaError( cudaMemcpy(t, this->data, this->row*this->column*sizeof(float), cudaMemcpyHostToDevice) ) );
	return t;
}

bool self_hostMat::copyTo(self_abstractMat& selfmat) {
	self_abstractMat::copyTo(selfmat);
	this->release();
	float *t = this->getCopy(selfmat);
	selfmat.data = t;
	return true;
}

bool self_hostMat::release() {
	if (this->data)
		free(this->data);
	this->data = NULL;
	return true;
}


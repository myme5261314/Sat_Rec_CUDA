/*
 * selfdeviceMat.cpp
 *
 *  Created on: 2014-6-1
 *      Author: peng
 */

#include "selfdeviceMat.h"

self_deviceMat::self_deviceMat(const cv::Mat& mat)
:self_abstractMat(mat.rows, mat.cols, NULL, false) {
	float *t = NULL;
	size_t byte = mat.rows*mat.cols*sizeof(float);
	assert( checkCudaError( cudaMalloc(&t,  byte) ) );
	assert( checkCudaError( cudaMemcpy(t, mat.data, byte, cudaMemcpyHostToDevice) ) );
	this->data = t;
}

self_deviceMat::~self_deviceMat() {
	this->release();
}

bool self_deviceMat::release() {
	if (this->data)
		assert( checkCudaError( cudaFree(this->data) ) );
	this->data = NULL;
	return true;
}

float* self_deviceMat::getHostCopy() {
	float *t = (float*)malloc(this->row*this->column*sizeof(float));
	assert( checkCudaError( cudaMemcpy(t, this->data, this->row*this->column*sizeof(float), cudaMemcpyDeviceToHost) ) );
	return t;
}
float* self_deviceMat::getDeviceCopy() {
	float *t = NULL;
	assert( checkCudaError( cudaMalloc(&t, this->row*this->column*sizeof(float)) ) );
	assert( checkCudaError( cudaMemcpy(t, this->data, this->row*this->column*sizeof(float), cudaMemcpyDeviceToDevice) ) );
	return t;
}

bool self_deviceMat::copyTo(self_abstractMat& selfmat) {
	self_abstractMat::copyTo(selfmat);
	this->release();
	float *t = this->getCopy(selfmat);
	selfmat.data = t;
	return true;
}

bool self_deviceMat::transferTo(cv::Mat& mat) {
	assert( (int)this->row == mat.rows );
	assert( (int)this->column == mat.cols );
	assert( mat.type()==CV_32F );
	return checkCudaError( cudaMemcpy(mat.data, this->data, this->row*this->column*sizeof(float), cudaMemcpyDeviceToHost ) );
}


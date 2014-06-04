/*
 * selfabstractMat.cpp
 *
 *  Created on: 2014-6-1
 *      Author: peng
 */

#include "selfabstractMat.h"

self_abstractMat::self_abstractMat() {
	this->row = 0;
	this->column = 0;
	this->data = NULL;
	this->trans = 0;
	this->isDeviceMemory = false;
}

self_abstractMat::self_abstractMat(size_t r, size_t c, float *d, bool t, bool isDeviceMemory) {
	this->row = r;
	this->column = c;
	this->data = d;
	this->trans = t;
	this->isDeviceMemory = isDeviceMemory;
}

self_abstractMat::self_abstractMat(const self_abstractMat& selfmat) {
	this->row = selfmat.row;
	this->column = selfmat.column;
	this->data = selfmat.data;
	this->trans = selfmat.trans;
	this->isDeviceMemory = selfmat.isDeviceMemory;
}

bool self_abstractMat::release() {
	this->data = NULL;
	return true;
}

bool self_abstractMat::copyTo(self_abstractMat& selfmat) {
	selfmat.row = this->row;
	selfmat.column = this->column;
	selfmat.trans = this->trans;
	return true;
}

float* self_abstractMat::getHostCopy() {
	return NULL;
}
float* self_abstractMat::getDeviceCopy() {
	return NULL;
}
float* self_abstractMat::getCopy(self_abstractMat& selfmat) {
	if (selfmat.isDeviceMemory)
		return this->getDeviceCopy();
	else
		return this->getHostCopy();
}


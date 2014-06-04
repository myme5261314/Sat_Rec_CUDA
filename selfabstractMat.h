/*
 * selfabstractMat.h
 *
 *  Created on: 2014-6-1
 *      Author: peng
 */

#ifndef SELFABSTRACTMAT_H_
#define SELFABSTRACTMAT_H_

#include <stdio.h>
#include <opencv2/opencv.hpp>
/*
 *
 */
class self_abstractMat {
public:
	float *data;
	size_t row;	// This row and column is row-major for C++.
	size_t column;
	bool trans;	// Whether transpose the matrix.
	bool isDeviceMemory;
	self_abstractMat();
	self_abstractMat(size_t r, size_t column, float *d, bool t, bool isDeviceMemory=false);
	self_abstractMat(const self_abstractMat& selfmat);
	virtual bool release();
	virtual ~self_abstractMat() {};
	virtual bool copyTo(self_abstractMat& selfmat);
	virtual float* getHostCopy();
	virtual float* getDeviceCopy();
	virtual float* getCopy(self_abstractMat& selfmat);
	inline void transpose() { trans = ~trans; }
};

#endif /* SELFABSTRACTMAT_H_ */

/*
 * selfhostMat.h
 *
 *  Created on: 2014-6-1
 *      Author: peng
 */

#ifndef SELFHOSTMAT_H_
#define SELFHOSTMAT_H_

#include "selfabstractMat.h"
#include <cuda_runtime.h>
#include <cula.h>

#include "utils.h"
/*
 *
 */
class self_hostMat: public self_abstractMat {
public:
	// Constructor part
	self_hostMat() {}
	self_hostMat(const cv::Mat& mat);
	virtual ~self_hostMat();

	virtual float* getHostCopy();
	virtual float* getDeviceCopy();
	bool copyTo(self_abstractMat& selfmat);

	virtual bool release();
};

#endif /* SELFHOSTMAT_H_ */

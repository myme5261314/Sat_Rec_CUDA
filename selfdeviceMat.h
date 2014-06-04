/*
 * selfdeviceMat.h
 *
 *  Created on: 2014-6-1
 *      Author: peng
 */

#ifndef SELFDEVICEMAT_H_
#define SELFDEVICEMAT_H_

#include "selfabstractMat.h"
#include "utils.h"

/*
 *
 */
class self_deviceMat: public self_abstractMat {
public:
	// Constructor part
	self_deviceMat() {};
	self_deviceMat(const cv::Mat& mat);
	virtual ~self_deviceMat();

	virtual float* getHostCopy();
	virtual float* getDeviceCopy();
	bool copyTo(self_abstractMat& selfmat);
	bool transferTo(cv::Mat& mat);

	virtual bool release();
};

#endif /* SELFDEVICEMAT_H_ */

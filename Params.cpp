/*
 * Params.cpp
 *
 *  Created on: 2014-5-4
 *      Author: peng
 */

#include "Params.h"

size_t matSize2Size(matSize mat) {
	return mat.height * mat.width;
}

matSize imgSize2DataMatSize(imgSize ims, size_t WindowSize,  size_t StrideSize) {
	float blank = float(WindowSize-StrideSize)/2;
	assert(blank==float(int(blank)));
	float width = ims.width;
	float height = ims.height;
	size_t c = size_t( floor( (width-blank*2)/StrideSize ) );
	size_t r = size_t( floor( (height-blank*2)/StrideSize ) );
	matSize ms = {r, c};
	return ms;
}




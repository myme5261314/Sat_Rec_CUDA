/*
 * Self_Utils.cpp
 *
 *  Created on: 2014-5-25
 *      Author: peng
 */

#include "Self_Utils.h"

using namespace std;

bool dispCULAStatus(culaStatus &s) {
	if( s != culaNoError )
	{
		int info;
		char buf[256];
	    info = culaGetErrorInfo();
	    culaGetErrorInfoString(s, info, buf, sizeof(buf));

	    printf("%s", buf);
	    return false;
	} else {
		return true;
	}
}



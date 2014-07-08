/*
 * utils.cpp
 *
 *  Created on: 2014-6-1
 *      Author: peng
 */

#include "utils.h"

using namespace std;

bool checkCudaError(cudaError_t err)
{
    if(!err)
        return true;
    printf("%s\n", cudaGetErrorString(err));
    culaShutdown();
    return false;
}

bool dispCULAStatus(culaStatus &s) {
	if( s != culaNoError )
	{
		int info;
		char buf[256];
	    info = culaGetErrorInfo();
	    culaGetErrorInfoString(s, info, buf, sizeof(buf));

	    cout << buf << endl;
//	    printf("%s", buf);
	    delete [] buf;
	    return false;
	} else {
		return true;
	}
}



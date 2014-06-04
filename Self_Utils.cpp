/*
 * Self_Utils.cpp
 *
 *  Created on: 2014-5-25
 *      Author: peng
 */

#include "Self_Utils.h"
#include <cula_device.h>
#include <cublas_v2.h>

using namespace std;


bool calMatMultiplication(const self_abstractMat& A, const self_abstractMat& B, const self_abstractMat& C) {
	assert(A.isDeviceMemory == B.isDeviceMemory);
	assert(B.isDeviceMemory == C.isDeviceMemory);
	size_t m,k1,k2,n;
	m = !A.trans?A.row:A.column;
	k1 = !A.trans?A.column:A.row;
	k2 = !B.trans?B.row:B.column;
	n = !B.trans?B.column:B.row;
	assert(k1==k2);
	char trans_A = !A.trans?'N':'T';
	char trans_B = !B.trans?'N':'T';
	int lda = trans_A!='N'?m:k1;
	int ldb = trans_B!='N'?k1:n;
	int ldc = n;
	culaStatus s;
	if (A.isDeviceMemory) {
		s = culaDeviceGemm(trans_B, trans_A, n, m, k1, 1.0f,
									(culaDeviceFloat*)B.data, ldb, (culaDeviceFloat*)A.data, lda,
									0.0f, (culaDeviceFloat*)C.data, ldc);
	} else {
		s = culaGemm(trans_B, trans_A, n, m, k1, 1.0f,
									(culaFloat*)B.data, ldb, (culaFloat*)A.data, lda,
									0.0f, (culaFloat*)C.data, ldc);
	}
	return dispCULAStatus(s);
}

void testMatMultiplication() {
    Mat A = (Mat_<float>(3,3) << 0, 100, 0, -1, 5, -1, 0, -1, 0);
//    Mat A = (Mat_<float>(2,2) << 100, 200, 300, 400);
    cout << "A = " << endl << " " << A << endl << endl;
    Mat B = (Mat_<float>(3,2) << 1, 2, 3, 4, 5, 6);
//    Mat B = (Mat_<float>(2,1) << 1.5, 2.5);
    cout << "B = " << endl << " " << B << endl << endl;
    Mat result = Mat::zeros(A.rows,B.cols,CV_32F);

    self_hostMat a(A), b(B), c(result);
    assert( calMatMultiplication(a,b,c) );

    cout << "c = " << endl << " " << result << endl << endl;

    self_deviceMat da(A), db(B), dc(result);
    assert( calMatMultiplication(da,db,dc) );
    Mat temp = Mat::zeros(A.rows,B.cols,CV_32F);
    dc.transferTo(temp);
    cout << "gc = " << endl << " " << temp << endl << endl;
}


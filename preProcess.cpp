/**
 * @author Peng Liu
 * @version 2014-05-12
 */

#include "preProcess.h"

using namespace std;

cv::Mat XImage2Data(const Params *params, const cv::Mat &ximage) {
	size_t WindowSize = params->data.WindowSize;
	size_t StrideSize = params->data.StrideSize;
	size_t ChannelSize = params->data.ChannelSize;
	imgSize is = { size_t(ximage.rows), size_t(ximage.cols) };
	matSize ms = imgSize2DataMatSize(is, WindowSize, StrideSize);
	cv::Mat dataMat = cv::Mat::zeros(ms.height * ms.width,
			WindowSize * WindowSize * ChannelSize, CV_8U);
	#pragma omp parallel for
	for (size_t i = 0; i < ms.height; ++i) {
		for (size_t j = 0; j < ms.width; ++j) {
			try {
				int c = i * ms.width + j;
				cv::Range rRange = cv::Range(i * StrideSize,
						i * StrideSize + WindowSize);
				cv::Range cRange = cv::Range(j * StrideSize,
						j * StrideSize + WindowSize);
				cv::Mat windowvec = ximage(rRange, cRange).clone().reshape(1,
						1);
				windowvec.copyTo(dataMat.row(c++));
			} catch (cv::Exception &e) {
				std::cout << e.what() << std::endl;
				std::cout << i * ms.width + j << std::endl;
			}
		}
	}
	return dataMat;
}

void calpreMean(const Params *params, const vector<cv::Mat> &inputImage,
		cv::Mat &out_Mat, size_t &num) {
	cv::Mat sum = cv::Mat::zeros(inputImage.size(), params->data.data_D, CV_32S);
	int *count = new int[inputImage.size()];
	try {
		#pragma omp parallel for
		for (size_t i=0; i < inputImage.size(); i++) {
			cv::Mat data = XImage2Data( params, inputImage[i] );
			cv::reduce( data, sum.row(i), 0, CV_REDUCE_SUM, CV_32S );
			count[i] = data.rows;
		}
		cv::Mat accum_count = cv::Mat::zeros(1,1,CV_32S);
		sum.convertTo( sum, out_Mat.type() );
		cv::reduce( sum, out_Mat, 0, CV_REDUCE_SUM );
		size_t temp=0;
		for (size_t i=0; i < inputImage.size(); i++)
			temp += count[i];
		num = temp;
		delete [] count;
	} catch (Exception &e) {
		cout << e.what() << endl;
		exit(1);
	}

}

void calpreStd(const Params *params, const vector<cv::Mat> &inputImage,
		cv::Mat &out_Mat, size_t &num, const cv::Mat &premu) {
	try {
		#pragma omp parallel for
		for (size_t i=0; i < inputImage.size(); i++) {
			cv::Mat data = XImage2Data( params, inputImage[i] );
			data.convertTo( data, out_Mat.type());
			#pragma omp parallel for
			for (int i = 0; i < data.rows; ++i)
			{
				data.row(i) -= premu;
			}
			cv::pow(data, 2, data);
			cv::Mat data_sum;
			cv::reduce( data, data_sum, 0, CV_REDUCE_SUM );
			#pragma omp critical
			{
				out_Mat += data_sum;
				num += data.rows;
			}
			data.release();
		}
	} catch(cv::Exception &e) {
		std::cout << e.what() << std::endl;
	}
}

void calCov(const Params *params, const vector<cv::Mat> &inputImage,
		cv::Mat &out_Mat, size_t &num, const cv::Mat &premu,
		const cv::Mat &presigma) {
	// #TODO: Need to use parallel for-loop
	cv::Mat data_sum = cv::Mat::zeros(inputImage.size(), out_Mat.cols,
			out_Mat.type());
	vector<int> num_vec(inputImage.size());
	cv::Mat t_mu, t_sigma;
	t_mu = premu.clone();
	t_sigma = presigma.clone();
	premu.convertTo(t_mu, out_Mat.type());
	presigma.convertTo(t_sigma, out_Mat.type());
	int cuNum = cv::gpu::getCudaEnabledDeviceCount();
	assert(cuNum);
//#pragma omp parallel for num_threads(cuNum)
	for (size_t i = 0; i < inputImage.size(); ++i) {
		try {
			cv::Mat data = XImage2Data(params, inputImage[i]);
			data.convertTo(data, out_Mat.type());
			for (int i = 0; i < data.rows; ++i) {
				data.row(i) -= t_mu;
				data.row(i) /= t_sigma;
			}
			std::clock_t t1 = std::clock();
			culaFloat *result = (culaFloat*)malloc(data.cols*data.cols*sizeof(float));
			culaStatus s;
			culaFloat* in = (culaFloat*)data.data;
			s = culaSgemm('T', 'N', data.cols, data.cols, data.rows, 1.0f, in, data.rows, in, data.rows, 0.0f, result, data.cols);
			dispCULAStatus(s);
			num_vec[i] = data.rows;
			cv::Mat resultdata(data.cols, data.cols, CV_32F, (void*)result, 0);
			out_Mat += resultdata;
			data.release();
			resultdata.release();
			std::cout << double(std::clock() - t1) / CLOCKS_PER_SEC
					<< std::endl;
		} catch (cv::Exception &e) {
			std::cout << e.what() << std::endl;
		}
	}
	cv::Mat summary;
	for (size_t i = 0; i < num_vec.size(); ++i) {
		num += num_vec[i];
	}
}

void testpreProcess(const Params *params) {
	// Load Image File
	std::clock_t t1, t2;
	t1 = std::clock();
	t2 = std::clock();
	boost::filesystem::path root(params->path.dataFloder);
	boost::filesystem::path trainSat = root / params->path.trainFloder
			/ params->path.satFloder;
	std::vector<string> trainSat_vec = extractFileList(
			excludeFileList(getDirList(trainSat), ".tiff", false),
			params->debug.debugSize);
	dispVector(trainSat_vec);
	boost::filesystem::path trainMap = root / params->path.trainFloder
			/ params->path.mapFloder;
	std::vector<string> trainMap_vec = extractFileList(
			excludeFileList(getDirList(trainMap), ".tif", false),
			params->debug.debugSize);
	dispVector(trainMap_vec);
	std::vector<cv::Mat> trainMapImgVec = batchLoadImage(trainMap_vec);
	std::vector<cv::Mat> trainSatImgVec = batchLoadImage(trainSat_vec);
	trainMapImgVec.clear();
	t2 = std::clock();
	std::cout << "Finish Load Image, Elpased time is:";
	std::cout << double(t2 - t1) / CLOCKS_PER_SEC << std::endl;

	// Calculate premean stage.
	cv::Mat premean = cv::Mat::zeros(1,
			params->data.WindowSize * params->data.WindowSize
					* params->data.ChannelSize, CV_32F);
	size_t total = 0;
	calpreMean(params, trainSatImgVec, premean, total);
	premean /= total;
	std::cout << "Finish Cal preMean, Elpased time is:";
	std::cout << double(std::clock() - t2) / CLOCKS_PER_SEC << std::endl;
//	std::cout << premean << endl;
//	exit(0);
	// Calculate prestd stage.
	t2 = std::clock();
	cv::Mat prestd = cv::Mat::zeros(1,
			params->data.WindowSize * params->data.WindowSize
					* params->data.ChannelSize, CV_32F);
	total = 0;
	calpreStd(params, trainSatImgVec, prestd, total, premean);
	prestd /= total;
	cv::sqrt(prestd, prestd);
	std::cout << "Finish Cal preStd, Elpased time is:";
	std::cout << double(std::clock() - t2) / CLOCKS_PER_SEC << std::endl;
//	std::cout << prestd << endl;
//	exit(0);

	// Calculate covariance stage.
	t2 = std::clock();
	cv::Mat sig = cv::Mat::zeros(
			params->data.WindowSize * params->data.WindowSize
					* params->data.ChannelSize,
			params->data.WindowSize * params->data.WindowSize
					* params->data.ChannelSize, CV_32F);
	total = 0;
	calCov(params, trainSatImgVec, sig, total, premean, prestd);
//	cout << "total=" << total << endl;
	sig /= total;
	std::cout << "Finish Cal Cov, Elpased time is:";
	std::cout << double(std::clock() - t2) / CLOCKS_PER_SEC << std::endl;

	// Calculate SVD Stage.
	t2 = std::clock();
//	cv::Mat mat = cv::Mat::ones(10,10,CV_32F);
//	cv::Mat S = cv::Mat::zeros(1,10,CV_32F);
//	cv::Mat U = cv::Mat::zeros(10,10,CV_32F);
//	cv::Mat VT = cv::Mat::zeros(10,10,CV_32F);
//	self_hostMat selfmat(mat), selfS(S), selfU(U), selfVT(VT);
//	calMatSVD(selfmat, selfU, selfS, selfVT);
//	cout<< U << endl;

	sig = sig(Range(0,sig.rows/3), Range(0,sig.rows/3));
	self_hostMat mat(sig);
	cv::Mat S = cv::Mat::zeros(1, sig.rows, CV_32F);
	self_hostMat selfS(S);
	cv::Mat U = cv::Mat::zeros(sig.rows, sig.rows, CV_32F);
	self_hostMat selfU(U);
	cv::Mat VT = cv::Mat::zeros(sig.rows, sig.rows, CV_32F);
	self_hostMat selfVT(VT);
	calMatSVD(mat, selfS);
	cout<< S << endl;

//	float *A = new float[sig.rows * sig.cols];
////	memcpy(A, (const void*) sig.data, sig.rows * sig.cols * sizeof(float));
//	for(int i=0; i<sig.rows*sig.cols; i++) {
//		A[i] = i+1;
//	}
//	float *U = new float[sig.rows * sig.rows];
////	float *VT = new float[sig.rows * sig.rows];
//	float *S = new float[1 * sig.rows];
//	assert(culaInitialize() == culaNoError);
////	assert(culaSgesvd('A', 'N', sig.rows, sig.rows, A, sig.rows, S, U,
////			sig.rows, VT, sig.rows) == culaNoError);
//	culaStatus s = culaSgesvd('A', 'N', sig.rows, sig.rows, A, sig.rows, S, U,
//			sig.rows, NULL, sig.rows);
//	cout<< U[0]<<endl;
//	delete [] A;
//	delete [] U;
//	delete [] S;
////	delete [] VT;

//	t2 = std::clock();
	std::cout << double(std::clock() - t2) / CLOCKS_PER_SEC << std::endl;
//	std::cout << premean << std::endl;
}

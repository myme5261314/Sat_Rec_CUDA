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

	for (size_t i = 0; i < num_vec.size(); ++i) {
		num += num_vec[i];
	}
}

void calpostMean(const Params *params, const vector<cv::Mat> &inputImage,
		cv::Mat &out_Mat, size_t &num, const cv::Mat &premu,
		const cv::Mat &presigma, const cv::Mat &Ureduce) {
	cv::Mat sum = cv::Mat::zeros(inputImage.size(), params->data.data_D/3, CV_32F);
	int *count = new int[inputImage.size()];
	try {
		#pragma omp parallel for
		for (size_t i=0; i < inputImage.size(); i++) {
			cv::Mat data = XImage2Data( params, inputImage[i] );
			data.convertTo(data, out_Mat.type());
			#pragma omp parallel for
			for (int j=0; j < data.rows; j++) {
				data.row(j) -= premu;
				data.row(j) -= presigma;
			}
			cv::Mat tempdata = cv::Mat::zeros(data.rows, Ureduce.cols, data.type());
			self_hostMat A(data), B(Ureduce), C(tempdata);
			#pragma omp critical
			{
				calMatMultiplication(A, B, C);
			}
			cv::reduce( tempdata, sum.row(i), 0, CV_REDUCE_SUM, CV_32F );
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

void calpostStd(const Params *params, const vector<cv::Mat> &inputImage,
		cv::Mat &out_Mat, size_t &num, const cv::Mat &premu,
		const cv::Mat &presigma, const cv::Mat &Ureduce, const cv::Mat &postmu) {
	try {
		#pragma omp parallel for
		for (size_t i=0; i < inputImage.size(); i++) {
			cv::Mat data = XImage2Data( params, inputImage[i] );
			data.convertTo( data, out_Mat.type());
			#pragma omp parallel for
			for (int j = 0; j < data.rows; ++j)
			{
				data.row(j) -= premu;
				data.row(j) /= presigma;
			}
			cv::Mat tempdata = cv::Mat::zeros(data.rows, Ureduce.cols, data.type());
			self_hostMat A(data), B(Ureduce), C(tempdata);
			#pragma omp critical
			{
				calMatMultiplication(A, B, C);
			}
			#pragma omp parallel for
			for (int j = 0; j < data.rows; ++j)
			{
				tempdata.row(j) -= postmu;
			}
			cv::pow(tempdata, 2, tempdata);
			cv::Mat data_sum;
			cv::reduce( tempdata, data_sum, 0, CV_REDUCE_SUM );
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

void testpreProcess(Params *params) {
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
	std::vector<cv::Mat> trainSatImgVec = batchLoadImage(trainSat_vec);
	t2 = std::clock();
	string message = "Finish Load Image, Elapsed time is:";
	dispMessage(message, t1, t2);

	// Calculate premean stage.
	params->data.premean = cv::Mat::zeros(1,
			params->data.WindowSize * params->data.WindowSize
					* params->data.ChannelSize, CV_32F);
	string keyword = "premean";
	string path = params->path.dataFloder + params->path.cacheFloder + params->path.cachePreMean;
	size_t total = 0;
	if (!loadMat(path, keyword, params->data.premean)) {
		calpreMean(params, trainSatImgVec, params->data.premean, total);
		params->data.premean /= total;
		saveMat(path, keyword, params->data.premean);
	}
	message = "Finish Cal preMean, Elapsed time is:";
	dispMessage(message, t1, t2);

	// Calculate prestd stage.
	t2 = std::clock();
	params->data.prestd = cv::Mat::zeros(1,
			params->data.WindowSize * params->data.WindowSize
					* params->data.ChannelSize, CV_32F);
	keyword = "prestd";
	path = params->path.dataFloder + params->path.cacheFloder + params->path.cachePreStd;
	if (!loadMat(path, keyword, params->data.prestd)) {
		total = 0;
		calpreStd(params, trainSatImgVec, params->data.prestd, total, params->data.premean);
		params->data.prestd /= total;
		cv::sqrt(params->data.prestd, params->data.prestd);
		saveMat(path, keyword, params->data.prestd);
	}
	message = "Finish Cal preStd, Elapsed time is:";
	dispMessage(message, t1, t2);

	// Calculate covariance stage.
	t1 = std::clock();
	params->data.Ureduce = cv::Mat::zeros(
			params->data.WindowSize * params->data.WindowSize
					* params->data.ChannelSize,
			params->data.WindowSize * params->data.WindowSize
					* params->data.ChannelSize, CV_32F);
	total = 0;
	calCov(params, trainSatImgVec, params->data.Ureduce, total, params->data.premean, params->data.prestd);
	params->data.Ureduce /= total;
	t2 = std::clock();
	message = "Finish Cal Cov, Elapsed time is:";
	dispMessage(message, t1, t2);

	// Calculate SVD Stage.
	t1 = std::clock();
	keyword = "pca";
	path = params->path.dataFloder + params->path.cacheFloder + params->path.cachePca;
//	sig = sig(Range(0,sig.rows/3), Range(0,sig.rows/3));
	if (!loadMat(path, keyword, params->data.Ureduce)) {
		self_hostMat mat(params->data.Ureduce);
		cv::Mat S = cv::Mat::zeros(1, params->data.Ureduce.rows, CV_32F);
		self_hostMat selfS(S);
		calMatSVD(mat, selfS);
		cout<< S << endl;
		int c = params->data.Ureduce.cols;
		params->data.Ureduce = params->data.Ureduce( cv::Range::all(), Range(0,c/3) );
		saveMat(path, keyword, params->data.Ureduce);
	}
	t2 = std::clock();
	message = "Finish Cal SVD, Elapsed time is:";
	dispMessage(message, t1, t2);

	// Calculate PostMean Stage.
	t1 = std::clock();
	params->data.postmean = cv::Mat::zeros(1, params->data.Ureduce.cols, CV_32F);
	keyword = "postmean";
	path = params->path.dataFloder + params->path.cacheFloder + params->path.cachePostMean;
	if (!loadMat(path, keyword, params->data.postmean)) {
		total = 0;
		calpostMean(params, trainSatImgVec, params->data.postmean, total, params->data.premean, params->data.prestd, params->data.Ureduce);
		params->data.postmean /= total;
		saveMat(path, keyword, params->data.postmean);
	}
	t2 = std::clock();
	message = "Finish Cal postMean, Elapsed time is:";
	dispMessage(message, t1, t2);

	// Calculate PostStd stage.
	t1 = std::clock();
	params->data.poststd = cv::Mat::zeros(1, params->data.Ureduce.cols, CV_32F);
	keyword = "poststd";
	path = params->path.dataFloder + params->path.cacheFloder + params->path.cachePostStd;
	if (!loadMat(path, keyword, params->data.poststd)) {
		total = 0;
		calpostStd(params, trainSatImgVec, params->data.poststd, total, params->data.premean, params->data.prestd, params->data.Ureduce, params->data.postmean);
		params->data.poststd /= total;
		cv::sqrt(params->data.poststd, params->data.poststd);
		saveMat(path, keyword, params->data.poststd);
	}
	t2 = std::clock();
	message = "Finish Cal postStd, Elapsed time is:";
	dispMessage(message, t1, t2);

}

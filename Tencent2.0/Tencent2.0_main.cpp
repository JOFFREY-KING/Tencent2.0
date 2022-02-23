//#define USE_CUDA

#include <iostream>
#include<stdio.h>
#include<io.h>
#include <random>
#include <string>
#include <iterator>
#include <algorithm>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "utils.h"

#ifdef USE_CUDA
#include <opencv2/cudaoptflow.hpp>
#else
#include "pixflow/OpticalFlowFactory.h"
#endif

using namespace std;
using namespace cv;
using cv::cuda::GpuMat;

#ifdef USE_CUDA
void extractTexturePart(const cv::cuda::GpuMat& d_src, cv::cuda::GpuMat& d_dst) {
	CV_Assert(d_src.type() == CV_8U);
	GpuMat d_buf8U(d_src.size(), CV_8U);
	d_src.copyTo(d_buf8U);
	cuda::bilateralFilter(d_buf8U, d_dst, 5, 12, 35, 4);
	cuda::bilateralFilter(d_dst, d_buf8U, 25, 25, 50, 4);
	cuda::addWeighted(d_dst, 1, d_buf8U, -0.5f, 0, d_buf8U, -1);
	d_buf8U.copyTo(d_buf8U, d_dst);
}

void normallizeFlowInput(const cv::cuda::GpuMat& d_src, cv::cuda::GpuMat& d_dst)
{
	CV_Assert(d_src.type() == CV_8UC3);
	GpuMat d_buf8U(d_src.size(), CV_8U);
	GpuMat d_buf8U_final(d_src.size(), CV_8U);
	cuda::cvtColor(d_src, d_buf8U, COLOR_BGR2GRAY, 0);
	extractTexturePart(d_buf8U, d_buf8U);
	Ptr<cuda::Filter> gaussian8UC1 = cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(5, 5), 0);
	gaussian8UC1->apply(d_buf8U, d_buf8U_final);
	d_buf8U_final.convertTo(d_dst, CV_32FC1, 1 / 255.);
}

void flow_match(const cv::Mat& img0, const cv::Mat& img1, cv::Mat& warp) {
	GpuMat d_0(img0), d_1(img1), d_flow;
	normallizeFlowInput(d_0, d_0);
	normallizeFlowInput(d_1, d_1);

	Mat im0, im1, flow;
	//Ptr < cuda::BroxOpticalFlow > brox = cuda::BroxOpticalFlow::create(0.197, 50.0, 0.9, 10, 150, 10);
	Ptr < cuda::BroxOpticalFlow > brox = cuda::BroxOpticalFlow::create();
	brox->calc(d_0, d_1, d_flow);

	d_0.download(im0);
	d_1.download(im1);
	d_flow.download(flow);

	resize(flow, flow, Size(1024, 1024));
	cv::pyrDown(flow, flow);
	cv::pyrDown(flow, flow);
	cv::pyrDown(flow, flow);
	cv::pyrDown(flow, flow);
	resize(flow, flow, im0.size());
	cv::min(flow, 3.0, flow);
	cv::max(flow, -3.0, flow);
	cv::GaussianBlur(flow, flow, Size(5, 5), 0);

	Mat xmap(flow.size(), CV_32F), ymap(flow.size(), CV_32F);
	for (int y = 0; y < flow.rows; y++) {
		for (int x = 0; x < flow.cols; x++) {
			Point2f motion = flow.at<Point2f>(y, x);
			//cout << motion << endl;
			xmap.at<float>(y, x) = x + motion.x;
			ymap.at<float>(y, x) = y + motion.y;
		}
	}

	remap(img1, warp, xmap, ymap, INTER_CUBIC, BORDER_REPLICATE);
	cout << d_0.size() << d_1.size() << endl;
}
void calc_flow(const cv::Mat& flowInput0, const cv::Mat& flowInput1, cv::Mat& flow) {
	GpuMat d_0, d_1, d_flow;
	d_0.upload(flowInput0);
	d_1.upload(flowInput1);
	normallizeFlowInput(d_0, d_0);
	normallizeFlowInput(d_1, d_1);
	//Ptr < cuda::BroxOpticalFlow > brox = cuda::BroxOpticalFlow::create();
	//Ptr < cuda::BroxOpticalFlow > brox = cuda::BroxOpticalFlow::create(0.197, 50.0, 0.9, 10, 150, 10);
	brox->calc(d_0, d_1, d_flow);
	d_0.download(flowInput0);
	d_1.download(flowInput1);
	d_flow.download(flow);
}
#else
void flow_match(const cv::Mat& img0, const cv::Mat& img1, cv::Mat& warp) {
}
void calc_flow(const cv::Mat& flowInput0, const cv::Mat& flowInput1, cv::Mat& flow) {
	Ptr<surround360::optical_flow::OpticalFlowInterface> pixflow =
		surround360::optical_flow::makeOpticalFlowByName("pixflow_search_20");
	Mat I0BGRA, I1BGRA, prevFlow, prevI0BGRA, prevI1BGRA;
	cv::cvtColor(flowInput0, I0BGRA, COLOR_BGR2BGRA);
	cv::cvtColor(flowInput1, I1BGRA, COLOR_BGR2BGRA);
	//prevFlow = Mat::zeros(flowInput0.size(), CV_32FC2);
	//prevI0BGRA = I0BGRA;
	//prevI1BGRA = I1BGRA;

	pixflow->computeOpticalFlow(
		I0BGRA,
		I1BGRA,
		prevFlow,
		prevI0BGRA,
		prevI1BGRA,
		flow,
		surround360::optical_flow::OpticalFlowInterface::DirectionHint::UNKNOWN);
}
#endif

cv::Mat visualizeKeypointMatches(
	const cv::Mat& imageL, const cv::Mat& imageR, const vector< Point2f>& ps0, const vector< Point2f>& ps1) {

	Mat visualization = Mat::zeros(max(imageL.rows, imageR.rows), imageL.cols + imageR.cols, CV_8UC3);
	imageL.copyTo(visualization(Rect(0, 0, imageL.cols, imageL.rows)));
	imageR.copyTo(visualization(Rect(imageL.cols, 0, imageR.cols, imageR.rows)));

	const Scalar kVisPointColor = Scalar(110, 220, 0);
	for (int i = 0; i < ps0.size(); i++) {

		line(
			visualization,
			ps0[i],
			ps1[i] + Point2f(imageL.cols, 0),
			kVisPointColor,
			1, // thickness
			LINE_AA);
	}
	//namedWindow("f", WINDOW_NORMAL);
	//	imshow("f", visualization);
	//waitKey();

	return visualization;
}

cv::Rect feat_match_simple(const cv::Mat& img0Ori, const cv::Mat& img1Ori, cv::Mat& warp0, cv::Mat& warp1) {
	std::vector<cv::Point2f> ps0, ps1;

	Mat img0Filterd, img1Filterd, img0, img1;
	//bilateralFilter(img1Ori, img0, 5, 18, 35, BORDER_REPLICATE);
	//bilateralFilter(img1Ori, img1, 5, 18, 35, BORDER_REPLICATE);
	fastNlMeansDenoising(img0Ori, img0, 10, 7, 11);
	fastNlMeansDenoising(img1Ori, img1, 10, 7, 11);

	for (int algo_i = 0; algo_i < 3; algo_i++) {
		std::vector<cv::KeyPoint> keypoints0, keypoints1;
		cv::Mat desc0, desc1;
		std::vector<cv::DMatch>  matches, goodMatches;
		std::vector<std::vector<cv::DMatch>> matches_knn;

		Ptr<Feature2D> detector, descriptor;
		if (algo_i == 0) {
			detector = AKAZE::create();
			descriptor = detector;
		}
		else if (algo_i == 1) {
			detector = ORB::create(2000, 1.15, 14, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
			descriptor = detector;
		}
		else if (algo_i == 2) {
			detector = BRISK::create();
			descriptor = detector;
		}

		detector->detect(img0, keypoints0);
		descriptor->compute(img0, keypoints0, desc0);
		detector->detect(img1, keypoints1);
		descriptor->compute(img1, keypoints1, desc1);

		cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NormTypes::NORM_HAMMING, true);
		matcher->match(desc0, desc1, matches);
		//matcher = cv::BFMatcher::create(cv::NormTypes::NORM_HAMMING);
		//matcher->knnMatch(desc0, desc1, matches_knn, 2);

		//for (size_t i = 0; i < matches_knn.size(); i++) {
		//	const cv::DMatch &bestMatch = matches_knn[i][0];
		//	const cv::DMatch &betterMatch = matches_knn[i][1];

		//	float distanceRatio = bestMatch.distance / betterMatch.distance;

		//	// Pass only matches where distance ratio between
		//	// nearest matches is greater than 1.5 (distinct criteria)
		//	if (distanceRatio < 0.75) {
		//		matches.push_back(bestMatch);
		//	}
		//}

		goodMatches.clear();
		float minDist = std::min_element(
			matches.begin(), matches.end(),
			[](const cv::DMatch& m1, const cv::DMatch& m2) {
				return m1.distance < m2.distance;
			})->distance;
		double distThresh = max<float>(3 * minDist, 30.0);

		for (int j = 0; j < matches.size(); j++) {
			// todo: cross check
			// retain those matches with enough small distance
			if (matches[j].distance <= distThresh) {
				goodMatches.push_back(matches[j]);
			}
		}
		if (goodMatches.size() > 1000) {
			std::sort(goodMatches.begin(), goodMatches.end(),
				[](const cv::DMatch& m1, const cv::DMatch& m2) { return m1.distance < m2.distance; });
			goodMatches.resize(1000);
		}

		for (const DMatch& match : goodMatches) {
			ps0.push_back(keypoints0[match.queryIdx].pt);
			ps1.push_back(keypoints1[match.trainIdx].pt);
		}

	}
	std::vector<uchar> inliersMask;
	float reprojectionThreshold = 10.0;
	Mat H = cv::findHomography(ps0, ps1, RANSAC, reprojectionThreshold, inliersMask);
	//visualizeKeypointMatches(img0, img1, ps0, ps1);

	Mat msk = Mat::zeros(img0Ori.size(), CV_32FC3) + Scalar(1.0, 1.0, 1.0);
	warpPerspective(msk, msk, H, img1.size(), INTER_CUBIC, BORDER_CONSTANT);
	dilate(msk, msk, getStructuringElement(MORPH_RECT, Size(15, 15)));
	blur(msk, msk, Size(31, 31));
	//blur(msk, msk, Size(31, 31));
	//blur(msk, msk, Size(31, 31));
	cv::min(msk, 1.0, msk);
	cv::max(msk, 0.0, msk);

	warpPerspective(img0Ori, warp0, H, img1.size(), INTER_CUBIC, BORDER_REPLICATE);
	warp0.convertTo(warp0, CV_32FC3);
	cv::multiply(warp0, msk, warp0);
	//Mat bg; img1.convertTo(bg, CV_32FC3);
	//multiply(bg, Scalar(1.0, 1.0, 1.0) -msk, bg);
	//warp = warp + bg;
	warp0.convertTo(warp0, CV_8UC3);

	img1Ori.convertTo(warp1, CV_32FC3);
	cv::multiply(warp1, msk, warp1);
	warp1.convertTo(warp1, CV_8UC3);


	vector<Point2f> corners = { Point2f(0, 0), Point2f(img0.cols, img0.rows), Point2f(img0.cols,0), Point2f(0, img0.rows) };
	vector<Point2f> corners_warp;
	perspectiveTransform(corners, corners_warp, H);

	float xmin = 10000000000, ymin = 100000000000000, xmax = -1, ymax = -1;
	for (auto p : corners_warp) {
		xmin = min(xmin, p.x);
		ymin = min(ymin, p.y);
		xmax = max(xmax, p.x);
		ymax = max(ymax, p.y);
	}
	xmin = fmaxf(xmin - 200, 0);
	ymin = fmaxf(ymin - 200, 0);
	xmax = fminf(xmax + 200, warp0.cols - 1);
	ymax = fminf(ymax + 200, warp0.rows - 1);

	Rect bbox(xmin, ymin, xmax - xmin, ymax - ymin);


	Mat vis = visualizeKeypointMatches(img0, img1, ps0, ps1);
	return bbox;
}

int features_num = 1000;
float scale_factor = 1.2f;
int levels_num = 8;
int default_fast_T = 20;   //FAST默认检测阈值
int min_fast_T = 7;        //FAST最小检测阈值
int edge_threshold = 19;   //边界尺度
int PATCH_SIZE = 31;

class ExtractorNode {
public:
	ExtractorNode() :bNoMore(false) {}
	void DivideNode(ExtractorNode& n1, ExtractorNode& n2, ExtractorNode& n3, ExtractorNode& n4);
	std::vector<cv::KeyPoint> vkeys;
	cv::Point2i UL, UR, BL, BR;
	std::list<ExtractorNode>::iterator lit;
	bool bNoMore;   
};

void ExtractorNode::DivideNode(ExtractorNode& n1, ExtractorNode& n2, ExtractorNode& n3, ExtractorNode& n4)
{
	const int halfx = ceil(static_cast<float>(UR.x - UL.x) / 2);
	const int halfy = ceil(static_cast<float>(BR.y - UL.y) / 2);

	n1.UL = UL;
	n1.UR = cv::Point2i(UL.x + halfx, UL.y);
	n1.BL = cv::Point2i(UL.x, UL.y + halfy);
	n1.BR = cv::Point2i(UL.x + halfx, UL.y + halfy);
	n1.vkeys.reserve(vkeys.size());

	n2.UL = n1.UR;
	n2.UR = UR;
	n2.BL = n1.BR;
	n2.BR = cv::Point2i(UR.x, UL.y + halfy);
	n2.vkeys.reserve(vkeys.size());

	n3.UL = n1.BL;
	n3.UR = n1.BR;
	n3.BL = BL;
	n3.BR = cv::Point2i(UL.x + halfx, BL.y);
	n3.vkeys.reserve(vkeys.size());

	n4.UL = n3.UR;
	n4.UR = n2.BR;
	n4.BL = n3.BR;
	n4.BR = BR;
	n4.vkeys.reserve(vkeys.size());

	for (size_t i = 0; i < vkeys.size(); ++i)
	{
		const cv::KeyPoint& kp = vkeys[i];
		if (kp.pt.x < n1.UR.x)
		{
			if (kp.pt.y < n1.BL.y)
				n1.vkeys.push_back(kp);
			else
				n3.vkeys.push_back(kp);
		}
		else if (kp.pt.y < n1.BR.y)
		{
			n2.vkeys.push_back(kp);
		}
		else
		{
			n4.vkeys.push_back(kp);
		}
	}

	if (n1.vkeys.size() == 1)
	{
		n1.bNoMore = true;
	}
	if (n2.vkeys.size() == 1)
	{
		n2.bNoMore = true;
	}
	if (n3.vkeys.size() == 1)
	{
		n3.bNoMore = true;
	}
	if (n4.vkeys.size() == 1)
	{
		n4.bNoMore = true;
	}

}

vector<cv::KeyPoint> Quadtree_detector(Mat img) {

	if (img.empty())
		std::cout << "no picture was found ...." << endl;
	else
		std::cout << "img load successed!" << endl;

	vector<int> feature_num_per_level;
	vector<float> vec_scale_per_factor;

	vec_scale_per_factor.resize(levels_num);
	vec_scale_per_factor[0] = 1.0f;
	for (int i = 1; i < levels_num; ++i) {
		vec_scale_per_factor[i] = vec_scale_per_factor[i - 1] * scale_factor;
	}

	vector<Mat> vec_img_pyramid(levels_num);
	for (int level = 0; level < levels_num; ++level) {
		float scale = 1.0f / vec_scale_per_factor[level];
		Size sz(cvRound((float)img.cols * scale), cvRound((float)img.rows * scale));

		if (level == 0)
		{
			vec_img_pyramid[level] = img;
		}
		else
			cv::resize(vec_img_pyramid[level - 1], vec_img_pyramid[level], sz, 0, 0, INTER_LINEAR);

		std::cout << "正在构建第 " << level + 1 << " 层金字塔" << endl;
		//imshow("img_pyramid", vec_img_pyramid[level]);
		//waitKey(100);

	}
	std::cout << "*************************" << endl << endl;

	vector<vector<KeyPoint>> all_keypoints;
	all_keypoints.resize(levels_num);

	const float border_width = 30;

	for (int level = 0; level < levels_num; level++) {
		std::cout << level << endl;
		const int min_boder_x = edge_threshold - 3;
		const int min_boder_y = min_boder_x;
		const int max_boder_x = vec_img_pyramid[level].cols - edge_threshold + 3;
		const int max_boder_y = vec_img_pyramid[level].rows - edge_threshold + 3;

		vector<cv::KeyPoint> vec_to_per_distribute_keys;
		vec_to_per_distribute_keys.reserve(features_num * 10);

		const float width = max_boder_x - min_boder_x;
		const float height = max_boder_y - min_boder_y;

		const int cols = width / border_width;
		const int rows = height / border_width;

		const int width_cell = ceil(width / cols);
		const int height_cell = ceil(height / rows);

		std::cout << "第" << level + 1 << "层图像被切割成 " << rows << " 行，" << cols << " 列" << endl;
		std::cout << "格子列数： " << width_cell << "， 格子行数：" << height_cell << endl;

		for (int i = 0; i < rows; ++i) {
			const float ini_y = min_boder_y + i * height_cell;
			float max_y = ini_y + height_cell + 6;

			if (ini_y >= max_boder_y - 3)
				continue;
			if (max_y >= max_boder_y)
				max_y = max_boder_y;

			for (int j = 0; j < cols; ++j) {
				const float ini_x = min_boder_x + j * width_cell;
				float max_x = ini_x + width_cell + 6;

				if (ini_x >= max_boder_x - 6)
					continue;
				if (max_x >= max_boder_x)
					max_x = max_boder_x;

				vector<KeyPoint> vec_keys_cell;
				vector<KeyPoint> desc;

				FAST(vec_img_pyramid[level].rowRange(ini_y, max_y).colRange(ini_x, max_x), vec_keys_cell, default_fast_T, true);


				if (vec_keys_cell.empty())
				{
					cv::FAST(vec_img_pyramid[level].rowRange(ini_y, max_y).colRange(ini_x, max_x), vec_keys_cell, min_fast_T, true);
				}

				if (!vec_keys_cell.empty())
				{
					for (std::vector<cv::KeyPoint>::iterator vit = vec_keys_cell.begin(); vit != vec_keys_cell.end(); vit++)
					{
						(*vit).pt.x += j * width_cell;
						(*vit).pt.y += i * height_cell;
						vec_to_per_distribute_keys.push_back(*vit);
					}

				}

			}

		}

		std::cout << "这层图像共有 " << vec_to_per_distribute_keys.size() << " 个特征点" << endl;

		std::vector<cv::KeyPoint>& keypoints = all_keypoints[level];
		keypoints.reserve(features_num);

		const int init_node_num = round(static_cast<float>(max_boder_x - min_boder_x) / (max_boder_y - min_boder_y));
		std::cout << "初始化时有 " << init_node_num << " 个节点" << endl;

		const float interval_x = static_cast<float>(max_boder_x - min_boder_x) / init_node_num;
		std::cout << "节点间隔： " << interval_x << endl;

		std::list<ExtractorNode> list_nodes;   
		std::vector<ExtractorNode*> init_nodes_address;   
		init_nodes_address.resize(init_node_num);


		for (int i = 0; i < init_node_num; ++i) {
			ExtractorNode ni;
			ni.UL = cv::Point2i(interval_x * static_cast<float>(i), 0);
			ni.UR = cv::Point2i(interval_x * static_cast<float>(i + 1), 0);
			ni.BL = cv::Point2i(ni.UL.x, max_boder_y - min_boder_y);
			ni.BR = cv::Point2i(ni.UR.x, max_boder_y - min_boder_x);
			ni.vkeys.reserve(vec_to_per_distribute_keys.size());

			list_nodes.push_back(ni);
			init_nodes_address[i] = &list_nodes.back();
		}

		for (size_t i = 0; i < vec_to_per_distribute_keys.size(); ++i)
		{
			const cv::KeyPoint& kp = vec_to_per_distribute_keys[i];
			init_nodes_address[kp.pt.x / interval_x]->vkeys.push_back(kp);

		}

		list<ExtractorNode>::iterator lit = list_nodes.begin();
		while (lit != list_nodes.end())
		{
			if (lit->vkeys.size() == 1)
			{
				lit->bNoMore = true;
				lit++;
			}
			else if (lit->vkeys.empty())
			{
				lit = list_nodes.erase(lit);
			}
			else
				lit++;
		}

		bool is_finish = false;
		int iteration = 0;

		vector<std::pair<int, ExtractorNode*>> key_size_and_node;
		key_size_and_node.reserve(list_nodes.size() * 4);

		while (!is_finish)
		{
			iteration++;
			int pre_size = list_nodes.size();

			lit = list_nodes.begin();
			int to_expand_num = 0;
			key_size_and_node.clear();

			while (lit != list_nodes.end()) {
				if (lit->bNoMore) {
					lit++;
					continue;
				}
				else {
					ExtractorNode n1, n2, n3, n4;
					lit->DivideNode(n1, n2, n3, n4);

					if (n1.vkeys.size() > 0) {
						list_nodes.push_front(n1);
						if (n1.vkeys.size() > 1) {
							to_expand_num++;
							key_size_and_node.push_back(std::make_pair(n1.vkeys.size(), &list_nodes.front()));
							list_nodes.front().lit = list_nodes.begin();
							
						}
					}
					if (n2.vkeys.size() > 0) {
						list_nodes.push_front(n2);
						if (n2.vkeys.size() > 1) {
							to_expand_num++;
							key_size_and_node.push_back(std::make_pair(n2.vkeys.size(), &list_nodes.front()));
							list_nodes.front().lit = list_nodes.begin();
						}
					}
					if (n3.vkeys.size() > 0) {
						list_nodes.push_front(n3);
						if (n3.vkeys.size() > 1) {
							to_expand_num++;
							key_size_and_node.push_back(std::make_pair(n3.vkeys.size(), &list_nodes.front()));
							list_nodes.front().lit = list_nodes.begin();
						}
					}
					if (n4.vkeys.size() > 0) {
						list_nodes.push_front(n4);
						if (n4.vkeys.size() > 1) {
							to_expand_num++;
							key_size_and_node.push_back(std::make_pair(n4.vkeys.size(), &list_nodes.front()));
							list_nodes.front().lit = list_nodes.begin();
						}
					}

					lit = list_nodes.erase(lit);
					continue;
				}

			}

			feature_num_per_level.resize(levels_num);
			float factor = 1.0f / scale_factor;
			float desired_feature_per_scale = features_num * (1 - factor) / (1 - (float)pow((double)factor, (double)levels_num));
			int sum_features = 0;
			for (int i = 0; i < levels_num - 1; ++i) {
				feature_num_per_level[i] = cvRound(desired_feature_per_scale);
				sum_features += feature_num_per_level[i];
				desired_feature_per_scale *= factor;
			}
			feature_num_per_level[levels_num - 1] = std::max(features_num - sum_features, 0);

			if ((int)list_nodes.size() >= features_num || (int)list_nodes.size() == pre_size) {
				is_finish = true;

			}
			 
			else if (((int)list_nodes.size() + to_expand_num * 3) > feature_num_per_level[level]) {
				while (!is_finish) {
					pre_size = list_nodes.size();
					vector<pair<int, ExtractorNode*> > prve_size_and_node_adderss = key_size_and_node;
					key_size_and_node.clear();

					sort(prve_size_and_node_adderss.begin(), prve_size_and_node_adderss.end());

					for (int j = prve_size_and_node_adderss.size() - 1; j >= 0; --j) {
						ExtractorNode n1, n2, n3, n4;
						prve_size_and_node_adderss[j].second->DivideNode(n1, n2, n3, n4);

						if (n1.vkeys.size() > 0) {
							list_nodes.push_front(n1);
							if (n1.vkeys.size() > 1) {
								key_size_and_node.push_back(std::make_pair(n1.vkeys.size(), &list_nodes.front()));
								list_nodes.front().lit = list_nodes.begin();
							}
						}

						if (n2.vkeys.size() > 0) {
							list_nodes.push_front(n2);
							if (n2.vkeys.size() > 1) {
								key_size_and_node.push_back(std::make_pair(n2.vkeys.size(), &list_nodes.front()));
								list_nodes.front().lit = list_nodes.begin();
							}
						}
						if (n3.vkeys.size() > 0) {
							list_nodes.push_front(n3);
							if (n3.vkeys.size() > 1) {
								key_size_and_node.push_back(std::make_pair(n3.vkeys.size(), &list_nodes.front()));
								list_nodes.front().lit = list_nodes.begin();
							}
						}
						if (n4.vkeys.size() > 0) {
							if (n4.vkeys.size() > 1) {
								key_size_and_node.push_back(std::make_pair(n4.vkeys.size(), &list_nodes.front()));
								list_nodes.front().lit = list_nodes.begin();
							}
						}

						if ((int)list_nodes.size() >= feature_num_per_level[level]) {
							break;
						}
					}
					if ((int)list_nodes.size() >= features_num || (int)list_nodes.size() == pre_size)
						is_finish = true;
				}

			}
		}
		std::vector<cv::KeyPoint> result_keys;
		result_keys.reserve(feature_num_per_level[level]);

		for (std::list<ExtractorNode>::iterator lit = list_nodes.begin(); lit != list_nodes.end(); lit++)
		{
			vector<cv::KeyPoint>& node_keys = lit->vkeys;
			cv::KeyPoint* keypoint = &node_keys[0];
			float max_response = keypoint->response;
			for (size_t k = 1; k < node_keys.size(); ++k)
			{
				if (node_keys[k].response > max_response)
				{
					keypoint = &node_keys[k];
					max_response = node_keys[k].response;
				}
			}

			result_keys.push_back(*keypoint);

		}

		keypoints = result_keys;

		const int scale_patch_size = PATCH_SIZE * vec_scale_per_factor[level];

		const int kps = keypoints.size();
		for (int l = 0; l < kps; ++l)
		{
			keypoints[l].pt.x += min_boder_x;
			keypoints[l].pt.y += min_boder_y;
			keypoints[l].octave = level;           
			keypoints[l].size = scale_patch_size; 
		}
		std::cout << "经过四叉数筛选，第 " << level + 1 << " 层剩余 " << result_keys.size() << " 个特征点" << endl;

	}

	int num_keypoints = 0;
	for (int level = 0; level < levels_num; ++level)
	{
		num_keypoints += (int)all_keypoints[level].size();
	}
	std::cout << "total " << num_keypoints << " keypoints" << endl;

	vector<cv::KeyPoint> out_put_all_keypoints(num_keypoints);

	for (int level = 0; level < levels_num; ++level)
	{
		if (level == 0)
		{
			for (int i = 0; i < all_keypoints[level].size(); ++i)
			{
				out_put_all_keypoints.push_back(all_keypoints[level][i]);
			}
		}
		float scale = vec_scale_per_factor[level];
		for (vector<cv::KeyPoint>::iterator key = all_keypoints[level].begin(); key != all_keypoints[level].end(); key++)
		{
			key->pt *= scale;
		}
		out_put_all_keypoints.insert(out_put_all_keypoints.end(), all_keypoints[level].begin(), all_keypoints[level].end());
	}
	return out_put_all_keypoints;
}


#define cut_rows 3   //分块行
#define cut_cols 3   //分块列
vector<cv::KeyPoint> Block_detector(Mat img) {
	Mat srcImg = img;

	vector<Mat> ceilImg;//存放切割图的向量

	int height = srcImg.rows;//原图的高
	int width = srcImg.cols;//原图的宽

	//(cut_rows-1)*(cut_cols-1)的图像块大小应该是一样的，最右边一列和最下边一行图像块大小可能不一样
	int ceil_height = (int)(height / cut_rows);
	int ceil_width = (int)(width / cut_cols);

	int ceil_down_height = height - (cut_rows - 1) * ceil_height;//最下边图像块的高
	int ceil_right_width = width - (cut_cols - 1) * ceil_width;//最右边图像块的宽

	for (int i = 0; i < cut_rows - 1; i++)    //不包括最下边一行
		for (int j = 0; j < cut_cols; j++)
		{
			if (j < cut_cols - 1)
			{
				Rect rect(j * ceil_width, i * ceil_height, ceil_width, ceil_height);
				ceilImg.push_back(srcImg(rect));

			}
			else   //最右边一列（不包括右下角）
			{
				Rect rect((cut_cols - 1) * ceil_width, i * ceil_height, ceil_right_width, ceil_height);
				ceilImg.push_back(srcImg(rect));
			}
		}

	for (int i = 0; i < cut_cols; i++)//最下边一行（包括右下角），包含一行多列情况
	{
		if (i < cut_cols - 1)
		{
			Rect rect(i * ceil_width, (cut_rows - 1) * ceil_height, ceil_width, ceil_down_height);
			ceilImg.push_back(srcImg(rect));
		}
		else   //右下角这个图像块
		{
			Rect rect((cut_cols - 1) * ceil_width, (cut_rows - 1) * ceil_height, ceil_right_width, ceil_down_height);
			ceilImg.push_back(srcImg(rect));
		}
	}
	std::cout << "分块个数：" << ceilImg.size() << endl;

	std::vector<cv::KeyPoint> keypointTotal;
	cv::Mat dst;

	for (int i = 0; i < ceilImg.size(); i++) {
		/*
		const int widthBlockNum = srcImg.cols / ceilImg[i].cols;
		const int heighBlockNum = srcImg.rows / ceilImg[i].rows;
		*/

		cout << "图像共被分成了" << ceilImg.size() << "块" << "现在正在对第" << i << "个分块进行操作" << endl;
		std::vector<cv::KeyPoint> keypoints0;
		for (int algo_i = 0; algo_i < 3; algo_i++) {
			Ptr<Feature2D> detector;
			if (algo_i == 0) {
				double hessianThreshold = 200;
				int nOctaves = 4, nOctaveLayers = 3;
				bool extended = false, upright = false;
				detector = xfeatures2d::SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);

			}
			else if (algo_i == 1) {
				detector = ORB::create(5000, 1.15, 14, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

			}
			else if (algo_i == 2) {
				int thresh = 30, octaves = 3;
				float patternScale = 1.0f;
				detector = BRISK::create(thresh, octaves, patternScale);

			}
			detector->detect(ceilImg[i], keypoints0);

			if (keypoints0.size() > 300) {
				auto rng = std::default_random_engine{};
				std::shuffle(std::begin(keypoints0), std::end(keypoints0), rng);
				keypoints0 = std::vector<cv::KeyPoint>(keypoints0.begin(), keypoints0.begin() + 300);
			}

			std::cout << "keypoints  size in subImage :" << i << "is :" << keypoints0.size() << endl;
		}
		dst = ceilImg[i];

		for (std::vector<cv::KeyPoint>::iterator vit = keypoints0.begin(); vit != keypoints0.end(); vit++)
		{

			(*vit).pt.x += ceilImg[i].cols * (i % cut_cols);
			(*vit).pt.y += ceilImg[i].rows * (i / cut_rows);

			keypointTotal.push_back(*vit);
		}

	}
	std::cout << "total number of keypointTotal is=============:" << keypointTotal.size() << endl;

	return keypointTotal;

}

bool expandEdge(const Mat& img, int edge[], const int edgeID)
{
	//[1] --初始化参数
	int nc = img.cols;
	int nr = img.rows;
	switch (edgeID) {
	case 0:
		if (edge[0] > nr)
			return false;
		for (int i = edge[3]; i <= edge[1]; ++i)
		{
			if (img.at<uchar>(edge[0], i) == 255)//遇见255像素表明碰到边缘线
				return false;
		}
		edge[0]++;
		return true;
		break;
	case 1:
		if (edge[1] > nc)
			return false;
		for (int i = edge[2]; i <= edge[0]; ++i)
		{
			if (img.at<uchar>(i, edge[1]) == 255)//遇见255像素表明碰到边缘线
				return false;
		}
		edge[1]++;
		return true;
		break;
	case 2:
		if (edge[2] < 0)
			return false;
		for (int i = edge[3]; i <= edge[1]; ++i)
		{
			if (img.at<uchar>(edge[2], i) == 255)//遇见255像素表明碰到边缘线
				return false;
		}
		edge[2]--;
		return true;
		break;
	case 3:
		if (edge[3] < 0)
			return false;
		for (int i = edge[2]; i <= edge[0]; ++i)
		{
			if (img.at<uchar>(i, edge[3]) == 255)//遇见255像素表明碰到边缘线
				return false;
		}
		edge[3]--;
		return true;
		break;
	default:
		return false;
		break;
	}

}

void method()
{
	String dir2 = "D:\\途\\OpenCv\\项目材料\\交付汇总-5901\\游戏-568";
	String dir = "D:\\途\\OpenCv\\项目材料\\snapshot_total\\游戏类";
	std::vector<cv::String> files;
	std::vector<cv::String> files2;
	cv::glob(dir, files);
	cv::glob(dir2, files2);
	bool findGame = false;
	string game = "游戏-568";
	const int flowSz = 768;
    int n = 0;
	for (int i = 0; i < files.size(); i++) {
		String fOriginal = files[i].substr(files[i].find_last_of("\\") + 1);
		String originalNameMatcher = files[i].substr(files[i].find("+") + 1);

		for (int j = 0; j < files2.size(); ++j) {
			String fFinder = files2[j].substr(files2[j].find_last_of("\\") + 1);
			String findNameMatcher = files2[j].substr(files2[j].find("+") + 1);
			
			if (!originalNameMatcher.compare(findNameMatcher)) {
				string leftPath = dir + "\\" + fOriginal;
				string rightPath = dir2 + "\\" + fFinder;

				std::cout << "拍照路径" << leftPath << endl;
				std::cout << "截图路径" << rightPath << endl;

				
				Mat photo = imread(leftPath);
				Mat screamShot = imread(rightPath);

				if (screamShot.empty() || photo.empty()) {
					printf("read frames failed!\n");
				}
				std::cout<<"第"<<n<<"组匹配"<<endl;

				cv::GaussianBlur(photo, photo, Size(5, 5), 0);
				cv::GaussianBlur(screamShot, screamShot, Size(3, 3), 0);

				//Size changer
				Size certainsize = Size(screamShot.cols, screamShot.rows);
				Mat Photo,temp;
				Mat ScreamShot;
				cv::transpose(photo,temp);
				cv::flip(temp, Photo, 0);
				cv::resize(Photo, Photo, certainsize);
				cv::resize(screamShot, ScreamShot, certainsize);

				vector<Mat> rvecs, tvecs;
				vector<float> reprojErrs;
				Mat cameraMatrix;
				Mat distCoeffs;
				vector<Point2f> psL, psR;
				vector<vector<Point3f> > objectPoints;
				vector<vector<Point2f> > imagePoints;
				vector<KeyPoint> keypointsL, keypointsR;
			
				psL.clear(); psR.clear();
				keypointsL.clear(); keypointsR.clear();
				Mat img0Filterd, img1Filterd;

						vector<KeyPoint> keypoints_ScreamShot, keypoints1_Photo;

						keypoints_ScreamShot = Block_detector(ScreamShot);
						keypoints1_Photo = Block_detector(Photo);

						Ptr<BRISK> Descriptor = BRISK::create();

						Mat imageDesc1, imageDesc2;
						Descriptor->compute(ScreamShot, keypoints_ScreamShot, imageDesc1);
						Descriptor->compute(Photo, keypoints1_Photo, imageDesc2);

						FlannBasedMatcher matcher;
						vector<vector<DMatch> > matchePoints;
						vector<DMatch> GoodMatchePoints,goodMatches;

						if (imageDesc1.type() != CV_32F|| imageDesc2.type() != CV_32F)
						{
							imageDesc1.convertTo(imageDesc1, CV_32F);
							imageDesc2.convertTo(imageDesc2, CV_32F);
						}
						matcher.match(imageDesc1, imageDesc2, GoodMatchePoints);

						goodMatches.clear();
						float minDist = std::min_element(
							GoodMatchePoints.begin(), GoodMatchePoints.end(),
							[](const cv::DMatch& m1, const cv::DMatch& m2) {
								return m1.distance < m2.distance;
							})->distance;
						double distThresh = max<float>(3 * minDist, 30.0);

						for (int j = 0; j < GoodMatchePoints.size(); j++) {
							// todo: cross check
							// retain those matches with enough small distance
							if (GoodMatchePoints[j].distance <= distThresh) {
								goodMatches.push_back(GoodMatchePoints[j]);
							}
						}

						for (const DMatch& match : goodMatches) {
							psL.push_back(keypoints_ScreamShot[match.queryIdx].pt);
							psR.push_back(keypoints1_Photo[match.trainIdx].pt);
						}
			
				Mat vis = visualizeKeypointMatches(ScreamShot, Photo, psL, psR);
                
				//至此匹配结束

				std::vector<uchar> inliersMask; //内样本点
				float reprojectionThreshold = min(max(ScreamShot.cols, ScreamShot.rows) * 0.01, 20.0);
				Mat H = cv::findHomography(psL, psR, RANSAC, reprojectionThreshold, inliersMask); //用来求取“射影变换”的H转制矩阵函数  X'=H X ，并使用RANSAC消除一些出错的点
				objectPoints.clear(); imagePoints.clear();
				objectPoints.resize(1);
				imagePoints.resize(1);
				for (int j = 0; j < psL.size(); j++) {
					if (inliersMask[j] > 0) {
						const Point2f& p = psL[j];
						objectPoints[0].push_back(Point3f(p.x, p.y, 0));
					}
				}
				for (int j = 0; j < psR.size(); j++) {
					if (inliersMask[j] > 0) {
						const Point2f& p = psR[j];
						imagePoints[0].push_back(p);
					}
				}

				cameraMatrix = Mat::eye(3, 3, CV_64F);
				distCoeffs = Mat::zeros(8, 1, CV_64F);
				//初始化两个矩阵
				double rms = calibrateCamera(objectPoints, imagePoints, Photo.size(), cameraMatrix,
					distCoeffs, rvecs, tvecs, 0); //相机标定
				std::cout << "rms:" << rms << endl;
				std::cout << "cameraMatrix:" << cameraMatrix << endl;
				std::cout << "distCoeffs:" << distCoeffs << endl;
				std::cout << "rvecs:" << rvecs[0] << endl;
				std::cout << "tvecs:" << tvecs[0] << endl;
				//Mat warp;
				//warpPerspective(imgL, warp, H, imgR.size(), INTER_CUBIC, BORDER_CONSTANT);
				Mat warp0, warp1;

				//Mat  map1, map2;
				//initUndistortRectifyMap(
				//	cameraMatrix, distCoeffs, Mat(),
				//	getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imgR.size(), 1, imgR.size(), 0), imgR.size(),
				//	CV_16SC2, map1, map2);


				//remap(imgR, warp1, map1, map2, INTER_LINEAR);

				//Mat psDistort, psOri = Mat(psR);
				//undistortPoints(psOri, psDistort, cameraMatrix, distCoeffs);

				Mat xyz(ScreamShot.size(), CV_32FC3);
				for (int y = 0; y < ScreamShot.rows; y++)
					for (int x = 0; x < ScreamShot.cols; x++)
					{
						xyz.at<Vec3f>(y, x)[0] = x;
						xyz.at<Vec3f>(y, x)[1] = y;
						xyz.at<Vec3f>(y, x)[2] = 0;//Z平面
					}
				

				xyz = xyz.reshape(0, xyz.size().area());
				Mat mapToSrc(xyz.size().area(), 1, CV_32FC2);
				cv::projectPoints(xyz, rvecs[0], tvecs[0], cameraMatrix, distCoeffs, mapToSrc);//透视图转俯视图
				Mat maps[2];
				mapToSrc = mapToSrc.reshape(0, ScreamShot.rows);
				cv::split(mapToSrc, maps); //通道分离

				//apply map
				cv::remap(Photo, warp1, maps[0], maps[1], INTER_CUBIC);
				Mat msk = Mat::zeros(Photo.size(), CV_32FC3) + Scalar(1.0, 1.0, 1.0);
				cv::remap(msk, msk, maps[0], maps[1], INTER_LINEAR);//像素重映射
				
				cv::min(msk, 1.0, msk);
				cv::max(msk, 0.0, msk);

				ScreamShot.convertTo(warp0, CV_32FC3);
				cv::multiply(warp0, msk, warp0);
				
				warp0.convertTo(warp0, CV_8UC3);

				Mat mskPts;
				msk.convertTo(msk, CV_8UC3, 255);
				cv::cvtColor(msk, msk, COLOR_BGR2GRAY);
				
				vector<Point2f> featPtsUsed;
				for (auto p : objectPoints[0]) {
					featPtsUsed.push_back(Point2f(p.x, p.y));
				} 


				Rect bbox/*(0, 0, ScreamShot.cols , ScreamShot.rows); */= boundingRect(featPtsUsed);
				

				int cx = int(bbox.x + bbox.width * 0.5);
				int cy = int(bbox.y + bbox.height * 0.5);

				vector<Vec4f> box_cands;
				Vec4f box = Vec4f(bbox.x - cx, bbox.y - cy, bbox.br().x - cx, bbox.br().y - cy);

				vector<double> ratios({ 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7 });
				for (auto ratio : ratios) {
					Vec4f box_cur;
					Vec4f r_coord;
					if (ratio <= 1.0) r_coord = Vec4f(1.0, ratio, 1.0, ratio);
					else r_coord = Vec4f(1.0 / ratio, 1.0, 1.0 / ratio, 1.0);
					box_cur = box.mul(r_coord);
					cout << "bbox(Rect)" << bbox << "bbox(Vec4f)" << box << ". box_cur:" << box_cur << ". ratio:" << r_coord << endl;;
					for (int j = 1; j <= 50; j++) {
						float factor = 1.0 / 50.0 * j;
						box_cands.push_back(factor * box_cur);
					}
				}
				for (auto& b : box_cands) {
					b = Vec4f(b[0] + cx, b[1] + cy, b[2] + cx, b[3] + cy);
				}
				std::sort(std::begin(box_cands), std::end(box_cands), [&](Vec4f bi, Vec4f bj) {
					return (bi[2] - bi[0]) * (bi[3] - bi[1]) > (bj[2] - bj[0]) * (bj[3] - bj[1]);
					});


				Rect  box_select;
				
				Point XLYT(INT16_MAX, INT16_MAX);
				Point XRYB(INT16_MIN, INT16_MIN);

				for (std::vector<cv::KeyPoint>::iterator vit = keypoints_ScreamShot.begin(); vit != keypoints_ScreamShot.end(); vit++)
				{
					if ((*vit).pt.x < XLYT.x) {
						XLYT.x = (*vit).pt.x;

					}
					if ((*vit).pt.y < XLYT.y) {
						XLYT.y = (*vit).pt.y;

					}
					if ((*vit).pt.x > XRYB.x) {
						XRYB.x = (*vit).pt.x;

					}
					if ((*vit).pt.y > XRYB.y) {
						XRYB.y = (*vit).pt.y;

					}

				}

				cout << "Position for XLYT is" << "(" << XLYT.x << "," << XLYT.y << ")" << endl;
				cout << "Position for XRYB is" << "(" << XRYB.x << "," << XRYB.y << ")" << endl;
				
				Rect box_generater(XLYT.x, XLYT.y, XRYB.x - XLYT.x, XRYB.y - XLYT.y);
				box_select = box_generater;

				/*Mat mask_inv = (255 - msk) / 255;
				for (auto b : box_cands) {
					Rect cur_rect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
					float r = cv::mean(mask_inv(cur_rect))[0];
					cout << "cur_bbox: " << b << ". r:" << r << endl;

					if (r < 0.001) { box_select = cur_rect; break; }
				}*/
				cout << "box_select:" << box_select << endl;

				warp0 = warp0(box_select).clone();
				warp1 = warp1(box_select).clone();
				
				
				circle(ScreamShot, XLYT, 5, Scalar(0, 255, 0), -1);
				circle(ScreamShot, XRYB, 5, Scalar(0, 0, 255), -1);

				cv::rectangle(Photo, box_select, Scalar(0, 0, 255), 5, LINE_AA, 0);
				cv::rectangle(ScreamShot, box_select, Scalar(0, 0, 255), 5, LINE_AA, 0);
				rectangle(msk, bbox, Scalar(0, 0, 255), 5, LINE_AA, 0);
				

				Mat flowInput0, flowInput1, flow;
				cv::GaussianBlur(warp0, flowInput0, Size(3, 3), 0);
				cv::GaussianBlur(warp1, flowInput1, Size(3, 3), 0);
				cv::resize(flowInput0, flowInput0, Size(flowSz, flowSz));
				cv::resize(flowInput1, flowInput1, Size(flowSz, flowSz));

				calc_flow(flowInput0, flowInput1, flow);

				cv::min(flow, flowSz * 0.02, flow);
				cv::max(flow, -flowSz * 0.02, flow);
				cv::GaussianBlur(flow, flow, Size(5, 5), 0); 
				vector<Mat> uv;
				cv::split(flow, uv);
				Mat& u = uv[0];
				Mat& v = uv[1];
				u = u * warp0.cols / float(u.cols);
				v = v * warp0.cols / float(v.cols);
				cv::merge(uv, flow);
				//cv::pyrDown(flow, flow);
				//cv::pyrDown(flow, flow);
				//cv::pyrDown(flow, flow);
				//cv::pyrDown(flow, flow);
				cv::resize(flow, flow, warp0.size());

				Mat xmap(flow.size(), CV_32F), ymap(flow.size(), CV_32F);
				for (int y = 0; y < flow.rows; y++) {
					for (int x = 0; x < flow.cols; x++) {
						Point2f motion = flow.at<Point2f>(y, x);
						//cout << motion << endl;
						xmap.at<float>(y, x) = x + motion.x;
						ymap.at<float>(y, x) = y + motion.y;
					}
				}

				Mat warp;
				cv::remap(warp1, warp, xmap, ymap, INTER_CUBIC, BORDER_REPLICATE);
 				n++;
			}
			else {
				continue;
			}
		}
	}
}


int main() {
	
	method();
	return 0;

}
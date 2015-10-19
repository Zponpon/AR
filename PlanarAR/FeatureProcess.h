#ifndef FEATUREPROCESS_H
#define FEATUREPROCESS_H

#include <iostream>
#include <windows.h>
#include <GL/gl.h>
#include "glut.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/video/video.hpp"
#include "PoseEstimation.h"
#include "debugfunc.h"

using std::cout;
using std::endl;
using cv::imshow;
using cv::waitKey;
using cv::imwrite;

//void SurfDetection(cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, unsigned int minHessian);
void FlannMatching(cv::Mat &descriptors1, cv::Mat &descriptors2, std::vector<cv::DMatch> &matches);
bool FeatureDetection(FeatureMap &featureMap, unsigned int minHessian);
bool FeatureDetection(unsigned int minHessian, FrameMetaData &currData, cv::Mat &currFrameImg);

//For homography
bool FeatureMatching(FeatureMap &featureMap, FrameMetaData &currFrame, cv::Mat &currFrameImg, cv::Mat &prevFrameImg, std::vector<cv::Point2f> &featureMapGoodMatches, std::vector<cv::Point2f> &frameGoodMatches, std::vector<cv::Point2f> &prevFeatureMapGoodMatches, std::vector<cv::Point2f> &prevFrameGoodMatches);

//For solvePnPRansac
bool FeatureMatching(double *cameraPara, std::vector<KeyFrame> &keyFrames, FrameMetaData &currData, cv::Mat &currFrameImg, std::vector<int> &neighboringKeyFrameIdx, std::vector< std::vector<cv::DMatch> > &goodMatchesSet);

//For triangulation
void FeatureMatching(KeyFrame &query, KeyFrame &train, std::vector<cv::DMatch> &goodMatches);


#endif
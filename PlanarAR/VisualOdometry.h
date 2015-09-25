#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include <vector>
#include "KeyFrameSelection.h"
//#include "FeatureProcess.h"
#include "PoseEstimation.h"
//#include "SFMUtil.h"
using std::vector;
//extern vector<EstimateCamInfos> cameraPose;

//void CreateFeatureMap(FeatureMap &featureMap, int minHessian);
void LoadFeatureMap(int argc, char *argv[]);
bool VOD(double *cameraPara, double trans[3][4], FeatureMap &featureMap, vector<KeyFrame> &keyFrames, cv::Mat &prevFrameImg, cv::Mat &currFrameImg);
//void VOD();

#endif
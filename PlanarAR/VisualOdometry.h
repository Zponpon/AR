#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include <vector>
#include "KeyFrame.h"
//#include "FeatureProcess.h"
#include "PoseEstimation.h"
//#include "SFMUtil.h"
using std::vector;

//void CreateFeatureMap(FeatureMap &featureMap, int minHessian);
void LoadFeatureMap(int argc, char *argv[]);
bool VO(double *cameraPara, double trans[3][4], FeatureMap &featureMap, cv::Mat &prevFrameImg, cv::Mat &currFrameImg, char &m);
//void VOD();

#endif
#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include <vector>
#include "KeyFrame.h"
//#include "FeatureProcess.h"
#include "PoseEstimation.h"
//#include "SFMUtil.h"
using std::vector;

//void CreateFeatureMap(FeatureMap &featureMap, int minHessian);
void CreateFeatureMaps(FeatureMap &featureMap, unsigned int minHessian);
void LoadFeatureMaps(int argc, char *argv[]);
void StopMultiThread();
bool VO(double *cameraPara, double trans[3][4], FeatureMap &featureMap, cv::Mat &prevFrameImg, cv::Mat &currFrameImg, char &m);
//void VOD();

#endif
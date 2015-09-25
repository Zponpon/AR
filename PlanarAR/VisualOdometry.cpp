#include "glut.h"
#include "VisualOdometry.h"
#include "FeatureProcess.h"
#include "SFMUtil.h"
static vector<cv::Point2f> prevFrameInliers, prevFeatureMapInliers;
static vector<EstimateCamInfos> cameraPoseInfos;
static vector<FeatureMap> featureMaps;
//static vector<KeyFrame> keyFrames

void LoadFeatureMap(int argc, char *argv[])
{
	//Loading featureMap
}

bool VOD(double *cameraPara, double trans[3][4], FeatureMap &featureMap, vector<KeyFrame> &keyFrames, cv::Mat &prevFrameMat, cv::Mat &currFrameMat)
{
	Frame currFrame;
	if (!FeatureDetection(3000, currFrame, currFrameMat)) return false;

	vector<cv::Point2f> currFrameGoodMatches, featureMapGoodMatches;
	vector<int> goodKeyFrameIdx;
	vector< vector<cv::DMatch> > goodMatchesSet;
	
	if (FeatureMatching(featureMap, currFrame, currFrameMat, prevFrameMat, featureMapGoodMatches, currFrameGoodMatches, prevFeatureMapInliers, prevFrameInliers))
	{
		EstimateCameraTransformation(cameraPara, trans, featureMap, currFrame, featureMapGoodMatches, currFrameGoodMatches, prevFeatureMapInliers, prevFrameInliers);

		//frames.push_back(currFrame);

		if (keyFrames.size() == 0)
			CreateKeyFrame(cameraPara, currFrame, currFrameMat, keyFrames);
	}
	else if (FeatureMatching(cameraPara, keyFrames, currFrame, currFrameMat, goodKeyFrameIdx, goodMatchesSet))
	{
		EstimateCameraTransformation(cameraPara, trans, keyFrames, currFrame, goodKeyFrameIdx, goodMatchesSet);

		//frames.push_back(currFrame);
		//frameInfos.push_back(currFrame);
	}
	else return false;
	if (KeyFrameSelection(keyFrames, currFrame))
	{
		CreateKeyFrame(cameraPara, currFrame, currFrameMat, keyFrames);
		if (keyFrames.size() == 2)
			Triangulation(keyFrames[0], keyFrames[1], cameraPara);
		else
			Triangulation(cameraPara, keyFrames, goodKeyFrameIdx);
	}
	return true;
}
#include <thread>
#include <ctime>
#include <chrono>
#include <mutex>
#include "VisualOdometry.h"
#include "FeatureProcess.h"
#include "SFMUtil.h"

static vector<cv::Point2f> prevFrameInliers, prevFeatureMapInliers;
static vector<FrameMetaData> frameMetaDatas;
static vector<FeatureMap> featureMaps;
static vector<KeyFrame> keyFrames;
static std::thread Optimization;


void LoadFeatureMaps(int argc, char *argv[])
{
	//Read a file
	//Loading featureMap
}

void CreateFeatureMaps(FeatureMap &featureMap, unsigned int minHessian)
{
	if (FeatureDetection(featureMap, minHessian))
	{
		for (std::vector<cv::KeyPoint>::size_type i = 0; i < featureMap.keypoints.size(); ++i)
		{
			featureMap.keypoints[i].pt.x -= featureMap.image.cols / 2;
			featureMap.keypoints[i].pt.y -= featureMap.image.rows / 2;

			// Because image y coordinate is positive in downward direction
			featureMap.keypoints[i].pt.y = -featureMap.keypoints[i].pt.y; 
		}
	}
}

void StopMultiThread()
{
	using std::chrono::system_clock;
	std::this_thread::sleep_until(system_clock::now());
}

void RemoveRedundancyIdx()
{
	for (vector<KeyFrame>::iterator it = keyFrames.begin(); it != keyFrames.end(); ++it)
	{
		//it->coresIdx
	}
}

bool VO(double *cameraPara, double trans[3][4], FeatureMap &featureMap, cv::Mat &prevFrameMat, cv::Mat &currFrameMat, char &m)
{	
	cout << "KeyFrame Set : " << keyFrames.size() << endl;
	if (Optimization.joinable())
		Optimization.join();
	
	FrameMetaData currData;
	if (!FeatureDetection(3000, currData, currFrameMat)) return false;

	vector<cv::Point2f> currFrameGoodMatches, featureMapGoodMatches;
	vector<int> neighboringKeyFrameIdx;
	vector< vector<cv::DMatch> > goodMatchesSet;
	
	if (FeatureMatching(featureMap, currData, currFrameMat, prevFrameMat, featureMapGoodMatches, currFrameGoodMatches, prevFeatureMapInliers, prevFrameInliers))
	{
		EstimateCameraTransformation(cameraPara, trans, featureMap, currData, featureMapGoodMatches, currFrameGoodMatches, prevFeatureMapInliers, prevFrameInliers);
		if (keyFrames.size() == 0)
			CreateKeyFrame(cameraPara, currData, currFrameMat, keyFrames);
	}
	else if (FeatureMatching(cameraPara, keyFrames, currData, currFrameMat, neighboringKeyFrameIdx, goodMatchesSet))
	{
		EstimateCameraTransformation(cameraPara, trans, keyFrames, currData, neighboringKeyFrameIdx, goodMatchesSet);
		if (currData.state == 'F')
			return false;
	}
	else return false;
	if (KeyFrameSelection(keyFrames.back(), currData))
	{
		CreateKeyFrame(cameraPara, currData, currFrameMat, keyFrames);
		//Triangulation(cameraPara, keyFrames);
		Optimization = std::thread(Triangulation, cameraPara, ref(keyFrames));
		RemoveRedundancyIdx();
	}
	m = currData.state;
	frameMetaDatas.push_back(currData);

	return true;
}
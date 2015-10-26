#include <thread>
#include <ctime>
#include <chrono>
#include <map>
#include "VisualOdometry.h"
#include "FeatureProcess.h"
#include "SFMUtil.h"

static vector<cv::Point2f> prevFrameInliers, prevFeatureMapInliers;
static vector<FrameMetaData> frameMetaDatas;
//static vector<FeatureMap> featureMaps;
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
		//Initialize the featureMap
		for (std::vector<cv::KeyPoint>::size_type i = 0; i < featureMap.keypoints.size(); ++i)
		{
			featureMap.keypoints[i].pt.x -= featureMap.image.cols / 2;
			featureMap.keypoints[i].pt.y -= featureMap.image.rows / 2;
			featureMap.keypoints[i].pt.y = -featureMap.keypoints[i].pt.y; // Because image y coordinate is positive in downward direction
		}
	}
}

void StopMultiThread()
{
	using std::chrono::system_clock;
	std::this_thread::sleep_until(system_clock::now());
}

char EstimationMethod()
{
	/*	Get the latest estimation method	*/
	return frameMetaDatas.back().state;
}

void RemoveRedundancyIdx()
{
	/*	刪除重複的對應點	*/
	for (vector<KeyFrame>::iterator KF = keyFrames.begin(); KF != keyFrames.end(); ++KF)
	{
		std::map<int, cv::Point3d> indexMap;

		int i = (int)KF->coresIdx.size() - 1;
		int j = (int)KF->r3dPts.size() - 1;
		while (i & j)
		{
			indexMap.insert(std::pair<int, cv::Point3d>(KF->coresIdx[i--], KF->r3dPts[j--]));
		}
		KF->coresIdx.clear();
		KF->r3dPts.clear();

		for (std::map<int, cv::Point3d>::iterator itMap = indexMap.begin(); itMap != indexMap.end(); ++itMap)
		{
			KF->coresIdx.push_back(itMap->first);
			KF->r3dPts.push_back(itMap->second);
		}
	}
}

bool VO(double *cameraPara, double trans[3][4], FeatureMap &featureMap, cv::Mat &prevFrameMat, cv::Mat &currFrameMat)
{	
	/*	KeyFrameSelection仍然有問題	*/
	cout << "KeyFrame Sets : " << keyFrames.size() << endl;
	if (Optimization.joinable())
		Optimization.join();//等待multithread的部分做完
	
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
	else
	{
		currData.state = 'F';
		return false;
	}

	if (KeyFrameSelection(keyFrames.back(), currData))
	{
		CreateKeyFrame(cameraPara, currData, currFrameMat, keyFrames);
		Optimization = std::thread(Triangulation, cameraPara, ref(keyFrames));//multithread
		RemoveRedundancyIdx();
	}

	frameMetaDatas.push_back(currData);

	return true;
}
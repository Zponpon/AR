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
static vector <Measurement> measurementData;
static MyMatrix K(3, 3); // Camera Matrix

void SetCameraMatrix(double *cameraPara)
{
	for (int i = 0; i < 9; ++i)
		K.m_lpdEntries[i] = cameraPara[i];
}

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
	WriteMeasurementDataFile(measurementData);
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
			indexMap.insert(std::pair<int, cv::Point3d>(KF->coresIdx[(std::vector<int>::size_type)i], KF->r3dPts[(std::vector<int>::size_type)j]));
			i--;
			j--;
		}
		KF->coresIdx.clear();
		KF->r3dPts.clear();

		for (std::map<int, cv::Point3d>::iterator index = indexMap.begin(); index != indexMap.end(); ++index)
		{
			KF->coresIdx.push_back(index->first);
			KF->r3dPts.push_back(index->second);
		}
	}
}

/*
void KeyFrameTesting()
{
	keyFrames.resize(5);
	keyFrames[0].image = cv::imread("KeyFrames/KeyFrames1.jpg");
	keyFrames[1].image = cv::imread("KeyFrames/KeyFrames2.jpg");
	keyFrames[2].image = cv::imread("KeyFrames/KeyFrames3.jpg");
	keyFrames[3].image = cv::imread("KeyFrames/KeyFrames4.jpg");
	keyFrames[5].image = cv::imread("KeyFrames/KeyFrames5.jpg");

}
*/

bool VO(double *cameraPara, double trans[3][4], FeatureMap &featureMap, cv::Mat &prevFrameMat, cv::Mat &currFrameMat)
{	
	if (keyFrames.size() == 0)
		SetCameraMatrix(cameraPara);
	if (Optimization.joinable())
	{
		//	Multithread
		Optimization.join(); //	wait another thread completes
		RemoveRedundancyIdx();
	}
	
	FrameMetaData currData; //	Record the current frame data
	if (!FeatureDetection(3000, currData, currFrameMat)) return false;

	vector<cv::Point2f> currFrameGoodMatches, featureMapGoodMatches;
	vector<int> neighboringKeyFrameIdx;
	vector< vector<cv::DMatch> > goodMatchesSet;

	if (FeatureMatching(featureMap, currData, currFrameMat, prevFrameMat, featureMapGoodMatches, currFrameGoodMatches, prevFeatureMapInliers, prevFrameInliers))
	{
		EstimateCameraTransformation(cameraPara, trans, featureMap, currData, featureMapGoodMatches, currFrameGoodMatches, prevFeatureMapInliers, prevFrameInliers);
		if (keyFrames.size() == 0)
		{
			CreateKeyFrame(K, currData, currFrameMat, keyFrames);
			currData.state = 'H';
			frameMetaDatas.push_back(currData);
			return true;
		}
	}
	else if (FeatureMatching(cameraPara, keyFrames, currData, currFrameMat, neighboringKeyFrameIdx, goodMatchesSet))
	{
		EstimateCameraTransformation(cameraPara, trans, keyFrames, currData, neighboringKeyFrameIdx, goodMatchesSet);
		if (currData.state == 'F')
			return false;
	}
	else return false;

	if (KeyFrameSelection(K, keyFrames.back(), currData, measurementData))
	{
		if (keyFrames.size() < 3)
		{
			CreateKeyFrame(K, currData, currFrameMat, keyFrames);
			Triangulation(cameraPara, keyFrames);
			//Optimization = std::thread(Triangulation, cameraPara, ref(keyFrames));
		}
	}

	frameMetaDatas.push_back(currData);

	return true;
}
#include <thread>
#include <ctime>
#include <chrono>
#include <map>
#include "VisualOdometry.h"
#include "FeatureProcess.h"
#include "SFMUtil.h"

#pragma region Variables
static vector<cv::Point2f> prevFrameInliers, prevFeatureMapInliers;
static vector<FrameMetaData> frameMetaDatas;
//static vector<FeatureMap> featureMaps;
static vector<KeyFrame> keyframes;
static std::thread Optimization;
static vector <Measurement> measurementData;
static MyMatrix K(3, 3); // Camera Matrix
static vector<cv::Point3d> r3dPts;	//world coordinate 3d points
static vector<SFM_Feature> SFM_Features;
vector<int> neighboringKeyFrameIdx;
#pragma endregion


void LoadFeatureMaps(int argc, char *argv[])
{
	//Read a file
	//Loading featureMap
}

void CreateFeatureMaps(FeatureMap &featureMap, unsigned int minHessian)
{
	if (FeatureDetection(featureMap, minHessian))
	{
		//	Initialize the featureMap
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
	std::this_thread::sleep_until(std::chrono::system_clock::now());
	WriteMeasurementDataFile(measurementData);
}

PoseEstimationMethod EstimationMethod()
{
	//	Get the latest estimation method
	//	Homography, Ransac PnP or Fail

	return frameMetaDatas.back().method;
}

bool VO(double *cameraPara, double trans[3][4], FeatureMap &featureMap, cv::Mat &prevFrameMat, cv::Mat &currFrameMat)
{	
	#pragma region Initialize
	if (keyframes.size() == 0)
	{
		for (int i = 0; i < 9; ++i)
			K.m_lpdEntries[i] = cameraPara[i];
	}
	
	cout << "KeyFrame count : " << keyframes.size() << endl;

	FrameMetaData currData;
	if (!FeatureDetection(currData, currFrameMat, 3000)) return false;

	vector<cv::Point2f> currFrameGoodMatches, featureMapGoodMatches;
	//vector<int> neighboringKeyFrameIdx;
	vector< vector<cv::DMatch> > goodMatchesSet;
	#pragma endregion

	if (FeatureMatching(featureMap, currData, currFrameMat, prevFrameMat, featureMapGoodMatches, currFrameGoodMatches, prevFeatureMapInliers, prevFrameInliers))
	{
		//	Homography
		EstimateCameraTransformation(cameraPara, trans, featureMap, currData, featureMapGoodMatches, currFrameGoodMatches, prevFeatureMapInliers, prevFrameInliers);

		//	First keyframe
		if (keyframes.size() == 0)
		{
			CreateKeyFrame(K, currData, currFrameMat, keyframes);
			//neighboringKeyFrameIdx.push_back(0);
			frameMetaDatas.push_back(currData);
			return true;
		}
	}
	else 
	{
		if (Optimization.joinable())
		{
			//	Multithread
			//	Wait for another thread completes
			//Optimization.join();
		}

		if (FeatureMatching(cameraPara, SFM_Features, keyframes, currData, currFrameMat, neighboringKeyFrameIdx, goodMatchesSet))
		{
			//	Ransac PnP
			EstimateCameraTransformation(cameraPara, trans, r3dPts, keyframes, currData, neighboringKeyFrameIdx, goodMatchesSet);
		}
		else
		{
			currData.method = PoseEstimationMethod::Fail;
		}
	}


	//	Save all frames
	frameMetaDatas.push_back(currData);

	if (currData.method == PoseEstimationMethod::Fail)	return false;

	if (KeyFrameSelection(keyframes.back(), currData, measurementData))
	{
		if (keyframes.size() < 4)
		{
			CreateKeyFrame(K, currData, currFrameMat, keyframes);
			neighboringKeyFrameIdx.push_back((int)keyframes.size() - 1);
			
			//	Multithread
			//Optimization = std::thread(Triangulation, cameraPara, std::ref(SFM_Features), std::ref(keyframes), std::ref(r3dPts));
			
			Triangulation(cameraPara, SFM_Features, keyframes, r3dPts);
		}
	}

	return true;
}
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
static vector<KeyFrame> keyframes;
static std::thread Optimization;
static vector <Measurement> measurementData;
static MyMatrix K(3, 3); // Camera Matrix
static vector<cv::Point3d> r3dPts;	//world coordinate 3d points
static vector<SFM_Feature> SFM_Features;
vector<int> neighboringKeyFrameIdx;

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

bool VO(double *cameraPara, double trans[3][4], FeatureMap &featureMap, cv::Mat &prevFrameMat, cv::Mat &currFrameMat)
{	
	if (keyframes.size() == 0)
		SetCameraMatrix(cameraPara);
	/*
	if (Optimization.joinable())
	{
		//	Multithread
		Optimization.join(); //	wait for another thread completes
		RemoveRedundancyCoresIdx();
	}
	*/

	cout << "KeyFrame count : " << keyframes.size() << endl;
	FrameMetaData currData; //	Record the current frame data
	if (!FeatureDetection(currData, currFrameMat, 3000)) return false;

	vector<cv::Point2f> currFrameGoodMatches, featureMapGoodMatches;
	//vector<int> neighboringKeyFrameIdx;
	vector< vector<cv::DMatch> > goodMatchesSet;

	if (FeatureMatching(featureMap, currData, currFrameMat, prevFrameMat, featureMapGoodMatches, currFrameGoodMatches, prevFeatureMapInliers, prevFrameInliers))
	{
		EstimateCameraTransformation(cameraPara, trans, featureMap, currData, featureMapGoodMatches, currFrameGoodMatches, prevFeatureMapInliers, prevFrameInliers);
		if (keyframes.size() == 0)
		{
			CreateKeyFrame(K, currData, currFrameMat, keyframes);
			neighboringKeyFrameIdx.push_back(0);
			//currData.state = 'H';
			currData.method = PoseEstimationMethod::ByHomography;
			frameMetaDatas.push_back(currData);
			return true;
		}
	}
	else if (FeatureMatching(cameraPara, SFM_Features, keyframes, currData, currFrameMat, neighboringKeyFrameIdx, goodMatchesSet))
	{
		EstimateCameraTransformation(cameraPara, trans, r3dPts, keyframes, currData, neighboringKeyFrameIdx, goodMatchesSet);

		if (currData.method == PoseEstimationMethod::Fail) return false;
	}
	else
	{
		//currData.state = 'F';
		currData.method = PoseEstimationMethod::Fail;
		cout << "FeatureMatching failed\n";
		frameMetaDatas.push_back(currData);
		return false;
	}

	if (KeyFrameSelection(K, keyframes.back(), currData, measurementData))
	{
		if (keyframes.size() < 2)
		{
			neighboringKeyFrameIdx.push_back((int)keyframes.size() - 1);
			CreateKeyFrame(K, currData, currFrameMat, keyframes);
			Triangulation(cameraPara, SFM_Features, keyframes, r3dPts);
			//Optimization = std::thread(Triangulation, cameraPara, ref(SFM_Features), ref(keyFrames), ref(r3dpts));
		}
	}

	frameMetaDatas.push_back(currData);

	return true;
}
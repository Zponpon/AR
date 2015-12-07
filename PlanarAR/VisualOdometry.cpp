#include <thread>
#include <ctime>
#include <chrono>
#include <map>
#include "VisualOdometry.h"
#include "FeatureProcess.h"
#include "SFMUtil.h"

#pragma region Variables
static vector<cv::Point2f> prevFrameInliers, prevFeatureMapInliers;
static unsigned int *lastCount = NULL;
static std::thread Optimization;

static vector<FrameMetaData> frameMetaDatas;
static vector<KeyFrame> keyframes;

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

void InitializeBundlerApp(double f)
{
	static MyBundlerApp *bundlerApp = NULL;
	bundlerApp = new MyBundlerApp();
	
	bundlerApp->Initialize((int)keyframes.size());

	for (std::vector<KeyFrame>::size_type i = 0; i < keyframes.size(); ++i)
	{
		bundlerApp->SetImageData((int)i, keyframes[i].image.cols, keyframes[i].image.rows, f);
	}

	//EstablishImageCorrespondences(SFM_Features, keyframes);
}

void InitializeKeyFrame(unsigned int FrameCount, double *cameraPara, cv::Mat &currFrameMat)
{
	if (lastCount == NULL)
	{
		for (int i = 0; i < 9; ++i)
			K.m_lpdEntries[i] = cameraPara[i];
	}
	
	unsigned int interval = 0;
	FrameMetaData currData;

	if (lastCount != NULL)
		interval = FrameCount - *lastCount;

	if (interval > 20 || lastCount == NULL)
	{
		if (lastCount == NULL)
			lastCount = new unsigned(FrameCount);
		else
		{
			*lastCount = FrameCount;
			if (!FeatureDetection(currData, currFrameMat, 3000))
				return;

			CreateKeyFrame(K, currData, currFrameMat, keyframes);
		}
	}

	if (keyframes.size() == 4)
	{
		delete lastCount;
		//BudlerApp
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

int GetKeyFrameSize()
{
	return (int)keyframes.size();
}

bool VO(unsigned int FrameCount, double *cameraPara, double trans[3][4], FeatureMap &featureMap, cv::Mat &prevFrameMat, cv::Mat &currFrameMat)
{	
	#pragma region Initialize
	
	cout << "KeyFrame count : " << keyframes.size() << endl;

	if (keyframes.size() < 4) return false;
	
	FrameMetaData currData;
	if (!FeatureDetection(currData, currFrameMat, 3000)) return false;

	vector<cv::Point2f> currFrameGoodMatches, featureMapGoodMatches;
	//vector<int> neighboringKeyFrameIdx;
	vector< vector<cv::DMatch> > goodMatchesSet;
	#pragma endregion

	if (Optimization.joinable())
	{
		//	Multithread
		//	Wait for another thread completes
		//Optimization.join();
	}

	if (FeatureMatching(SFM_Features, keyframes, currData, currFrameMat, neighboringKeyFrameIdx, goodMatchesSet))
	{
		//	Ransac PnP
		EstimateCameraTransformation(K, trans, r3dPts, keyframes, currData, neighboringKeyFrameIdx, goodMatchesSet);
	}
	else
	{
		currData.method = PoseEstimationMethod::Fail;
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
#ifndef VIDEO_H
#define VIDEO_H

#include "videoInput.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/video/tracking.hpp"
#include "opencv2/video/video.hpp"

extern int videoWidth, videoHeight;
extern videoInput VI;
extern int dev;
extern unsigned char *frame;

void InitVideoDeviceOpenCV(int DesiredVideoWidth, int DesiredVideoHeight);
void InitVideoDevice(int DesiredVideoWidth, int DesiredVideoHeight);
void SetupVideo(void);
void StopVideoDevice(void);
void WriteVideo(cv::VideoWriter &VW, unsigned char *frame, unsigned long FrameCount);

#endif
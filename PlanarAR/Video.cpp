#include "Video.h"
#include "debugfunc.h"

videoInput VI;
int dev =0;
int videoWidth, videoHeight;
unsigned char * frame=NULL;

void InitVideoDevice(int DesiredVideoWidth, int DesiredVideoHeight)
{
	//videoInput::listDevices();	
	videoInput::setVerbose(false);
	VI.setUseCallback(false);

	VI.setupDevice(dev,  DesiredVideoWidth, DesiredVideoHeight, VI_COMPOSITE); 
	VI.setIdealFramerate(dev, 60); //­ì³]¬°60

	videoWidth = VI.getWidth(dev);
	videoHeight = VI.getHeight(dev);

	frame = new unsigned char[VI.getSize(dev)];

	printf("Video Width=%d video Height = %d\n",videoWidth,videoHeight);
}

void SetupVideo(void)
{
	VI.showSettingsWindow(dev);
}

void StopVideoDevice(void)
{
	VI.stopDevice(dev);
	if(frame!=NULL) delete [] frame;
}
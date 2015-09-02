#include <windows.h>
#include <fstream>
#include <GL/gl.h>
#include "glut.h"
#include <AR/gsub.h>
#include <ctime>
#include "PoseEstimation.h"
#include "KeyFrameSelection.h"
#include "FeatureProcess.h"
#include "DebugFunc.h"
#include "Video.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#define VIEW_VOLUME_NEAR 1.0
#define VIEW_VOLUME_FAR 3000.0

#define CAMERA_ORIENTATION_POSITIVE_Z 1
#define CAMERA_ORIENTATION_NEGATIVE_Z -1
//#define WRITEVIDEO
//#define DEBUG

int winWidth=800, winHeight=600;
unsigned char *pbImage;
unsigned long FrameCount = 0;
unsigned char *prevFrame = NULL;
//double camera_para[9] = { 738.41709, 0.00000, 378.50000, 0.00000, 733.88828, 341.50000, 0.00000, 0.00000, 1.00000 };
double camera_para[9] = { 9.1317151001595698e+002, 0.00000, 3.9695336273339319e+002, 0.00000, 9.1335671139215276e+002, 2.9879860363446750e+002, 0.00000, 0.00000, 1.00000 };

FeatureMap featureMap;
cv::VideoWriter writer("GoProTestVideo.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10.0, cv::Size(800, 600));

clock_t t_start,t_end;

//-----For triangulate
Frame keyFrame1;		//k-2
Frame keyFrame2;		//k-1
std::vector<Frame > keyFrames(2);
void InitOpenGL(void)
{
	glClearColor(1.0,1.0,1.0,0.0);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glEnable(GL_DEPTH_TEST);
}

void DrawMode2D(int winWidth, int winHeight)
{          
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, winWidth, 0.0, winHeight);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, winWidth, winHeight);
}

void DrawMode3D(double camera_para[9], int winWidth, int winHeight, bool flipImage, int orientation)
{
	double invK[9];
	double left,right,bottom,top;
	double u,v,w;

	glViewport(0, 0, winWidth, winHeight);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// Compute the inverse of camera intrinsic matrix
	invK[0] = 1.0/camera_para[0];
	invK[1] = -camera_para[1]/(camera_para[0]*camera_para[4]);
	invK[2] = (camera_para[1]*camera_para[5]-camera_para[4]*camera_para[2])/(camera_para[0]*camera_para[4]*camera_para[8]);
	invK[4] = 1.0/camera_para[4];
	invK[5] = -camera_para[5]/(camera_para[4]*camera_para[8]);
	invK[3] = invK[6] = invK[7] = 0.0;
	invK[8] = 1.0/camera_para[8];

	// [u v w]' = invK*[0 0 1]'
	u = invK[2]; v=invK[5]; w=invK[8]; 
	left = u/w*VIEW_VOLUME_NEAR;
	bottom = v/w*VIEW_VOLUME_NEAR;

	// [x y z]' = invK*[Width Height 1]'
	u =  invK[0]*winWidth+invK[1]*winHeight+invK[2];
	v =  invK[3]*winWidth+invK[4]*winHeight+invK[5];
	w =  invK[6]*winWidth+invK[7]*winHeight+invK[8];
	right = u/w*VIEW_VOLUME_NEAR;
	top = v/w*VIEW_VOLUME_NEAR;

	if(orientation == CAMERA_ORIENTATION_NEGATIVE_Z)
	{
		if(flipImage) glFrustum(left,right,top,bottom,VIEW_VOLUME_NEAR,VIEW_VOLUME_FAR);
		else glFrustum(left,right,bottom,top,VIEW_VOLUME_NEAR,VIEW_VOLUME_FAR);
	}

	if(orientation == CAMERA_ORIENTATION_POSITIVE_Z)
	{
		double gl_para[16];

		gl_para[0] = 2*VIEW_VOLUME_NEAR/(right-left);
		gl_para[1] = 0.0;
		gl_para[2] = 0.0;
		gl_para[3] = 0.0;
		gl_para[4] = 0.0;
		gl_para[5] = 2*VIEW_VOLUME_NEAR/(top-bottom);
		gl_para[6] = 0.0;
		gl_para[7] = 0.0;
		gl_para[8] = (right+left)/(left-right);
		gl_para[9] = (bottom+top)/(bottom-top);
		gl_para[10] = (VIEW_VOLUME_FAR+VIEW_VOLUME_NEAR)/(VIEW_VOLUME_FAR-VIEW_VOLUME_NEAR);
		gl_para[11] = 1.0;
		gl_para[12] = 0.0;
		gl_para[13] = 0.0;
		gl_para[14] = 2*VIEW_VOLUME_NEAR*VIEW_VOLUME_FAR/(VIEW_VOLUME_NEAR-VIEW_VOLUME_FAR);
		gl_para[15] = 0.0;

		if(flipImage) 
		{
			gl_para[5] = -gl_para[5];
			gl_para[9] = -gl_para[9];
		}
		glLoadMatrixd(gl_para);
	}
}

void DisplayImage(unsigned char *image)
{
	glRasterPos2i(0, 0);
	glDrawPixels(winWidth, winHeight, GL_RGB, GL_UNSIGNED_BYTE, image);
}

void DisplayFlipImage(unsigned char *image)
{
	glPixelZoom(1.0, -1.0);  // For flipped video
	glRasterPos2i(0, winHeight); // For fliped video
	glDrawPixels(winWidth, winHeight, GL_RGB, GL_UNSIGNED_BYTE, image);
}

void draw_axes(double size)
{
	int isEnableLighting = 0;

	if(glIsEnabled(GL_LIGHTING))
	{
		isEnableLighting = 1;
		glDisable(GL_LIGHTING);
	}

	glBegin(GL_LINES);
	glColor3d(1.0,0.0,0.0);
	glVertex3d(0.0,0.0,0.0);
	glVertex3d(size,0.0,0.0);
	glColor3d(0.0,1.0,0.0);
	glVertex3d(0.0,0.0,0.0);
	glVertex3d(0.0,size,0.0);
	glColor3d(0.0,0.0,1.0);
	glVertex3d(0.0,0.0,0.0);
	glVertex3d(0.0,0.0,size);
	glEnd();

	int isEnableColorMaterial = 1;

	if(!glIsEnabled(GL_COLOR_MATERIAL))
	{
		isEnableColorMaterial = 0;
		glEnable(GL_COLOR_MATERIAL);
	}

	glEnable(GL_LIGHTING);

	glColor3d(1.0,0.0,0.0);
	glPushMatrix();
	glTranslated(size,0.0,0.0);
	glRotated(90.0,0.0,1.0,0.0);
	glutSolidCone(0.05*size, 0.15*size, 16, 32);
	glPopMatrix();

	glColor3d(0.0,1.0,0.0);
	glPushMatrix();
	glTranslated(0.0,size,0.0);
	glRotated(-90.0,1.0,0.0,0.0);
	glutSolidCone(0.05*size, 0.15*size, 16, 32);
	glPopMatrix();

	glColor3d(0.0,0.0,1.0);
	glPushMatrix();
	glTranslated(0.0,0.0,size);
	glutSolidCone(0.05*size, 0.15*size, 16, 32);
	glPopMatrix();

	if(isEnableLighting == 1) glEnable(GL_LIGHTING);
	else glDisable(GL_LIGHTING);

	if(isEnableColorMaterial == 0) glDisable(GL_COLOR_MATERIAL);
	else glEnable(GL_COLOR_MATERIAL);
}

void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glDisable(GL_DEPTH_TEST);
	DrawMode2D(winWidth,winHeight);

#ifndef DEBUG
	if (FrameCount > 0)	//因為第二張frame開始才需要記錄前一張frame
	{
		prevFrame = frame;
		frame = new unsigned char[VI.getSize(dev)];
	}
#endif

	if( VI.isFrameNew(dev))	
	{
		VI.getPixels(dev, frame, true, true);

		/*prevFrame = new unsigned char[VI.getSize(dev)];
		WriteVideo(writer, frame, FrameCount);*/
#ifdef DEBUG
		if (FrameCount == 0)
		{
			DebugLoadImage("img1.raw", frame, winWidth, winHeight, 3);
			prevFrame = frame;
		}
		else if (FrameCount == 1)
		{
			frame = new unsigned char[VI.getSize(dev)];
			DebugLoadImage("img2.raw", frame, winWidth, winHeight, 3);
		}
		else 
			DebugLoadImage("img2.raw", frame, winWidth, winHeight, 3);
#endif
		DisplayFlipImage(frame);
	}
	else return;

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	GLfloat   light_position[]  = {100.0,-200.0,200.0,0.0};
	
	double trans[3][4];
	double gl_para[16];

	bool rtn = EstimateCameraTransformation(keyFrames, prevFrame, &frame, winWidth, winHeight, featureMap, camera_para, trans);

	if (rtn == true)
	{
		argConvGlpara(trans, gl_para);

		DrawMode3D(camera_para, winWidth, winHeight, true, CAMERA_ORIENTATION_POSITIVE_Z);

		GLdouble projection[16];
		glGetDoublev(GL_PROJECTION_MATRIX, projection);

		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixd(gl_para);

		glLightfv(GL_LIGHT0, GL_POSITION, light_position);

		//draw_axes(100.0);

		glTranslated(0.0, 0.0, 50.0);
		glRotated(90.0, 1.0, 0.0, 0.0);
		glutWireTeapot(100.0);
		//glutWireCube(50.0);
		//glutSolidSphere(10.0,16,32);
	}
	/*else
	{
		if (keyFrame1.image.data && keyFrame2.image.data)
		{
			if (EstimateCameraTransformation(featureMap, keyFrame1, keyFrame2, frame, winWidth, winHeight, camera_para, trans))
			{
				Triangulation(keyFrame1, keyFrame2, featureMap, camera_para);
				argConvGlpara(trans, gl_para);

				DrawMode3D(camera_para, winWidth, winHeight, true, CAMERA_ORIENTATION_POSITIVE_Z);

				GLdouble projection[16];
				glGetDoublev(GL_PROJECTION_MATRIX, projection);

				glMatrixMode(GL_MODELVIEW);
				glLoadMatrixd(gl_para);

				glLightfv(GL_LIGHT0, GL_POSITION, light_position);
				glTranslated(400.0, 300.0, 50.0);
				glutWireCube(100.0);
			}
		}
	}*/
	glutSwapBuffers();
	FrameCount++;
#ifndef DEBUG
	if (prevFrame!=NULL)
		delete[]prevFrame;//刪除前一次的記憶體位址
#endif
}

void KeyboardFunc(unsigned char key, int x, int y)
{
	if(key==0x1b || key=='Q' || key=='q')
	{
		t_end=clock();
		printf("Frame rate = %f \n",FrameCount*(double)CLK_TCK/((double)(t_end-t_start)));
		StopVideoDevice();
		
		glutDestroyWindow(glutGetWindow());
		exit(0);
	}
	
	if(key=='S' || key=='s')
	{
		SetupVideo();
		return;
	}
}

void main(int argc, char *argv[])
{
#ifndef WRITEVIDEO
	InitVideoDevice(winWidth,winHeight);
#endif
#ifdef WRITEVIDEO
	WriteVideo();
#endif
	glutInit(&argc,argv);

	glutInitDisplayMode(GLUT_DOUBLE|GLUT_DEPTH|GLUT_RGB); 
	glutInitWindowSize(winWidth,winHeight);
	glutInitWindowPosition(100,100);
	glutCreateWindow("Planar AR");

	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutKeyboardFunc(KeyboardFunc);

	InitOpenGL();
	featureMap.image = cv::imread("11647241_636517923150104_1570377205_n.jpg");
	CreateFeatureMap(featureMap, 5000);

	t_start = clock();

	glutMainLoop();
}
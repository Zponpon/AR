#ifndef HOMOGRAPHY_H
#define HOMOGRAPHY_H

#ifndef BASICTYPE_H
#error "BasicType.h should be included before Homography.h"
#endif

#ifndef MYMATRIX_H
#error "MyMatrix.h should be included before Homography.h"
#endif

void Homography(Point2d *pPoints1, Point2d *pPoints2, int nPoints, MyMatrix &H);
void HomographyWithNormalization(Point2d *pPoints1, Point2d *pPoints2, int nPoints, MyMatrix &H);
int LMedSHomography(Point2d *pPoints1, Point2d *pPoints2, int nPoints, MyMatrix &H, BYTE *pbInliers);
int AdaptiveLMedSHomography(Point2d *pPoints1, Point2d *pPoints2, int nPoints, MyMatrix &H, BYTE *pbInliers);

#endif
// Matrix.h: interface for the Matrix class.

#if !defined(AFX_MATRIX_H__F92ABF41_D3FA_11D4_B7DE_10AD49C10000__INCLUDED_)
#define AFX_MATRIX_H__F92ABF41_D3FA_11D4_B7DE_10AD49C10000__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#ifndef MYMATRIX_H
#define MYMATRIX_H

#define MYMATRIX_OK 0
#define MYMATRIX_FAIL -1

class MyMatrix  
{
public:
	int m_iM,m_iN;
	double *m_lpdEntries;
public:
	MyMatrix();
	MyMatrix(int M,int N);
	MyMatrix(const MyMatrix &M);
	~MyMatrix();
	MyMatrix& operator=(const MyMatrix &mat);
	MyMatrix& operator*(const MyMatrix &mat);
	MyMatrix& operator*(const double a);
	MyMatrix& operator+(const MyMatrix &mat);
	MyMatrix& operator-(const MyMatrix &mat);
	//void operator=(MyMatrix& A);
	void CreateMatrix(int M, int N);
	void DestroyMatrix(void);
	int LUFactorization(int *lpiRowOrder);
	int QRFactorization(MyMatrix *Q, MyMatrix *R);
	int QRFactorization(MyMatrix &Q, MyMatrix &R);
	int FindPivotRow(int *lpiRowOrder,int row, int col);
	void SolveLinearEquation(int *lpiRowOrder,double *lpdBvector, double *lpdSolution);
	void RemoveRedundantRow(double *lpdBVector, int *ValidRowNumber, int *lpiRemovedRow);
	void Multiplication(MyMatrix *B, MyMatrix *C);
	void Transpose(MyMatrix *B);
	MyMatrix& MyMatrix::Transpose(void);
	int Eigenvalue_2D(double *eigvalue,double *eigvector);
	//int Eigenvalue(double *eigvalue,double *eigvector);
    void Eigenvalue(double *eigvalue,double *eigvector);
    void SVD(MyMatrix *U, MyMatrix *S, MyMatrix *V);
    void MLDivide(MyMatrix *B, MyMatrix *C);
	MyMatrix& MLDivide(MyMatrix *B);
	int Chol(MyMatrix *L);
	MyMatrix& Chol(void);
    int Inverse(MyMatrix *Inv);
	MyMatrix& Inverse(void);
    double Determine(void);
	void Show(void);
};
#endif

#endif // !defined(AFX_MATRIX_H__F92ABF41_D3FA_11D4_B7DE_10AD49C10000__INCLUDED_)

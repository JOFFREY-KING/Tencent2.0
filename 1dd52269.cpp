/*****************************

按空格键可以看到效果

******************************/
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

#define cut_rows 3   //分块行
#define cut_cols 3   //分块列

int main()
{
	Mat srcImg=imread("1.jpg");
	imshow("src",srcImg);

	vector<Mat> ceilImg;//存放切割图的向量

	int height=srcImg.rows;//原图的高
	int width=srcImg.cols;//原图的宽

	//(cut_rows-1)*(cut_cols-1)的图像块大小应该是一样的，最右边一列和最下边一行图像块大小可能不一样
	int ceil_height=(int)(height/cut_rows);
	int ceil_width=(int)(width/cut_cols);
	int ceil_down_height=height-(cut_rows-1)*ceil_height;//最下边图像块的高
	int ceil_right_width=width-(cut_cols-1)*ceil_width;//最右边图像块的宽

	for(int i=0;i<cut_rows-1;i++)    //不包括最下边一行
		for(int j=0;j<cut_cols;j++)
		{
			if(j<cut_cols-1)
			{
				Rect rect(j*ceil_width,i*ceil_height,ceil_width,ceil_height);
				ceilImg.push_back(srcImg(rect));
			
			}
			else   //最右边一列（不包括右下角）
			{
				Rect rect((cut_cols-1)*ceil_width,i*ceil_height,ceil_right_width,ceil_height);
				ceilImg.push_back(srcImg(rect));
			}
		}

	for(int i=0;i<cut_cols;i++)//最下边一行（包括右下角），包含一行多列情况
	{
		if(i<cut_cols-1)
		{
			Rect rect(i*ceil_width,(cut_rows-1)*ceil_height,ceil_width,ceil_down_height);
			ceilImg.push_back(srcImg(rect));
		}
		else   //右下角这个图像块
		{
			Rect rect((cut_cols-1)*ceil_width,(cut_rows-1)*ceil_height,ceil_right_width,ceil_down_height);
			ceilImg.push_back(srcImg(rect));
		}
	}

	cout<<"分块个数："<<ceilImg.size()<<endl;
	Mat dst;
	for(int i=0;i<ceilImg.size();i++)
	{
		dst=ceilImg[i];
		imshow("dst",dst); 
		waitKey(0);
	}

	waitKey(0);
	return 0;
}

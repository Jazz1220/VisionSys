 /*
 * Scratch.cpp
 *
 *  Created on: Sep 4, 2015
 *      Author: ubuntu
 */

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <libfreenect.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <pthread.h>

#define Iterations 2

int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 31;


using namespace cv;
using namespace std;

int display_dst( int delay );

class myMutex {
public:
	myMutex() {
		pthread_mutex_init(&m_mutex, NULL);
	}
	void lock() {
		pthread_mutex_lock(&m_mutex);
	}
	void unlock() {
		pthread_mutex_unlock(&m_mutex);
	}
private:
	pthread_mutex_t m_mutex;
};

class MyFreenectDevice: public Freenect::FreenectDevice {
public:
	MyFreenectDevice(freenect_context *_ctx, int _index) :
			Freenect::FreenectDevice(_ctx, _index), m_buffer_depth(
					FREENECT_DEPTH_11BIT), m_buffer_rgb(FREENECT_VIDEO_RGB), m_gamma(
					2048), m_new_rgb_frame(false), m_new_depth_frame(false), depthMat(
					Size(640, 480), CV_16UC1), rgbMat(Size(640, 480), CV_8UC3,
					Scalar(0)), ownMat(Size(640, 480), CV_8UC3, Scalar(0)) {

		for (unsigned int i = 0; i < 2048; i++) {
			float v = i / 2048.0;
			v = std::pow(v, 3) * 6;
			m_gamma[i] = v * 6 * 256;
		}
	}

	// Do not call directly even in child
	void VideoCallback(void* _rgb, uint32_t timestamp) {
		std::cout << "RGB callback" << std::endl;
		m_rgb_mutex.lock();
		uint8_t* rgb = static_cast<uint8_t*>(_rgb);
		rgbMat.data = rgb;
		m_new_rgb_frame = true;
		m_rgb_mutex.unlock();
	}
	;

	// Do not call directly even in child
	void DepthCallback(void* _depth, uint32_t timestamp) {
		std::cout << "Depth callback" << std::endl;
		m_depth_mutex.lock();
		uint16_t* depth = static_cast<uint16_t*>(_depth);
		depthMat.data = (uchar*) depth;
		m_new_depth_frame = true;
		m_depth_mutex.unlock();
	}

	bool getVideo(Mat& output) {
		m_rgb_mutex.lock();
		if (m_new_rgb_frame) {
			cv::cvtColor(rgbMat, output, CV_RGB2BGR);
			m_new_rgb_frame = false;
			m_rgb_mutex.unlock();
			return true;
		} else {
			m_rgb_mutex.unlock();
			return false;
		}
	}

	bool getDepth(Mat& output) {
		m_depth_mutex.lock();
		if (m_new_depth_frame) {
			depthMat.copyTo(output);
			m_new_depth_frame = false;
			m_depth_mutex.unlock();
			return true;
		} else {
			m_depth_mutex.unlock();
			return false;
		}
	}
private:
	std::vector<uint8_t> m_buffer_depth;
	std::vector<uint8_t> m_buffer_rgb;
	std::vector<uint16_t> m_gamma;
	Mat depthMat;
	Mat rgbMat;
	Mat ownMat;
	myMutex m_rgb_mutex;
	myMutex m_depth_mutex;
	bool m_new_rgb_frame;
	bool m_new_depth_frame;
};

void morphOps(Mat &thresh)
{
    //Create structuring element that will be used to "dilate" and "erode" image.
 
    //MORPH_RECT - a rectangular structuring element
 
    //Element chosen here is a 5px by 5px rectangle for erosion
    Mat erodeElement = getStructuringElement( MORPH_RECT, Size(5,5));
 
    //Element chosen here is a 1px by 1px rectangle for dilation
    Mat dilateElement = getStructuringElement( MORPH_RECT, Size(1,1));
 
    //erode(thresh, thresh, erodeElement, Point(-1,-1), Iterations);
    erode(thresh, thresh, erodeElement, Point(-1,-1), 1);
 
    dilate(thresh, thresh, dilateElement, Point(-1,-1), Iterations);
}
 

int main() {
	initModule_features2d();
	bool die(false);
	int iter(0);
	int i_snap(0);
	string filename("snapshot");
	string suffix(".png");
	Freenect::Freenect freenect;
	vector<KeyPoint> Keypoints;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);
	OrbFeatureDetector detector;
	Mat depthMat(Size(640, 480), CV_16UC1);
	Mat bgrMat(Size(640, 480), CV_8UC3, Scalar(0));
	Mat bgrMatPost(Size(640, 480), CV_8UC3, Scalar(0));
	Mat depthf(Size(640, 480), CV_8UC1);
	Mat ownMat(Size(640, 480), CV_8UC3, Scalar(0));
	Mat inFrame(Size(640, 480), CV_8UC3, Scalar(0));
	Mat outFrame(Size(640, 480), CV_8UC3, Scalar(0));
	Mat postProc(Size(640, 480), CV_8UC3, Scalar(0));
	Mat hsvMat(Size(640, 480), CV_8UC3, Scalar(0));
	Mat rgbMat(Size(640, 480), CV_8UC3, Scalar(0));
	Mat threshMat(Size(640, 480), CV_8U, Scalar(0));
	Mat bgrThresh(Size(640, 480), CV_8UC1, Scalar(0));
	Mat greyMat(Size(640, 480), CV_8UC1, Scalar(0));
	Mat canny_output(Size(640, 480), CV_8UC1, Scalar(0));
	
	//HSV
	//Scalar min(0, 32, 0);
	//Scalar max(30, 255, 255);
	
	
	//RGB
	//Scalar min(101, 63, 21);
	//Scalar max(153, 75, 33);
	
	
	//BGR
	//Scalar min(0, 0, 0);
	//Scalar max(255, 0, 0);
	
	int lowThreshold;
	int edgeThresh = 1;
	
	namedWindow("Output", CV_WINDOW_AUTOSIZE);
	namedWindow("depth", CV_WINDOW_AUTOSIZE);
	namedWindow("Post", CV_WINDOW_AUTOSIZE);
	createTrackbar ("Min Threshold:", "Post", &lowThreshold, 100);//, CannyThreshold);

	device.startVideo();
	device.startDepth();
	
	while (die == false) {
		device.getVideo(bgrMat);
		//device.getVideo(postProc);
		
		device.getDepth(depthMat);
		cv::imshow("Output", bgrMat);
		
		//cv::cvtColor(bgrMat, outFrame, CV_BGR2GRAY);
		//detector.detect(outFrame, Keypoints);
		//drawKeypoints(bgrMat, Keypoints, postProc, Scalar(0, 0, 255)/*, DrawMatchesFlags::DRAW_RICH_KEYPOINTS*/);
		//cvSmooth( bgrMat, bgrMat,CV_GAUSSIAN,9,9, 0);
		

			//GaussianBlur( threshMat, threshMat, Size(1, 1), 0, 0);
		
		cvtColor(bgrMat, hsvMat, CV_BGR2HSV);
		//cvtColor(rgbMat, hsvMat, CV_RGB2HSV);
		
		
		
		inRange(hsvMat, Scalar(105, 75, 35), Scalar(135, 255, 255), threshMat);
		//cvtColor(bgrMat, greyMat, CV_BGR2GRAY);
		//erode(threshMat, threshMat2, 0, Point(0, 0), 4, BORDER_CONSTANT);
		//dilate( threshMat2, threshMat3, 0, Point (0, 0));
		//inRange(hsvMat, Scalar(165, 100, 0), Scalar(180, 255, 255), threshMat2);
		
		//cvtColor(hsvMat, bgrMat, CV_HSV2BGR);
		morphOps(threshMat);
		Canny (threshMat, canny_output, lowThreshold, lowThreshold*3, 3);
		findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		//adaptiveThreshold( greyMat, bgrThresh, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 13, 1);
		adaptiveThreshold( threshMat, bgrThresh, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, 1);
		
		
		
		detector.detect(threshMat, Keypoints);
		drawKeypoints(bgrMat, Keypoints, postProc, Scalar(0, 0, 255)/*, DrawMatchesFlags::DRAW_RICH_KEYPOINTS*/);
		drawContours( postProc, contours, -1, Scalar( 255, 255, 255), 2, 8, hierarchy, INT_MAX, Point(0, 0));
		
		cv::imshow("Post", postProc);
		//cv::imshow("Post", bgrThresh);
		//cv::imshow("Post", threshMat);
		//depthMat.convertTo(depthf, CV_8UC1, 255.0 / 2048.0);
		//cv::imshow("depth", depthf);

		char k = cvWaitKey(5);
		if (k == 27) {
			cvDestroyWindow("rgb");
			cvDestroyWindow("depth");
			cvDestroyWindow("Post");
			break;
		}
		if (k == 8) {
			std::ostringstream file;
			file << filename << i_snap << suffix;
			cv::imwrite(file.str(), bgrMat);
			i_snap++;
		}

		if (iter >= 10000)
			break;
		iter++;

	}
	device.stopVideo();
	device.stopDepth();
	return 0;

}


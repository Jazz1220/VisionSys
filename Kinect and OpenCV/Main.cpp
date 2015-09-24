/*
 * Main.cpp
 *
 *  Created on: Aug 28, 2015
 *      Author: ubuntu
 */
#undef min
#undef max


#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
#include <libfreenect.hpp>
#include <libfreenect.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <pthread.h>
#include <limits.h>
#include <time.h>
#include <iomanip>

#define Iterations 2

using namespace cv;
using namespace std;
using namespace cv::gpu;


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

void
morphOps(Mat & thresh)
{
    // Create structuring element that will be used to "dilate" and
    // "erode" image.

    // MORPH_RECT - a rectangular structuring element

    // Element chosen here is a 5px by 5px rectangle for erosion
    Mat             erodeElement =
	getStructuringElement(MORPH_RECT, Size(5, 5));

    // Element chosen here is a 1px by 1px rectangle for dilation
    Mat             dilateElement =
	getStructuringElement(MORPH_RECT, Size(1, 1));

    // erode(thresh, thresh, erodeElement, Point(-1,-1), Iterations);
    erode(thresh, thresh, erodeElement, Point(-1, -1), Iterations);

    dilate(thresh, thresh, dilateElement, Point(-1, -1), Iterations);
}

void gpuInRange(GpuMat src, GpuMat & threshMat,Scalar MinBGR, Scalar MaxBGR)
{

	GpuMat dst[3];
	GpuMat finishedMat;

	GpuMat Bmin;
	GpuMat Gmin;
	GpuMat Rmin;
	GpuMat Bmax;
	GpuMat Gmax;
	GpuMat Rmax;

	GpuMat CompBmin;
	GpuMat CompGmin;
	GpuMat CompRmin;
	GpuMat CompBmax;
	GpuMat CompGmax;
	GpuMat CompRmax;

	Gmin.setTo(Scalar(MinBGR.val[1]));
	Rmin.setTo(Scalar(MinBGR.val[2]));
	Bmin.setTo(Scalar(MinBGR.val[0]));
	Bmax.setTo(Scalar(MaxBGR.val[0]));
	Gmax.setTo(Scalar(MaxBGR.val[1]));
	Rmax.setTo(Scalar(MaxBGR.val[2]));
	//GpuMat mask(Scalar(255, 255, 255));
	GpuMat blue_bin;
	GpuMat green_bin;
	GpuMat red_bin;
	GpuMat bgBin;
	GpuMat grBin;
	gpu::split(src, dst);

	gpu::compare(dst[0], Bmin, CompBmin, CMP_GE);
	gpu::compare(dst[0], Bmax, CompBmax, CMP_LE);
	gpu::bitwise_and(CompBmin, CompBmax, blue_bin);

	gpu::compare(dst[1], Gmin, CompGmin, CMP_GE);
	gpu::compare(dst[1], Gmax, CompGmax, CMP_LE);
	gpu::bitwise_and(CompGmin, CompGmax, green_bin);

	gpu::compare(dst[2], Rmin, CompRmin, CMP_GE);
	gpu::compare(dst[2], Rmax, CompRmax, CMP_LE);
	gpu::bitwise_and(CompRmin, CompRmax, red_bin);

	gpu::bitwise_and(blue_bin, green_bin, bgBin);
	gpu::bitwise_and(green_bin, red_bin, grBin);
	gpu::bitwise_and(bgBin, grBin, threshMat);

	//blue_bin = (dst[0]<MinBGR.val[0]) && (dst[0]>MaxBGR.val[0]);
	//green_bin = (dst[1]<MinBGR.val[1]) && (dst[1]>MaxBGR.val[1]);
	//blue_bin = (dst[2]<MinBGR.val[2]) && (dst[2]>MaxBGR.val[2]);

	//threshMat = blue_bin & green_bin & red_bin;



}
int main(int argc, char **argv) {
	bool die(false);
	string filename("snapshot");
	string suffix(".png");
	int i_snap(0), iter(0);
	Mat depthMat(Size(640, 480), CV_16UC1);
	Mat depthf(Size(640, 480), CV_8UC1);
	Mat rgbMat(Size(640, 480), CV_8UC3, Scalar(0));
	Mat ownMat(Size(640, 480), CV_8UC3, Scalar(0));
	Mat hsvMat(Size(640, 480), CV_8UC3, Scalar(0));
	GpuMat threshMat;
	Mat threshMatCPU(Size(640, 480), CV_8U, Scalar(0));
	GpuMat gpuRGBMat;
	Freenect::Freenect freenect;
	MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);

	namedWindow("rgb", CV_WINDOW_AUTOSIZE);
	namedWindow("depth", CV_WINDOW_AUTOSIZE);
	namedWindow("postproc", CV_WINDOW_AUTOSIZE);
	device.startVideo();
	device.startDepth();
	while (!die) {
		device.getVideo(rgbMat);
		device.getDepth(depthMat);
		cv::imshow("rgb", rgbMat);
		gpuRGBMat.upload(rgbMat);
		gpuInRange(gpuRGBMat, threshMat, Scalar(0, 0, 0), Scalar (255, 100, 100));
		threshMat.download(threshMatCPU);
		imshow("post", threshMat);
		depthMat.convertTo(depthf, CV_8UC1, 255.0 / 2048.0);

		cv::imshow("depth", depthf);
		char k = cvWaitKey(5);
		if (k == 27) {
			cvDestroyWindow("rgb");
			cvDestroyWindow("depth");
			break;
		}
		if (k == 8) {
			std::ostringstream file;
			file << filename << i_snap << suffix;
			cv::imwrite(file.str(), rgbMat);
			i_snap++;
		}
		if (iter >= 1000)
			break;
		iter++;
	}

	device.stopVideo();
	device.stopDepth();
	return 0;
}

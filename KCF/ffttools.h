#pragma once

#ifndef FFTTOOLS_H
#define FFTTOOLS_H
#endif

#include "stdafx.h"

namespace FFTTools
{
	/* about FFT and complex process */


	// Previous declarations, to avoid warnings
	cv::Mat fftd(cv::Mat img, bool backwards = false);
	cv::Mat real(cv::Mat img);
	cv::Mat imag(cv::Mat img);
	cv::Mat magnitude(cv::Mat img);
	cv::Mat complexMultiplication(cv::Mat a, cv::Mat b);
	cv::Mat complexDivision(cv::Mat a, cv::Mat b);
	void rearrange(cv::Mat& img);
	void normalizedLogTransform(cv::Mat& img);


	cv::Mat fftd(cv::Mat img, bool backwards)
	{
		/**
		*  discretized fast fourier transformation
		*/
		if (img.channels() == 1)
		{
			cv::Mat planes[] = { cv::Mat_<float>(img), cv::Mat_<float>::zeros(img.size()) };
			cv::merge(planes, 2, img); //merge 2 arrays with input planes, and output img (one for real and the other is imaginary
		}
		cv::dft(img, img, backwards ? (cv::DFT_INVERSE | cv::DFT_SCALE) : 0); //discrete fourier transform 
		//backwards ? (cv::DFT_INVERSE | cv::DFT_SCALE) : 0
		//if backwards is true -> (cv::DFT_INVERSE | cv::DFT_SCALE) is done (inverse DFT and scaling), if backwards is false -> forward DFT

		return img;
	}

	cv::Mat real(cv::Mat img)
	{
		/* real part */
		std::vector<cv::Mat> planes;
		cv::split(img, planes); //multi-channel -> single channel, process of each plane is possible
		return planes[0]; //real part
	}

	cv::Mat imag(cv::Mat img)
	{
		/* imaginary part */
		std::vector<cv::Mat> planes;
		cv::split(img, planes);
		return planes[1];
	}

	cv::Mat magnitude(cv::Mat img)
	{
		/* magnitude */
		cv::Mat res;
		std::vector<cv::Mat> planes;
		cv::split(img, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
		if (planes.size() == 1) res = cv::abs(img);
		else if (planes.size() == 2) cv::magnitude(planes[0], planes[1], res); //compute the magnitude of a complex number from its real and imaginary parts.
		else assert(0);
		return res;
	}

	cv::Mat complexMultiplication(cv::Mat a, cv::Mat b)
	{
		std::vector<cv::Mat> pa;
		std::vector<cv::Mat> pb;
		cv::split(a, pa);
		cv::split(b, pb);

		std::vector<cv::Mat> pres;
		pres.push_back(pa[0].mul(pb[0]) - pa[1].mul(pb[1])); //(a+bi)(c+di)
		pres.push_back(pa[0].mul(pb[1]) + pa[1].mul(pb[0]));

		cv::Mat res;
		cv::merge(pres, res);

		return res;
	}

	cv::Mat complexDivision(cv::Mat a, cv::Mat b)
	{
		std::vector<cv::Mat> pa;
		std::vector<cv::Mat> pb;
		cv::split(a, pa);
		cv::split(b, pb);

		cv::Mat divisor = 1. / (pb[0].mul(pb[0]) + pb[1].mul(pb[1]));

		std::vector<cv::Mat> pres;

		pres.push_back((pa[0].mul(pb[0]) + pa[1].mul(pb[1])).mul(divisor)); //(a+bi)/(c+di) = ((ac+bd)+i(bc-ad))/(c^2+d^2)
		pres.push_back((pa[1].mul(pb[0]) - pa[0].mul(pb[1])).mul(divisor));

		cv::Mat res;
		cv::merge(pres, res);
		return res;
	}

	void rearrange(cv::Mat& img)
	{
		int cx = img.cols / 2;
		int cy = img.rows / 2;

		cv::Mat q0(img, cv::Rect(0, 0, cx, cy)); // Top-Left - Create a ROI per quadrant
		cv::Mat q1(img, cv::Rect(cx, 0, cx, cy)); // Top-Right
		cv::Mat q2(img, cv::Rect(0, cy, cx, cy)); // Bottom-Left
		cv::Mat q3(img, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

		cv::Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
		q0.copyTo(tmp);
		q3.copyTo(q0);
		tmp.copyTo(q3);
		q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
		q2.copyTo(q1);
		tmp.copyTo(q2);
	}

	void normalizedLogTransform(cv::Mat& img)
	{
		img = cv::abs(img);
		img += cv::Scalar::all(1);
		cv::log(img, img); //independently computes the logarithm of each element without considering the other elements
	}
}


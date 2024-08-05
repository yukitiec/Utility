#include "stdafx.h"
#include "tracker.h"
#include "kcftracker.h"
#include "ffttools.h"
#include "recttools.h"

const float thresh_score = 0.99;

// Constructor
KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab)
{

    // Parameters equal in all cases
    lambda = 0.0001;
    padding = 2.5;
    //output_sigma_factor = 0.1;
    output_sigma_factor = 0.125;

    if (hog) {    // HOG
        // VOT
        interp_factor = 0.012;
        sigma = 0.6;
        // TPAMI
        //interp_factor = 0.02;
        //sigma = 0.5; 
        cell_size = 4;
        _hogfeatures = true;
    }
    else {   // GRAY image -> RAW
        interp_factor = 0.075;
        sigma = 0.2;
        cell_size = 1;
        _hogfeatures = false;

        if (lab) {
            printf("Lab features are only used with HOG features.\n");
            _labfeatures = false;
        }
    }

    if (multiscale) { // multiscale
        template_size = 96;
        //template_size = 100;
        scale_step = 1.05;
        scale_weight = 0.95;
        if (!fixed_window) {
            //printf("Multiscale does not support non-fixed window.\n");
            fixed_window = true;
        }
    }
    else if (fixed_window) {  // fit correction without multiscale
        template_size = 96;
        //template_size = 100;
        scale_step = 1;
    }
    else {
        template_size = 1;
        scale_step = 1;
    }
}

// Initialize tracker 
void KCFTracker::init(const cv::Rect& roi, cv::Mat image)
{
    _roi = roi;
    assert(roi.width >= 0 && roi.height >= 0);
    //std::cout << "start initialization" << std::endl;
    _tmpl = getFeatures(image, true, 1.0f);
    //std::cout << "get features" << std::endl;
    _prob = createGaussianPeak(size_patch[0], size_patch[1]);
    //std::cout << "gaussian peak" << std::endl; 
    _alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    train(_tmpl, 1.0); // train with initial frame
}

// Update position based on the new frame
cv::Rect KCFTracker::update(cv::Mat image)
{
    //set boundary
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;
    //center
    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;
    //std::cout << "1" << std::endl;
    //response funtion -> peak_value is tracking score
    float peak_value;
    cv::Point2f res = detect(_tmpl, getFeatures(image, false, 1.0f), peak_value);
    //std::cout << "2" << std::endl;
    if (scale_step != 1) {
        // Test at a smaller _scale
        float new_peak_value;
        cv::Point2f new_res = detect(_tmpl, getFeatures(image, false, 1.0f / scale_step), new_peak_value);
        //std::cout << "3" << std::endl;
        if (scale_weight * new_peak_value > peak_value) {
            res = new_res;
            peak_value = new_peak_value;
            _scale /= scale_step;
            _roi.width /= scale_step;
            _roi.height /= scale_step;
        }
        //std::cout << "4" << std::endl;
        // Test at a bigger _scale
        new_res = detect(_tmpl, getFeatures(image, false,scale_step), new_peak_value);
        //std::cout << "5" << std::endl;
        if (scale_weight * new_peak_value > peak_value) {
            res = new_res;
            peak_value = new_peak_value;
            _scale *= scale_step;
            _roi.width *= scale_step;
            _roi.height *= scale_step;
        }
        //std::cout << "6" << std::endl;
    }

    // Adjust by cell size and _scale
    _roi.x = cx - _roi.width / 2.0f + ((float)res.x * cell_size * _scale); //move roi
    _roi.y = cy - _roi.height / 2.0f + ((float)res.y * cell_size * _scale); //move roi
    //adjust roi 
    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;
    //std::cout << "7" << std::endl;
    assert(_roi.width >= 0 && _roi.height >= 0);
    cv::Mat x = getFeatures(image, false, 1.0f);
    //std::cout << "8" << std::endl;
    if (peak_value > thresh_score)
        train(x, interp_factor); //train data
    _peak_value = peak_value; //save current peak_value
    std::cout << "peak_value = " << peak_value << std::endl;

    return _roi;
}

// Detect object in the current frame.
cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float& peak_value)
{
    using namespace FFTTools;

    cv::Mat k = gaussianCorrelation(x, z);//k_xz
    //std::cout << "detect-1" << std::endl;
    cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true))); //k_xz*alpha
    //std::cout << "detect-2" << std::endl;
    //minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
    cv::Point2i pi;
    double pv;
    cv::minMaxLoc(res, NULL, &pv, NULL, &pi); //get highest score
    //std::cout << "detect-3" << std::endl;
    peak_value = (float)pv;

    //subpixel peak estimation, coordinates will be non-integer
    cv::Point2f p((float)pi.x, (float)pi.y); //get peak position

    if (pi.x > 0 && pi.x < res.cols - 1) {
        p.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value, res.at<float>(pi.y, pi.x + 1)); //get most hiest score in pixel precision compare with heighborhood in x
    }

    if (pi.y > 0 && pi.y < res.rows - 1) {
        p.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value, res.at<float>(pi.y + 1, pi.x)); //get most hiest score in pixel precision compare with heighborhood in y
    }
    //std::cout << "detect-4" << std::endl;
    p.x -= (res.cols) / 2; //
    p.y -= (res.rows) / 2;

    return p; //return detected center position
}

// train tracker with a single image
void KCFTracker::train(cv::Mat x, float train_interp_factor)
{
    using namespace FFTTools;

    cv::Mat k = gaussianCorrelation(x, x); //k_xx
    cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda)); //alpha = prob/(k_xx+lambda)

    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor)*x; //update template with current image
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor)*alphaf; //update alpha
}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0));
    // HOG features
    if (_hogfeatures) {
        cv::Mat caux;
        cv::Mat x1aux;
        cv::Mat x2aux;
        for (int i = 0; i < size_patch[2]; i++) {
            x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
            x1aux = x1aux.reshape(1, size_patch[0]);
            x2aux = x2.row(i).reshape(1, size_patch[0]);
            cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
            caux = fftd(caux, true);
            rearrange(caux);
            caux.convertTo(caux, CV_32F);
            c = c + real(caux);
        }
    }
    // Gray features
    else {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true); //element-wise multiplication of two Fourier spectrums.
        c = fftd(c, true);
        rearrange(c);
        c = real(c);
    }
    cv::Mat d;
    cv::max(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) - 2. * c) / (size_patch[0] * size_patch[1] * size_patch[2]), 0, d); //enumerator of index of gauss kernel function

    cv::Mat k;
    cv::exp((-d / (sigma * sigma)), k); //gauss kernel
    return k;
}

// Create Gaussian Peak. Function called only in the first frame.
cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
{
    cv::Mat_<float> res(sizey, sizex);

    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;

    float output_sigma = std::sqrt((float)sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float)(ih * ih + jh * jh));
        }
    return FFTTools::fftd(res);
}

// Obtain sub-window from image, with replication-padding and extract features
cv::Mat KCFTracker::getFeatures(const cv::Mat& image, bool inithann, float scale_adjust)
{
    cv::Rect extracted_roi;

    float cx = _roi.x + _roi.width / 2;
    float cy = _roi.y + _roi.height / 2;
    
    if (inithann) {
        int padded_w = _roi.width * padding;
        int padded_h = _roi.height * padding;

        if (template_size > 1) {  // Fit largest dimension to the given template size
            if (padded_w >= padded_h)  //fit to width
                _scale = (float)padded_w / (float)template_size;
            else
                _scale = (float)padded_h / (float)template_size;

            _tmpl_sz.width = (int)((float)padded_w / _scale); //pad with 0
            _tmpl_sz.height = (int)((float)padded_h / _scale); //pad with 0
        }
        else {  //No template size given, use ROI size
            _tmpl_sz.width = padded_w;
            _tmpl_sz.height = padded_h;
            _scale = 1.0;
            /*if (sqrt(padded_w * padded_h) >= 100) {   //Normal size
                _tmpl_sz.width = padded_w;
                _tmpl_sz.height = padded_h;
                _scale = 1.0;
            }
            else {   //ROI is too big, track at half size
                _tmpl_sz.width = padded_w / 2;
                _tmpl_sz.height = padded_h / 2;
                _scale = 2.0;
            }*/
        }

        if (_hogfeatures) {
            // Round to cell size and also make it even
            _tmpl_sz.width = (((int)(_tmpl_sz.width / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
            _tmpl_sz.height = (((int)(_tmpl_sz.height / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
        }
        else {  //Make number of pixels even (helps with some logic involving half-dimensions)
            _tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
            _tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
        }
    }
    
    extracted_roi.width = scale_adjust * _scale * (float)_tmpl_sz.width;
    extracted_roi.height = scale_adjust * _scale * (float)_tmpl_sz.height;
    
    // center roi with new size
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;
    
    cv::Mat FeaturesMap;
    cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);
    
    if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {
        cv::resize(z, z, _tmpl_sz);
    }
    //for gray image
    FeaturesMap = RectTools::getGrayImage(z); //0 ~ 1
    FeaturesMap -= (float)0.5; // In Paper; -0.5~0.5
    size_patch[0] = z.rows;
    size_patch[1] = z.cols;
    size_patch[2] = 1;
    
    if (inithann) {
        createHanningMats();
    }
    FeaturesMap = hann.mul(FeaturesMap);
    //std::cout << "get_features-9" << std::endl;
    return FeaturesMap;
}


// Initialize Hanning window. Function called only in the first frame.
void KCFTracker::createHanningMats()
{
    cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1], 1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1, size_patch[0]), CV_32F, cv::Scalar(0));

    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float >(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float >(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d = hann2t * hann1t;
    // HOG features
    if (_hogfeatures) {
        cv::Mat hann1d = hann2d.reshape(1, 1); // Procedure do deal with cv::Mat multichannel bug

        hann = cv::Mat(cv::Size(size_patch[0] * size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
        for (int i = 0; i < size_patch[2]; i++) {
            for (int j = 0; j < size_patch[0] * size_patch[1]; j++) {
                hann.at<float>(i, j) = hann1d.at<float>(0, j);
            }
        }
    }
    // Gray features
    else {
        hann = hann2d;
    }
}

// Calculate sub-pixel peak for one dimension
float KCFTracker::subPixelPeak(float left, float center, float right)
{
    float divisor = 2 * center - right - left;

    if (divisor == 0)
        return 0;

    return 0.5 * (right - left) / divisor;
}
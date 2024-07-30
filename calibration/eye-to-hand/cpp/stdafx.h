// stdafx.h : 標準のシステム インクルード ファイルのインクルード ファイル、または
// 参照回数が多く、かつあまり変更されない、プロジェクト専用のインクルード ファイル
// を記述します。
//

#pragma once

//utility
#include <iostream>
#include <string>
#include <chrono>
#include <atomic>
#include <thread>
#include <vector>
#include <array>
#include <queue>
#include <mutex>
#include <chrono>
#include <fstream>
#include <sstream> // For stringstream
#include <ctime>
#include <direct.h>
#include <sys/stat.h>
#include <cmath>
#include <filesystem>

//Matrix
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

//boost : about queue
#include <boost/asio.hpp>
#include <boost/lockfree/spsc_queue.hpp>

//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/optflow/rlofflow.hpp>
#include <opencv2/tracking.hpp>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/tracking/tracking_legacy.hpp>

#include <iso646.h> 

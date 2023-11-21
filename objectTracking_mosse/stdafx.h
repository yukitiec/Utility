#pragma once


#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/ocl.hpp"
#include <opencv2/tracking/tracking_legacy.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <atomic>
#include <queue>
#include <thread>
#include <vector>
#include <mutex>
#include <array>
#include <ctime>
#include <direct.h>
#include <sys/stat.h>
#include <algorithm>
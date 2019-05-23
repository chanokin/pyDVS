#ifndef DVS_EMU_HPP
#define DVS_EMU_HPP

#include <iostream>
#include <stdint>
#include "opencv2/opencv.hpp"

// using namespace cv;
using namespace std;

class PyDVS{

public:
    PyDVS():_relaxRate(1.0f), _adaptUp(1.0f), _adaptDown(1.0f),
        _w(0), _h(0), _open(false);
    PyDVS(const size_t w, const size_t h, const size_t fps=0);
    ~PyDVS();
    bool init(const int cam_id=0);
    bool init(const String& filename);


private:
    cv::VideoCapture _cam;
    cv::Mat _in;
    cv::Mat _ref;
    cv::Mat _diff;
    cv::Mat _thr;

    float _relaxRate;
    float _adaptUp;
    float _adaptDown;

    size_t _w, _h;
    bool _open;
    void _get_size();
    void _get_fps();

};

#endif //DVS_EMU_HPP
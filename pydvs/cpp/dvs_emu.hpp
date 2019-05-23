#ifndef DVS_EMU_HPP
#define DVS_EMU_HPP

#include <iostream>
#include <stdint>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
// #include <opencv2/highgui.hpp>

// using namespace cv;
using namespace std;

class PyDVS{

public:
    PyDVS():_relaxRate(1.0f), _adaptUp(1.0f), _adaptDown(1.0f),
        _w(0), _h(0), _open(false);
    PyDVS(const size_t w, const size_t h, const size_t fps=0);
    ~PyDVS();
    bool init(const int cam_id=0, const float relaxRate=1.0f, 
        const float adaptUp=1.0f, const float adaptDown=1.0f);
    bool init(const String& filename, const float relaxRate=1.0f, 
        const float adaptUp=1.0f, const float adaptDown=1.0f);

    void inline setFPS(const size_t fps){_fps = fps;}
    void inline setWidth(const size_t w){_w = w;}
    void inline setHeight(const size_t h){_h = h;}
    void inline setRelaxRate(const float r){_relaxRate = r;}
    void inline setAdaptUp(const float u){_adaptUp = u;}
    void inline setAdaptDown(const float d){_adaptDown = d;}
    
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
    bool _is_vid;

    void _get_size();
    void _get_fps();
    bool _set_size();
    bool _set_fps();
    inline void _set_adapt(const float relaxRate, const float adaptUp, 
                            const float adaptDown){
        _relaxRate = relaxRate;
        _adaptUp = adaptUp;
        _adaptDown = adaptDown;
    }
    inline void _init_matrices(){
        // CV_32F 32-bit floating point numbers
        _ref = Mat::zeros(_h, _w, CV_32F);
        _diff = Mat::zeros(_h, _w, CV_32F);
        _thr = Mat::zeros(_h, _w, CV_32F);
    }
};

#endif //DVS_EMU_HPP
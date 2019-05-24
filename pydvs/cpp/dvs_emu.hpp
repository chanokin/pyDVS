#ifndef DVS_EMU_HPP
#define DVS_EMU_HPP

#include <iostream>
#include <stdint.h>
#include <string>
#include <opencv2/opencv.hpp>
// #include "opencv2/core.hpp"
// #include "opencv/videoio.hpp"
// #include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv_modules.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "dvs_op.hpp"

// using namespace cv;
using namespace std;

class PyDVS{

public:
    PyDVS():_relaxRate(1.0f), _adaptUp(1.0f), _adaptDown(1.0f),
        _w(0), _h(0), _fps(0), _open(false){}
    PyDVS(const size_t w, const size_t h, const size_t fps=0);
    ~PyDVS();
    bool init(const int cam_id=0, const float thr=12.75f,
        const float relaxRate=1.0f, const float adaptUp=1.0f, 
        const float adaptDown=1.0f);
    bool init(const string& filename, const float thr=12.75f,
        const float relaxRate=1.0f, const float adaptUp=1.0f, 
        const float adaptDown=1.0f);
    bool init(const char* filename, const float thr=12.75f,
        const float relaxRate=1.0f, const float adaptUp=1.0f, 
        const float adaptDown=1.0f);

    inline void setFPS(const size_t fps){_fps = fps;}
    inline void setWidth(const size_t w){_w = w;}
    inline void setHeight(const size_t h){_h = h;}
    inline void setRelaxRate(const float r){_relaxRate = r;}
    inline void setAdaptUp(const float u){_adaptUp = u;}
    inline void setAdaptDown(const float d){_adaptDown = d;}

    inline size_t getFPS(){return _fps;}
    inline size_t getWidth(){return _w;}
    inline size_t getHeight(){return _h;}
    inline float getRelaxRate(){return _relaxRate;}
    inline float getAdaptUp(){return _adaptUp;}
    inline float getAdaptDown(){return _adaptDown;}
    inline cv::Mat& getInput(){return _in;}
    inline cv::Mat& getReference(){return _ref;}
    inline cv::Mat& getDifference(){return _diff;}
    inline cv::Mat& getEvents(){return _events;}
    inline cv::Mat& getThreshold(){return _thr;}

    bool update();
    inline void setAdapt(const float relaxRate, const float adaptUp, 
                            const float adaptDown){
        _relaxRate = relaxRate;
        _adaptUp = adaptUp;
        _adaptDown = adaptDown;
    }

private:
    cv::VideoCapture _cap;
    cv::Mat _in;
    cv::Mat _frame;
    cv::Mat _ref;
    cv::Mat _diff;
    cv::Mat _absDiff;
    cv::Mat _thr;
    cv::Mat _events;
    cv::Mat _gray;

    float _relaxRate;
    float _adaptUp;
    float _adaptDown;

    size_t _w, _h, _fps;
    bool _open;
    bool _is_vid;
    DVSOperator _dvsOp;

    void _get_size();
    void _get_fps();
    bool _set_size();
    bool _set_fps();
    inline void _initMatrices(const float thr_init){
        // CV_32F 32-bit floating point numbers
        _gray  = cv::Mat::zeros(_h, _w, CV_8UC1);
        _in  = cv::Mat::zeros(_h, _w, CV_32F);
        _ref = cv::Mat::zeros(_h, _w, CV_32F);
        _diff = cv::Mat::zeros(_h, _w, CV_32F);
        _absDiff = cv::Mat::zeros(_h, _w, CV_32F);
        _events = cv::Mat::zeros(_h, _w, CV_32F);
        // try random init, might help, or not hurt
        // cv::RNG rng(1);
        // rng.fill(_diff, cv::RNG::UNIFORM, 64.0f, 192.0f, true);
        _thr = thr_init * cv::Mat::ones(_h, _w, CV_32F);
        _dvsOp.init(_in, _diff, _ref, _thr, _events,
                    _relaxRate, _adaptUp, _adaptDown);

    }
};




#endif //DVS_EMU_HPP
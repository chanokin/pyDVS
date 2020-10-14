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

#include "nvs_op.hpp"

// using namespace cv;
using namespace std;

class PyDVS{

public:
    PyDVS():_relaxRate(1.0f), _adaptUp(1.0f), _adaptDown(1.0f),
        _baseThresh(12.0f), _leakProb(0.8f),
         _w(0), _h(0), _fps(0), _open(false){}
    PyDVS(const size_t w, const size_t h, const size_t fps=0);
    ~PyDVS();
    bool init(const int cam_id=0, const float thr=12.75f,
        const float relaxRate=1.0f, const float adaptUp=1.0f, 
        const float adaptDown=1.0f, const float leakProb=0.8f);
    bool init(const string& filename, const float thr=12.75f,
        const float relaxRate=1.0f, const float adaptUp=1.0f, 
        const float adaptDown=1.0f, const float leakProb=0.8f);
    bool init(const char* filename, const float thr=12.75f,
        const float relaxRate=1.0f, const float adaptUp=1.0f, 
        const float adaptDown=1.0f, const float leakProb=0.8f);

    inline void setFPS(const size_t fps){_fps = fps;}
    inline void setWidth(const size_t w){_w = w;}
    inline void setHeight(const size_t h){_h = h;}
    inline void setRelaxRate(const float r){_relaxRate = r;}
    inline void setAdaptUp(const float u){_adaptUp = u;}
    inline void setAdaptDown(const float d){_adaptDown = d;}
    inline void setLeakProb(const float p){_leakProb = p;}

    inline size_t getFPS() const {return _fps;}
    inline size_t getWidth() const {return _w;}
    inline size_t getHeight()const {return _h;}
    inline float getRelaxRate()const {return _relaxRate;}
    inline float getAdaptUp()const {return _adaptUp;}
    inline float getAdaptDown()const {return _adaptDown;}
    inline float getLeakProb() const {return _leakProb;}
    inline const cv::Mat& getInput()const {return _in;}
    inline const cv::Mat& getReference()const {return _ref;}
    inline const cv::Mat& getDifference()const {return _diff;}
    inline const cv::Mat& getEvents()const {return _events;}
    inline const cv::Mat& getThreshold()const {return _thr;}

    bool update();
    inline void setAdapt(const float relaxRate, const float adaptUp, 
        const float adaptDown, const float threshold, const float leakProb){
        _relaxRate = relaxRate;
        _adaptUp = adaptUp;
        _adaptDown = adaptDown;
        _baseThresh = threshold;
        _leakProb = leakProb;
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
    float _baseThresh;
    float _leakProb;

    size_t _w, _h, _fps;
    bool _open;
    bool _is_vid;
    NVSOperator _nvsOp;


    void _get_size();
    void _get_fps();
    bool _set_size();
    bool _set_fps();
    inline void _initMatrices(const float thr_init=-1000.0f){
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
        cout << _relaxRate << "," << _adaptUp << "," << _adaptDown << endl;
        if(thr_init > _baseThresh){
            _baseThresh = thr_init;
        }
        _thr = cv::Mat(_h, _w, CV_32F, _baseThresh);
        _nvsOp.init(&_in, &_diff, &_ref, &_thr, &_events,
            _relaxRate, _adaptUp, _adaptDown, _leakProb, _baseThresh);

    }
};




#endif //DVS_EMU_HPP
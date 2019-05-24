#ifndef DVS_OP_HPP
#define DVS_OP_HPP

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

class DVSOperator: public cv::ParallelLoopBody
{
public:
    DVSOperator():src(nullptr), diff(nullptr), ref(nullptr), thr(nullptr),
        ev(nullptr), relax(1.0f), up(1.0f), down(1.0f){}
    DVSOperator(
        cv::Mat* _src, cv::Mat* _diff, 
        cv::Mat* _ref, cv::Mat* _thr, cv::Mat* _ev,
        float _relax, float _up, float _down)
        : src(_src), diff(_diff), ref(_ref), thr(_thr),
        ev(_ev), relax(_relax), up(_up), down(_down){}
    inline void init(cv::Mat* _src, cv::Mat* _diff, 
                    cv::Mat* _ref, cv::Mat* _thr, cv::Mat* _ev,
                    float _relax, float _up, float _down){
        src = _src;
        diff = _diff;
        ref = _ref; 
        thr = _thr;
        ev = _ev; 
        relax = _relax; 
        up = _up;
        down = _down;
    }

    inline void init(cv::Mat& _src, cv::Mat& _diff, 
                    cv::Mat& _ref, cv::Mat& _thr, cv::Mat& _ev,
                    float _relax, float _up, float _down){
        src = &_src;
        diff = &_diff;
        ref = &_ref; 
        thr = &_thr;
        ev = &_ev; 
        relax = _relax; 
        up = _up;
        down = _down;
    }


    void operator()(const cv::Range& range) const;

private:
    cv::Mat* src;
    cv::Mat* diff;
    cv::Mat* ref; 
    cv::Mat* thr; 
    cv::Mat* ev;
    float relax;
    float up;
    float down;

};

#endif // DVS_OP_HPP
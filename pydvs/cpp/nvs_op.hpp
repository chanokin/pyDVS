#ifndef DVS_OP_HPP
#define DVS_OP_HPP

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

class NVSOperator: public cv::ParallelLoopBody
{
public:
    NVSOperator():src(nullptr), diff(nullptr), ref(nullptr), thr(nullptr),
        ev(nullptr), relax(1.0f), up(1.0f), down(1.0f), prob(0.8f), base_thr(10.0f){}
    NVSOperator(
        cv::Mat* _src, cv::Mat* _diff, 
        cv::Mat* _ref, cv::Mat* _thr, cv::Mat* _ev,
        float _relax, float _up, float _down, float _prob, float _base_thr)
        : src(_src), diff(_diff), ref(_ref), thr(_thr),
        ev(_ev), relax(_relax), up(_up), down(_down), 
        prob(_prob), base_thr(_base_thr){}
    inline void init(cv::Mat* _src, cv::Mat* _diff, 
                    cv::Mat* _ref, cv::Mat* _thr, cv::Mat* _ev,
                    float _relax, float _up, float _down, float _prob,
                    float _base_thr){
        cout << "In DVS_OP init function " << endl;
        src = _src;
        diff = _diff;
        ref = _ref; 
        thr = _thr;
        ev = _ev; 
        relax = _relax; 
        up = _up;
        down = _down;
        prob = _prob;
        base_thr = _base_thr;
        
        cout << "relax " << relax << " up " << up \
             << " down " << down << " prob " << prob << endl;
    }

    // inline void init(cv::Mat _src, cv::Mat _diff, 
    //                 cv::Mat _ref, cv::Mat _thr, cv::Mat _ev,
    //                 float _relax, float _up, float _down){
    //     src = &_src;
    //     diff = &_diff;
    //     ref = &_ref; 
    //     thr = &_thr;
    //     ev = &_ev; 
    //     relax = _relax; 
    //     up = _up;
    //     down = _down;
    // }


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
    float prob;
    float base_thr;
};

#endif // DVS_OP_HPP
#include "dvs_emu.hpp"

PyDVS::PyDVS(const size_t w, const size_t h, const size_t fps){
    _w = w;
    _h = h;
    _fps = fps;
}

PyDVS::~PyDVS(){
    if(_open){
        _cap.release();
    }
}
bool PyDVS::init(const int cam_id, const float thr, const float relaxRate, 
        const float adaptUp, const float adaptDown){
    _cap = cv::VideoCapture(cam_id);
    _open = _cap.isOpened();
    if(!_open){
        cerr << "Error. Cannot open video feed!" << endl;
        return false;
    }
    
    bool success = true;
    _is_vid = false;
    if(_w == 0 || _h == 0 || _fps == 0){
        _get_size();
        _get_fps();
    } else {
        success &= _set_size();
        success &= _set_fps();
    }
    setAdapt(relaxRate, adaptUp, adaptDown, thr);
    _initMatrices(thr);
    return success;
}

bool PyDVS::init(const string& filename, const float thr, const float relaxRate, 
        const float adaptUp, const float adaptDown){
    _cap = cv::VideoCapture(filename);
    _open = _cap.isOpened();
    if(!_open){
        cerr << "Init. Cannot open video feed!" << endl;
        return false;
    }

    _is_vid = true;
    _get_size();
    _get_fps();
    // set output format as 32-bit floating point with a single channel
    _cap.set(CV_CAP_PROP_FORMAT, CV_32FC1);
    setAdapt(relaxRate, adaptUp, adaptDown, thr);
    _initMatrices(thr);

    return true;
}

bool PyDVS::init(const char* filename, const float thr, const float relaxRate, 
        const float adaptUp, const float adaptDown){
    _cap = cv::VideoCapture(filename);
    _open = _cap.isOpened();
    if(!_open){
        cerr << "Init. Cannot open video feed!" << endl;
        return false;
    }

    _is_vid = true;
    _get_size();
    _get_fps();
    // set output format as 32-bit floating point with a single channel
    _cap.set(CV_CAP_PROP_FORMAT, CV_32FC1);
    setAdapt(relaxRate, adaptUp, adaptDown, thr);
    _initMatrices(thr);

    return true;
}


void PyDVS::_get_size(){
    if(_open){
        _w = _cap.get(CV_CAP_PROP_FRAME_WIDTH);
        _h = _cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    }
}

bool PyDVS::_set_size(){
 
    if(_open){
        size_t w = _cap.get(CV_CAP_PROP_FRAME_WIDTH);
        size_t h = _cap.get(CV_CAP_PROP_FRAME_HEIGHT);
        bool success = true;
        success &= _cap.set(CV_CAP_PROP_FRAME_WIDTH, _w);
        success &= _cap.set(CV_CAP_PROP_FRAME_HEIGHT, _h);

        if (!success){
            _cap.set(CV_CAP_PROP_FRAME_WIDTH, w);
            _cap.set(CV_CAP_PROP_FRAME_HEIGHT, h);
            _w = w; 
            _h = h;
        }
        return success;
    }
    return false;
}

void PyDVS::_get_fps(){
    if(_open){
        _fps = _cap.get(CV_CAP_PROP_FPS);
    }
}

bool PyDVS::_set_fps(){
    if(_open){
        size_t fps = _cap.get(CV_CAP_PROP_FPS);
        bool success = true;
        success &= _cap.set(CV_CAP_PROP_FPS, _fps);
        if(!success){
            _cap.set(CV_CAP_PROP_FPS, fps);
            _fps = fps;
        }
        return success;
    }
    return false;
}

bool PyDVS::update(){
    _cap >> _frame;
    if (_frame.empty()){
        return false;
    }
    cv::cvtColor(_frame, _gray, CV_BGR2GRAY);
    _gray.convertTo(_in, CV_32F);
    cv::parallel_for_(cv::Range(0, _gray.rows), _dvsOp);
    
    // _diff.forEach<float>(Operator());
    // subtract(_in, _ref, _diff);
    // _absDiff = cv::abs(_diff);
    // cv::threshold(_absDiff, _absDiff, _thr);
    // // _events = _diff * _absDiff;
    // _ref = (_relaxRate * _ref) + _events;
    return true;

}
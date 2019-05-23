#include "dvs_emu.hpp"

PyDVS::PyDVS(const size_t w, const size_t h, const size_t fps){
    _w = w;
    _h = h;
    _fps = fps;
}

PyDVS::~PyDVS(){
    if(_open){
        _cam.release();
    }
}

bool PyDVS::init(const int cam_id, const float relaxRate, 
        const float adaptUp, const float adaptDown){
    _cam = VideoCapture(cam_id);
    _open = _cam.isOpened();
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

    _init_matrices();
    _set_adapt(relaxRate, adaptUp, adaptDown);
    return success;
}

bool PyDVS::init(const String& filename, const float relaxRate, 
        const float adaptUp, const float adaptDown){
    _cam = VideoCapture(filename);
    _open = _cam.isOpened();
    if(!_open){
        cerr << "Error. Cannot open video feed!" << endl;
        return false;
    }

    _is_vid = true;
    _get_size();
    _get_fps();
    _init_matrices();
    _set_adapt(relaxRate, adaptUp, adaptDown);

    return true;
}


void PyDVS::_get_size(){
    if(_open){
        _w = _cam.get(cv::CV_CAP_PROP_FRAME_WIDTH);
        _h = _cam.get(cv::CV_CAP_PROP_FRAME_HEIGHT);
    }
}

bool PyDVS::_set_size(){
 
    if(_open){
        size_t w = _cam.get(cv::CV_CAP_PROP_FRAME_WIDTH);
        size_t h = _cam.get(cv::CV_CAP_PROP_FRAME_HEIGHT);
        bool success = true;
        success &= _cam.set(cv::CV_CAP_PROP_FRAME_WIDTH, _w);
        success &= _cam.set(cv::CV_CAP_PROP_FRAME_HEIGHT, _h);

        if (!success){
            _cam.set(cv::CV_CAP_PROP_FRAME_WIDTH, w);
            _cam.set(cv::CV_CAP_PROP_FRAME_HEIGHT, h);
            _w = w; 
            _h = h;
        }
        return success;
    }
}

void PyDVS::_get_fps(){
    if(_open){
        _fps = _cam.get(cv::CV_CAP_PROP_FPS);
    }
}

bool PyDVS::_set_fps(){
    if(_open){
        size_t fps = _cam.get(cv::CV_CAP_PROP_FPS);
        bool success = true;
        success &= _cam.set(cv::CV_CAP_PROP_FPS, _fps);
        if(!success){
            _cam.set(cv::CV_CAP_PROP_FPS, fps);
            _fps = fps;
        }
        return success;
    }
}
#include "dvs_emu.hpp"

PyDVS::PyDVS(){

}

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

bool PyDVS::init(const int cam_id){
    _cam = VideoCapture(cam_id);
    _open = _cam.isOpened();
    if(!_open){
        return false;
    }

    if(_w == 0 || _h == 0 || _fps == 0){
        _get_size();
        _get_fps();
    } else {

    }
    return true;
}

bool PyDVS::init(const String& filename){
    _cam = VideoCapture(filename);
    _open = _cam.isOpened();
    if(!_open){
        return false;
    }

    _get_size();
    _get_fps();

    return true;
}


PyDVS::_get_size(){
    if(_open){
        _w = _cam.get(cv::CV_CAP_PROP_FRAME_WIDTH);
        _h = _cam.get(cv::CV_CAP_PROP_FRAME_HEIGHT);
    }
}

PyDVS::_get_fps(){
    if(_open){
        _fps = _cam.get(cv::CV_CAP_PROP_FPS);
    }
}
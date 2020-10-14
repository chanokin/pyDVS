#include "nvs_emu.hpp"
#include <opencv2/opencv.hpp>
#define B 0
#define G 1
#define R 2

void diffToBGR(cv::Mat& bgr, const cv::Mat& gray, const cv::Mat& diff, const float thr){
    for(int row = 0; row < gray.rows; row++){
        for(int col = 0; col < gray.cols; col++){
            cv::Vec3f color(0.0f, 0.0f, 0.0f);
            float val = diff.at<float>(row, col);
            if(val > thr){
                color[G] = 1.0;
            } else if(val < -thr){
                color[R] = 1.0;
            } else {
                val = gray.at<float>(row, col)/255.0f;
                color[R] = val;
                color[G] = val;
                color[B] = val;
            }
            bgr.at<cv::Vec3f>(row, col) = color;
        }
    }

}
void copyToFullFrame(cv::Mat& ff, const cv::Mat& gray, const cv::Mat& ref, 
    cv::Mat diff, cv::Mat bgr){

}

int main(int argc, const char* argv[]){
    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : " << CV_MAJOR_VERSION << endl;
    cout << "Minor version : " << CV_MINOR_VERSION << endl;
    cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;    
    
    PyDVS dvs;

    // compute this as exp(-1/frames)?
    float up = 1.5f;
    float down = 0.99f;
    float rate = 0.999f;
    float thr = 10.0f;
    float prob = 0.8f;
/*
    bool ok = dvs.init("./SampleVideo_360x240_30mb.mp4", thr, rate, up, down);
/*/
    dvs.setWidth(640);
    dvs.setHeight(480);
    bool ok = dvs.init(0, thr, rate, up, down, prob);
//*///

    if(!ok){
        cerr << "Main: unable to open video source" << endl;
        return 1;
    }

    dvs.setAdapt(rate, up, down, thr, prob);
    cout << "Relax rate = " << dvs.getRelaxRate() << endl;
    cout << "Adapt up = " << dvs.getAdaptUp() << endl;
    cout << "Adapt down = " << dvs.getAdaptDown() << endl;


    // cout << dvs.getThreshold().row(dvs.getHeight()/2);
    size_t w = dvs.getWidth();
    size_t h = dvs.getHeight();
    cv::Mat frame(h, w, CV_32FC3);
    cv::Mat full(h, w*4, CV_32FC3);
    cv::Mat fullGray = full(cv::Rect(0,0,w,h));
    cv::Mat fullRef = full(cv::Rect(w, 0,w,h));
    cv::Mat fullDiff = full(cv::Rect(2*w,0,w,h));
    cv::Mat fullOut = full(cv::Rect(3*w,0,w,h));
    cv::Mat tmp(h, w, CV_32FC3);
    // matSrc.copyTo(matRoi);
    // matRoi = matDst(Rect(matSrc.cols,0,matSrc.cols,matSrc.rows));
    // matGray.copyTo(matRoi);

    int count = 0;
    while(ok){
        ok = dvs.update();
        if(ok){

            diffToBGR(frame, dvs.getInput(), dvs.getDifference(), 0.0);
            cvtColor((dvs.getInput()*(1.0f/255.0f)),
                     tmp, cv::COLOR_GRAY2BGR);
            tmp.copyTo(fullGray);

            cvtColor((dvs.getReference()*(1.0f/255.0f)),
                     tmp, cv::COLOR_GRAY2BGR);
            tmp.copyTo(fullRef);

            cvtColor((dvs.getDifference()*(1.0f/255.0f)),
                     tmp, cv::COLOR_GRAY2BGR);
            tmp.copyTo(fullDiff);

            frame.copyTo(fullOut);

            // cv::imshow("Frame", dvs.getInput()*(1.0f/255.0f));
            // cv::imshow("Frame", dvs.getReference()*(1.0f/255.0f));
            // cv::imshow("Frame", dvs.getDifference()*(1.0f/255.0f));
            // cv::imshow("Frame", frame);
            cv::imshow("Frame", full);
            char c=(char)cv::waitKey(1);
            if(c==27 || c == 'q' || c == 'Q'){
                break;
            }
        }

        count++;
        // if(count == 3){
        //     cout << dvs.getThreshold().row(dvs.getHeight()/2);
        //     break;
        // }
    }
    
    return 0;
}

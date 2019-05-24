#include "dvs_emu.hpp"
#include "opencv2/highgui/highgui.hpp"

int main(int argc, const char* argv[]){
    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : " << CV_MAJOR_VERSION << endl;
    cout << "Minor version : " << CV_MINOR_VERSION << endl;
    cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;    
    
    PyDVS dvs;
    float up = 1.5f;
    float down = 0.9f;
    float rate = 0.9999f;

    bool ok = dvs.init("./SampleVideo_360x240_30mb.mp4");

    if(!ok){
        cerr << "Main: unable to open video source" << endl;
        return 1;
    }

    dvs.setAdapt(rate, up, down);
    cout << "Relax rate = " << dvs.getRelaxRate() << endl;

    cv::Mat frame(dvs.getHeight(), dvs.getWidth(), CV_32FC3);
    int count = 0;
    while(ok){
        ok = dvs.update();
        if(ok){
            for(size_t row = 0; row < dvs.getHeight(); row++){
                for(size_t col = 0; col < dvs.getWidth(); col++){
                    
                }
            }
            // cv::imshow("Frame", dvs.getInput()*(1.0f/255.0f));
            cv::imshow("Frame", dvs.getDifference()*(1.0f/255.0f));
            char c=(char)cv::waitKey(100);
            if(c==27 || c == 'q' || c == 'Q'){
                break;
            }
        }
        count++;
        // if(count == 3){
        //     break;
        // }
    }
    
    return 0;
}

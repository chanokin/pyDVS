#include "dvs_op.hpp"


void DVSOperator::operator()(const cv::Range& range) const{
    for (int32_t row(range.start); row < range.end; ++row) {
        float const* it_src(src->ptr<float>(row));
        float* it_diff(diff->ptr<float>(row));
        float* it_ref(ref->ptr<float>(row));
        float* it_thr(thr->ptr<float>(row));
        float* it_ev(ev->ptr<float>(row));

        for (int32_t col(0); col < src->cols; ++col) {
            (*it_diff) = (*it_src) - (*it_ref);
            bool test = (((*it_diff) < -(*it_thr)) || ((*it_diff) > (*it_thr)));
            (*it_diff) = (*it_diff) * ((float)test);
            (*it_ref) = (relax * (*it_ref)) + (*it_diff);
            if(test){
                (*it_thr) = (*it_thr) * up;
            }else{
                (*it_thr) = (*it_thr) * down;
            }


            
            //advance pointers
            ++it_src; 
            ++it_diff; 
            ++it_ref;
            ++it_thr; 
            ++it_ev;
        }
    }
}

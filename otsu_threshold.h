#ifndef OTSU_THRESHOLD_H
#define OTSU_THRESHOLD_H

#include "hls_math.h"
#include "hls_video.h"

namespace hls {

template<int SRC_T, int DST_T,int ROW, int COL>
void Threshold(
    Mat<ROW, COL, SRC_T>    &_src,
    Mat<ROW, COL, DST_T>    &_dst,
    ap_uint<8> &threshold)
{
    const int NUM_STATES=4;
    Window<1,NUM_STATES,ap_uint<8> > addr_win;

    ap_uint<BitWidth<ROW*COL>::Value> hist_out[256];
    Window<1,NUM_STATES,ap_uint<BitWidth<ROW*COL>::Value> > hist_win;

    ap_uint<BitWidth<ROW*COL>::Value> hist;
    ap_uint<8> addr;
    ap_uint<8> addr_last;
    ap_uint<BitWidth<ROW*COL>::Value> hist_last;
    ap_uint<8> addr_flag;
    ap_uint<BitWidth<ROW*COL>::Value> hist_flag;
    ap_uint<8> addr_w;
    ap_uint<BitWidth<ROW*COL>::Value> hist_w;

    ap_uint<BitWidth<ROW*COL>::Value> tmp=0;

    float pixel_probability[256];
    for(int i=0;i<NUM_STATES;i++) {
    #pragma HLS UNROLL
        addr_win(0,i)=i;
        hist_win(0,i)=0;
    }

    for(int i=0;i<256;i++) {
        hist_out[i]=0;
        pixel_probability[i] = 0.0f;
    }

    int cols=_src.cols;
    int rows=_src.rows;
    assert(rows<=ROW);
    assert(cols<=COL);
 loop_height: for(int i=0;i<rows;i++)
    {
    loop_width: for(int j=0;j<cols;j++)
        {
//#pragma HLS PIPELINE
//#pragma HLS LOOP_FLATTEN OFF
//#pragma HLS DEPENDENCE array inter false
            ap_uint<4> flag=NUM_STATES;
            HLS_TNAME(SRC_T) tempsrc=0;
            HLS_TNAME(DST_T) tempdst=0;
            _src.data_stream[0].read(tempsrc);
            tempdst=tempsrc > threshold ? 255 : 0;
            _dst.data_stream[0]<<tempdst;

            for (int m=0; m<NUM_STATES; m++) {
                if (tempsrc==addr_win(0,m)) {
                    flag = m;
                    break;
                }
            }

            latency_region:{
            #pragma HLS latency min=0 max=1
            addr_last = addr_win(0,NUM_STATES-1);
            hist_last = hist_win(0,NUM_STATES-1)+1;
            for (int m=NUM_STATES-1; m>0; m--) {
                addr = addr_win(0,m-1);
                hist = hist_win(0,m-1);
                if (m==NUM_STATES/2) {
                    addr_w = addr;
                    if (m==flag+1) {
                        hist_w = hist+1;
                    } else {
                        hist_w = hist;
                    }
                }
                if (m==flag+1) {
                    addr_flag = addr;
                    hist_flag = hist+1;
                    addr_win(0,m) = addr_flag;
                    hist_win(0,m) = hist_flag;
                } else {
                    addr_win(0,m) = addr;
                    hist_win(0,m) = hist;
                }
            }
            if (flag==NUM_STATES) {
                hist_win(0,0) = hist_out[tempsrc]+1;
                addr_win(0,0) = tempsrc;
            } else if (flag==NUM_STATES-1) {
                addr_win(0,0) = addr_last;
                hist_win(0,0) = hist_last;
            } else if (flag>=NUM_STATES/2) {
                addr_win(0,0) = addr_flag;
                hist_win(0,0) = hist_flag;
            } else {
                addr_win(0,0) = addr_w;
                hist_win(0,0) = hist_w;
            }
            hist_out[addr_w] = hist_w;
            }
        }
    }

    for (int m=0; m<NUM_STATES/2; m++) {
   // #pragma HLS PIPELINE
        hist_out[addr_win(0,m)]=hist_win(0,m);
    }

    int         front_pixel_count;          //前景图像像素个数
    int         back_pixel_count;           //背景图像像素个数
    float       front_pixel_probability;    //前景图像像素出现的概率
    float       back_pixel_probability;     //背景图像像素出现的概率
    int         front_gray_count;           //前景灰度总和
    int         back_gray_count;            //背景灰度总和
    int         total_gray;                 //整幅图像灰度总和
    float       front_gray_average;         //前景平均灰度
    float       back_gray_average;          //背景平均灰度
    float       total_gray_average;         //整幅图像的平均灰度
    int         threshold_tmp;              //临时阈值
    float       interclass_variance_tmp;    //临时类间方差
    float       interclass_variance_max;    //最大类间方差
    for(threshold_tmp = 0; threshold_tmp < 256; threshold_tmp++){

        front_pixel_count = 0;
        back_pixel_count = 0;
        front_pixel_probability = 0;
        back_pixel_probability = 0;
        front_gray_count = 0;
        back_gray_count = 0;
        front_gray_average = 0;
        back_gray_average = 0;
        total_gray_average = 0;

        for(int j = 0; j < 256; j++){
            //前景部分
            if(j <= threshold_tmp){
                //以threshold_tmp为阈值分类，计算前景图像像素出现的个数和灰度总和
                front_pixel_count += hist_out[j];
                front_gray_count += j * hist_out[j];
            }
            //背景部分
            else{
                //以threshold_tmp为阈值分类，计算背景图像像素出现的个数和灰度总和
                back_pixel_count += hist_out[j];
                back_gray_count += j * hist_out[j];
            }
        }

        //前景图像像素出现的概率
        front_pixel_probability = (float)front_pixel_count / (rows*cols);
        //背景图像像素出现的概率
        back_pixel_probability = (float)back_pixel_count / (rows*cols);
        //整幅图像灰度总和
        total_gray = front_gray_count + back_gray_count;
        //前景平均灰度
        front_gray_average = (float)front_gray_count / front_pixel_count;
        //背景平均灰度
        back_gray_average = (float)back_gray_count / back_pixel_count;
        //整幅图像平均灰度
        total_gray_average = (float)total_gray / (rows*cols);

        //计算类间方差
        interclass_variance_tmp = front_pixel_probability *
                                  (front_gray_average - total_gray_average) *
                                  (front_gray_average - total_gray_average)
                                + back_pixel_probability *
                                  (back_gray_average - total_gray_average) *
                                  (back_gray_average - total_gray_average);
        //找出最大类间方差以及对应的阈值
        if (interclass_variance_tmp > interclass_variance_max){
            interclass_variance_max = interclass_variance_tmp;
            threshold = threshold_tmp;
        }
    }
}

//otsu自适应二值化函数
static  ap_uint<8> threshold;
template<int SRC_T, int DST_T,int ROW, int COL>
void Otsu_threshold(
        Mat<ROW, COL, SRC_T>    &_src,
        Mat<ROW, COL, DST_T>    &_dst)
    {
   // #pragma HLS INLINE
        Threshold(_src, _dst, threshold);
    }
}

#define MAX_HEIGHT 800 //图像最大高度
#define MAX_WIDTH 1280 //图像最大宽度
typedef hls::stream<ap_axiu<24,1,1,1> > AXI_STREAM;
typedef hls::Mat<MAX_HEIGHT,MAX_WIDTH,HLS_8UC3> RGB_IMAGE;
typedef hls::Mat<MAX_HEIGHT,MAX_WIDTH,HLS_8UC1> GRAY_IMAGE;

void otsu_threshold(AXI_STREAM &INPUT_STREAM,
                           AXI_STREAM &OUTPUT_STREAM,
                           int rows,
                           int cols
);

#endif

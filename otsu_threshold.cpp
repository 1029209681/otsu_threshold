#include "otsu_threshold.h"

void otsu_threshold(AXI_STREAM &INPUT_STREAM,
					       AXI_STREAM &OUTPUT_STREAM,
						   int rows,
						   int cols
){

#pragma HLS INTERFACE axis port=INPUT_STREAM
#pragma HLS INTERFACE axis port=OUTPUT_STREAM
#pragma HLS INTERFACE s_axilite port=rows
#pragma HLS INTERFACE s_axilite port=cols
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS dataflow
//#pragma HLS PIPELINE

	//hls::mat 格式变量
	RGB_IMAGE img_0(rows,cols);
	GRAY_IMAGE img_1(rows,cols);
	GRAY_IMAGE img_2(rows,cols);
	RGB_IMAGE img_3(rows,cols);

	//将 AXI4 Stream 数据转换成 hls::mat 格式
	hls::AXIvideo2Mat(INPUT_STREAM,img_0);

	//将 RGB888 格式的彩色数据转换成灰度数据
	hls::CvtColor<HLS_RGB2GRAY,HLS_8UC3,HLS_8UC1>(img_0,img_1);

	//对灰度图像进行OTSU自适应二值化
	hls::Otsu_threshold<HLS_8UC1,HLS_8UC1,MAX_HEIGHT,MAX_WIDTH>(img_1,img_2);

	//将灰度数据转换成三个通道的灰度图像
	hls::CvtColor<HLS_GRAY2RGB,HLS_8UC1,HLS_8UC3>(img_2,img_3);

	//将 hls::mat 格式数据转换成 AXI4 Stream 格式
	hls::Mat2AXIvideo(img_3,OUTPUT_STREAM);
}

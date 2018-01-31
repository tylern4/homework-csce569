#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <sys/timeb.h>

using namespace std;

/* read timer in second */
double read_timer() {
  struct timeb tm;
  ftime(&tm);
  return (double)tm.time + (double)tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
  struct timeb tm;
  ftime(&tm);
  return (double)tm.time * 1000.0 + (double)tm.millitm;
}

short lpf_filter_6[3][3] = {{0, 1, 0}, {1, 2, 1}, {0, 1, 0}};

short lpf_filter_9[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};

short lpf_filter_10[3][3] = {{1, 1, 1}, {1, 2, 1}, {1, 1, 1}};

short lpf_filter_16[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

short lpf_filter_32[3][3] = {{1, 4, 1}, {4, 12, 4}, {1, 4, 1}};

short hpf_filter_1[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};

short hpf_filter_2[3][3] = {{-1, -1, -1}, {-1, 9, -1}, {-1, -1, -1}};

short hpf_filter_3[3][3] = {{1, -2, 1}, {-2, 5, -2}, {1, -2, 1}};

void averageing_smooth(cv::Mat src, cv::Mat dst);
void filter_smooth(cv::Mat src, cv::Mat dst, short filter[3][3], int type);

int main(int argc, char **argv) {
  cv::Mat src;
  cv::String imageName("lena.jpg"); // by default
  if (argc > 1) {
    imageName = argv[1];
  }
  src = imread(imageName, cv::IMREAD_COLOR);
  if (src.empty()) {
    std::cerr << "Error opening file\n\n";
    return -1;
  }
  cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
  double elapsed_smooth = read_timer();
  averageing_smooth(src, dst);
  elapsed_smooth = (read_timer() - elapsed_smooth);

  printf("=================================================================\n");
  printf("Calculating histogram for image\n");
  printf("-----------------------------------------------------------------\n");
  printf("averageing smoothing:\t\t\t\t%4f\n", elapsed_smooth * 1.0e3);

  filter_smooth(src, dst, lpf_filter_32, 32);
  namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
  namedWindow("New Image", cv::WINDOW_AUTOSIZE);
  imshow("Original Image", src);
  imshow("New Image", dst);
  cv::waitKey(0);

  return 0;
}

void averageing_smooth(cv::Mat src, cv::Mat dst) {

  for (int j = 1; j < src.rows - 1; j++) {
    for (int i = 1; i < src.cols - 1; i++) {
      unsigned int value[3];
      for (int x = 0; x < 3; x++) {
        value[x] += src.at<cv::Vec3b>(j, i)[x];
        value[x] = src.at<cv::Vec3b>(j - 1, i - 1)[x];
        value[x] += src.at<cv::Vec3b>(j + 1, i + 1)[x]; 
	value[x] += src.at<cv::Vec3b>(j, i + 1)[x];
        value[x] += src.at<cv::Vec3b>(j + 1, i)[x];
        value[x] += src.at<cv::Vec3b>(j - 1 , i)[x];
        value[x] += src.at<cv::Vec3b>(j, i - 1)[x];
        // take average
        value[x] /= 7.0;

        dst.at<cv::Vec3b>(j, i)[x] = value[x];
      }
    }
  }
}

void filter_smooth(cv::Mat src, cv::Mat dst, short filter[3][3], int type) {
  const int MAX_COLOR = 255;
  for (int j = 1; j < src.rows - 1; j++) {
    for (int i = 1; i < src.cols - 1; i++) {
      unsigned int value[3];
      for (int x = 0; x < 3; x++) {
        for (int a = -1; a < 2; a++)
          for (int b = -1; b < 2; b++)
            value[x] +=
                src.at<cv::Vec3b>(j + b, i + a)[x] * filter[a + 1][b + 1];

        value[x] /= type;
        value[x] = ((value[x] < 0) ? 0 : value[x]);
        value[x] = ((value[x] > MAX_COLOR) ? MAX_COLOR : value[x]);

        dst.at<cv::Vec3b>(j, i)[x] = value[x];
      }
    }
  }
}

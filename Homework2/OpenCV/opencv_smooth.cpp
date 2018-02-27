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

const int MAX_COLOR = 255;
void filter_smooth(cv::Mat src, cv::Mat dst, short filter[3][3]);
void filter_smooth_omp(cv::Mat src, cv::Mat dst, short filter[3][3]);

int main(int argc, char **argv) {
  cv::Mat src;
  cv::String imageName("../data/lena.jpg"); // by default
  bool batch = true;
  if (argc == 2) {
    imageName = argv[1];
  } else if (argc > 2) {
    batch = false;
    imageName = argv[2];
  }
  src = imread(imageName, cv::IMREAD_COLOR);
  if (src.empty()) {
    std::cerr << "Error opening file\n\n";
    return -1;
  }
  cv::Mat dst0 = cv::Mat::zeros(src.size(), src.type());

  double elapsed_smooth = read_timer();
  filter_smooth(src, dst0, lpf_filter_32);
  elapsed_smooth = (read_timer() - elapsed_smooth);

  cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
  double elapsed_omp = read_timer();
  filter_smooth_omp(src, dst, lpf_filter_32);
  elapsed_omp = (read_timer() - elapsed_omp);

  printf("=================================================================\n");
  printf("\t\tSmoothing image\n");
  printf("-----------------------------------------------------------------\n");
  printf("\t\t\t\tTime (ms)\n");
  printf("Filter smoothing:\t\t%4f\n", elapsed_smooth * 1.0e3);
  printf("Filter omp:\t\t\t%4f\n", elapsed_omp * 1.0e3);

  if (!batch) {
    namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
    imshow("Original Image", src);
    namedWindow("New Image", cv::WINDOW_AUTOSIZE);
    imshow("New Image", dst0);
    namedWindow("parallel", cv::WINDOW_AUTOSIZE);
    imshow("parallel", dst);

    cv::waitKey(0);
  }

  return 0;
}

void filter_smooth(cv::Mat src, cv::Mat dst, short filter[3][3]) {
  float weight = 0.0;
  for (int a = -1; a < 2; a++) {
    for (int b = -1; b < 2; b++) {
      weight += filter[a + 1][b + 1];
    }
  }

  for (int j = 1; j < src.rows - 1; j++) {
    for (int i = 1; i < src.cols - 1; i++) {
      unsigned int value[4];
      for (int x = 0; x < 4; x++) {
        for (int a = -1; a < 2; a++) {
          for (int b = -1; b < 2; b++) {
            value[x] +=
                src.at<cv::Vec3b>(j + b, i + a)[x] * filter[a + 1][b + 1];
          }
        }

        value[x] /= weight;
        value[x] = ((value[x] < 0) ? 0 : value[x]);
        value[x] = ((value[x] > MAX_COLOR) ? MAX_COLOR : value[x]);

        dst.at<cv::Vec3b>(j, i)[x] = value[x];
      }
    }
  }
}

void filter_smooth_omp(cv::Mat src, cv::Mat dst, short filter[3][3]) {
  float weight = 0.0;

  for (int a = -1; a < 2; a++) {
    for (int b = -1; b < 2; b++) {
      weight += filter[a + 1][b + 1];
    }
  }
  int i, j, x, a, b;
  unsigned int value[4];
// parallel over the rows and columns of the so the calculation can be done
// pixel by pixel
#pragma omp parallel for collapse(2) private(i, j, value)
  for (j = 1; j < src.rows - 1; j++) {
    for (i = 1; i < src.cols - 1; i++) {
      for (x = 0; x < 4; x++) {
        for (a = -1; a < 2; a++) {
          for (b = -1; b < 2; b++) {
            value[x] +=
                src.at<cv::Vec3b>(j + b, i + a)[x] * filter[a + 1][b + 1];
          }
        }

        value[x] /= weight;
        value[x] = ((value[x] < 0) ? 0 : value[x]);
        value[x] = ((value[x] > MAX_COLOR) ? MAX_COLOR : value[x]);

        dst.at<cv::Vec3b>(j, i)[x] = value[x];
      }
    }
  }
}

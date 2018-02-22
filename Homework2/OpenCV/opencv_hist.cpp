#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "omp.h"
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

void normalize(float histogram[256]);
void hist(cv::Mat src, float histogram[256]);
void normalize_omp(float histogram[256]);
void hist_omp(cv::Mat src, float histogram[256]);
void show_hist(float b[256], float g[256], float r[256]);

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

  float histogram_blue[256] = {0};
  float histogram_green[256] = {0};
  float histogram_red[256] = {0};

  float omp_blue[256] = {0};
  float omp_green[256] = {0};
  float omp_red[256] = {0};

  cv::Mat bgr[3];
  split(src, bgr);

  double elapsed_seq = read_timer();
  hist(bgr[2], histogram_red);
  hist(bgr[1], histogram_green);
  hist(bgr[0], histogram_blue);
  elapsed_seq = (read_timer() - elapsed_seq);

  double elapsed_omp = read_timer();
  hist_omp(bgr[2], omp_red);
  hist_omp(bgr[1], omp_green);
  hist_omp(bgr[0], omp_blue);
  elapsed_omp = (read_timer() - elapsed_omp);

  printf("=================================================================\n");
  printf("Calculating histogram for image\n");
  printf("-----------------------------------------------------------------\n");
  printf("\t\tTime (ms)\t\tMegaflops\n");
  printf("hist:\t\t%4f", elapsed_seq * 1.0e3);
  printf("\t\t%4f\n", (src.rows * src.cols * 3) / (elapsed_seq * 1.0e6));
  printf("hist_omp:\t%4f", elapsed_omp * 1.0e3);
  printf("\t\t%4f\n", (src.rows * src.cols * 3) / (elapsed_omp * 1.0e6));

  if (!batch)
    show_hist(omp_blue, omp_green, omp_red);
  return 0;
}

void normalize(float histogram[256]) {
  int max = histogram[0];
  for (int x = 0; x < 256; x++)
    max = max > histogram[x] ? max : histogram[x];

  for (int x = 0; x < 256; x++)
    histogram[x] /= (float)max;
}

void hist(cv::Mat src, float histogram[256]) {
  short k;
  for (int i = 0; i < src.cols; i++) {
    for (int j = 0; j < src.rows; j++) {
      k = src.at<uchar>(j, i);
      histogram[k] += 1;
    }
  }
  normalize(histogram);
}

void show_hist(float b[256], float g[256], float r[256]) {
  int histSize = 256;
  int hist_w = 512;
  int hist_h = 400;
  int bin_w = cvRound((double)hist_w / histSize);
  cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(1, 1, 1));
  for (int i = 1; i < histSize; i++) {
    line(histImage, cv::Point(bin_w * (i - 1), hist_h * (1 - b[i - 1])),
         cv::Point(bin_w * (i), hist_h * (1 - b[i])), cv::Scalar(255, 0, 0), 2,
         8, 0);
    line(histImage, cv::Point(bin_w * (i - 1), hist_h * (1 - g[i - 1])),
         cv::Point(bin_w * (i), hist_h - hist_h * g[i]), cv::Scalar(0, 255, 0),
         2, 8, 0);
    line(histImage, cv::Point(bin_w * (i - 1), hist_h * (1 - r[i - 1])),
         cv::Point(bin_w * (i), hist_h - hist_h * r[i]), cv::Scalar(0, 0, 255),
         2, 8, 0);
  }
  namedWindow("My Hist", cv::WINDOW_AUTOSIZE);
  imshow("My Hist", histImage);
  cv::waitKey(0);
}

void normalize_omp(float histogram[256]) {
  int max = 0;
#pragma omp parallel for reduction(max : max)
  for (int x = 0; x < 256; x++)
    max = max > histogram[x] ? max : histogram[x];

#pragma omp parallel for
  for (int x = 0; x < 256; x++)
    histogram[x] /= (float)max;
}

void hist_omp(cv::Mat src, float histogram[256]) {
  int max_threads = omp_get_max_threads();
  float temp[256][max_threads];

  short k = 0;

#pragma omp parallel private(k) shared(temp)
  {
    int thread = omp_get_thread_num();

// Loop to initiale temp to zero
#pragma omp parallel for
    for (int k = 0; k < 256; k++)
      temp[k][thread] = 0;

#pragma omp parallel for
    for (int i = 0; i < src.cols; i++) {
      for (int j = 0; j < src.rows; j++) {
        k = src.at<uchar>(j, i);
        temp[k][thread] += 1;
      }
    }
  } // End parallel section

  for (int t = 0; t < max_threads; t++)
    for (int k = 0; k < 256; k++)
      histogram[k] += temp[k][t];

  normalize_omp(histogram);
}

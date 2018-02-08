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

void normalize_hist(float histogram[256]);
void my_hist(cv::Mat src, float histogram[256]);
void show_hist(float b[256], float g[256], float r[256]);

int main(int argc, char **argv) {
  cv::Mat src;
  cv::String imageName("../data/lena.jpg"); // by default
  if (argc > 1) {
    imageName = argv[1];
  }
  src = imread(imageName, cv::IMREAD_COLOR);
  if (src.empty()) {
    std::cerr << "Error opening file\n\n";
    return -1;
  }

  float histogram_blue[256] = {0};
  float histogram_green[256] = {0};
  float histogram_red[256] = {0};

  cv::Mat bgr[3];
  split(src, bgr);

  double elapsed_my_blue = read_timer();
  my_hist(bgr[0], histogram_blue);
  elapsed_my_blue = (read_timer() - elapsed_my_blue);

  double elapsed_my_green = read_timer();
  my_hist(bgr[1], histogram_green);
  elapsed_my_green = (read_timer() - elapsed_my_green);

  double elapsed_my_red = read_timer();
  my_hist(bgr[2], histogram_red);
  elapsed_my_red = (read_timer() - elapsed_my_red);

  printf("=================================================================\n");
  printf("Calculating histogram for image\n");
  printf("-----------------------------------------------------------------\n");
  printf("blue:\t\t\t\t%4f\n", elapsed_my_blue * 1.0e3);
  printf("green:\t\t\t\t%4f\n", elapsed_my_green * 1.0e3);
  printf("red:\t\t\t\t%4f\n", elapsed_my_red * 1.0e3);

  show_hist(histogram_blue, histogram_green, histogram_red);
  return 0;
}

void normalize_hist(float histogram[256]) {
  int max = histogram[0];
  for (int x = 0; x < 256; x++) {
    if (histogram[x] > max)
      max = histogram[x];
  }
  for (int x = 0; x < 256; x++) {
    histogram[x] /= (float)max;
  }
}

void my_hist(cv::Mat src, float histogram[256]) {
  int k;
  for (int i = 0; i < src.cols; i++) {
    for (int j = 0; j < src.rows; j++) {
      k = src.at<uchar>(j, i);
      histogram[k] += 1;
    }
  }
  normalize_hist(histogram);
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

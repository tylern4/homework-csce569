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

void opencv_hist(cv::Mat src);
void my_hist(cv::Mat src, float histogram[256]);
void normalize_hist(float histogram[256]);
void csv_hist(float g[], float b[], float gr[], float r[]);

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

  float histogram_gray[256] = {0};
  float histogram_blue[256] = {0};
  float histogram_green[256] = {0};
  float histogram_red[256] = {0};

  cv::Mat bgr[3];
  split(src, bgr);
  double elapsed_my_gray = read_timer();
  my_hist(src, histogram_gray);
  elapsed_my_gray = (read_timer() - elapsed_my_gray);

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
  printf("gray:\t\t\t\t%4f\n", elapsed_my_gray * 1.0e3);
  printf("blue:\t\t\t\t%4f\n", elapsed_my_blue * 1.0e3);
  printf("green:\t\t\t\t%4f\n", elapsed_my_green * 1.0e3);
  printf("red:\t\t\t\t%4f\n", elapsed_my_red * 1.0e3);

  int histSize = 256;
  int hist_w = 512;
  int hist_h = 400;
  int bin_w = cvRound((double)hist_w / histSize);
  cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
  for (int i = 1; i < histSize; i++) {
    line(histImage,
         cv::Point(bin_w * (i - 1), hist_h - hist_h * histogram_blue[i - 1]),
         cv::Point(bin_w * (i), hist_h - hist_h * histogram_blue[i]),
         cv::Scalar(255, 0, 0), 2, 8, 0);
    line(histImage,
         cv::Point(bin_w * (i - 1), hist_h - hist_h * histogram_green[i - 1]),
         cv::Point(bin_w * (i), hist_h - hist_h * histogram_green[i]),
         cv::Scalar(0, 255, 0), 2, 8, 0);
    line(histImage,
         cv::Point(bin_w * (i - 1), hist_h - hist_h * histogram_red[i - 1]),
         cv::Point(bin_w * (i), hist_h - hist_h * histogram_red[i]),
         cv::Scalar(0, 0, 255), 2, 8, 0);
  }
  namedWindow("My Hist", cv::WINDOW_AUTOSIZE);
  imshow("My Hist", histImage);

  opencv_hist(src);
  cv::waitKey(0);
  // csv_hist(histogram_gray, histogram_blue, histogram_green, histogram_red);
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
  short k;
  for (int i = 0; i < src.cols; i++) {
    for (int j = 0; j < src.rows; j++) {
      k = src.at<uchar>(j, i);
      histogram[k] += 1;
    }
  }
  normalize_hist(histogram);
}

void opencv_hist(cv::Mat src) {

  vector<cv::Mat> bgr_planes;
  split(src, bgr_planes);
  int histSize = 256;
  float range[] = {0, 256};
  const float *histRange = {range};
  bool uniform = true;
  bool accumulate = false;
  cv::Mat b_hist, g_hist, r_hist;
  calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange,
           uniform, accumulate);
  calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange,
           uniform, accumulate);
  calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange,
           uniform, accumulate);
  // Draw the histograms for B, G and R
  int hist_w = 512;
  int hist_h = 400;
  int bin_w = cvRound((double)hist_w / histSize);
  cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
  normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
  normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
  normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
  for (int i = 1; i < histSize; i++) {
    line(histImage,
         cv::Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
         cv::Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
         cv::Scalar(255, 0, 0), 2, 8, 0);
    line(histImage,
         cv::Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
         cv::Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
         cv::Scalar(0, 255, 0), 2, 8, 0);
    line(histImage,
         cv::Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
         cv::Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
         cv::Scalar(0, 0, 255), 2, 8, 0);
  }
  namedWindow("calcHist Demo", cv::WINDOW_AUTOSIZE);
  imshow("calcHist Demo", histImage);
  cv::waitKey(0);
}

void csv_hist(float g[], float b[], float gr[], float r[]) {
  std::cerr << "x,gray,blue,green,red" << std::endl;
  for (int x = 0; x < 256; x++) {
    std::cerr << x << ",";
    std::cerr << g[x] << ",";
    std::cerr << b[x] << ",";
    std::cerr << gr[x] << ",";
    std::cerr << r[x] << endl;
  }
}

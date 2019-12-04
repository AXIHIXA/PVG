#include "auxiliary.h"
//#include "cuda_adaptive_evaluation.h"
#include "structure.h"
#include "region.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//void cuda_opencv_initialization(int w, int h)
//{
//    //CUDA initialization
//    std::cout << "CUDA Initialization ... ";
//    clock_t t = clock();
//    try
//    {
//        checkCudaErrors(cudaFree(0));
//    }
//    catch (const char * msg)
//    {
//        throw msg;
//    }
//    std::cout << (float) (clock() - t) / CLOCKS_PER_SEC << "s" << std::endl;
//
//    Mat mask = Mat::zeros(h, w, CV_8UC1);
//    dilate(mask, mask, Mat());
//}

bool is_image_boundary(const Region & region, int ring, int row, int col)
{
    return ((row == -ring || row == region.row() + ring - 1) && col >= -ring && col <= region.col() + ring - 1) ||
           ((col == -ring || col == region.col() + ring - 1) && row >= -ring && row <= region.row() + ring - 1);
}

void difference(const Mat & img1, const Mat & groundtruth, const cv::Rect & rect)
{
    Mat img2_;
    if (rect.height != 0 && rect.width != 0)
    {
        img2_ = groundtruth(rect);
    }
    else
    {
        img2_ = groundtruth;
    }

    if (img1.rows != img2_.rows || img1.cols != img2_.cols)
    {
        return;
    }

    Mat diff_img(img1.rows, img1.cols, CV_64FC3);
    diff_img.setTo(Vec3d(0, 0, 0));
    for (int i = 0; i < img1.rows; ++i)
    {
        for (int j = 0; j < img1.cols; ++j)
        {
            double b = img1.at< Vec3f >(i, j)[0] - img2_.at< Vec3f >(i, j)[0];
            double g = img1.at< Vec3f >(i, j)[1] - img2_.at< Vec3f >(i, j)[1];
            double r = img1.at< Vec3f >(i, j)[2] - img2_.at< Vec3f >(i, j)[2];
            diff_img.at< Vec3d >(i, j) = Vec3d(b, g, r);
        }
    }

    Mat lap(diff_img.size(), CV_64FC3);
    Laplacian(diff_img, lap, CV_64F);

    double norm_sum = 0;
    Vec3d sum(0, 0, 0);
    Vec3d max_val(0, 0, 0);
    for (int i = 0; i < lap.rows; ++i)
    {
        for (int j = 0; j < lap.cols; ++j)
        {
            Vec3d v = lap.at< Vec3d >(i, j);
            if (v[0] < 0)
            {
                v[0] = -v[0];
            }
            if (v[1] < 0)
            {
                v[1] = -v[1];
            }
            if (v[2] < 0)
            {
                v[2] = -v[2];
            }
            sum += v;
            double d = norm(lap.at< Vec3d >(i, j), NORM_L2);
            norm_sum += d;
            if (d > norm(max_val, NORM_L2))
            {
                max_val = lap.at< Vec3d >(i, j);
            }
        }
    }
    sum /= lap.rows * lap.cols;
    norm_sum /= lap.rows * lap.cols;
    //	imwrite("./resultImg/lap.png", abs(lap));
    cout << "max lap " << max_val << endl;
    cout << "average lap " << sum << endl;
    cout << "average lap norm " << norm_sum << endl;

    Mat texture = imread("./resultImg/colorbar.png");
    Mat color(diff_img.rows, diff_img.cols, CV_8UC3);
    int count = 0;
    double diff = 0;
    double max_d = 0.1 * sqrt(3 * 255 * 255);

    for (int i = 0; i < diff_img.rows; ++i)
    {
        for (int j = 0; j < diff_img.cols; ++j)
        {
            double d = norm(diff_img.at< Vec3d >(i, j), NORM_L2);
            diff += d;
            ++count;


            if (d > max_d)
            {
                d = max_d;
            }
            double t = d / max_d;
            t = 1 - t;
            int r = (int) round(t * (texture.rows - 1));
            int c = texture.cols / 2;
            color.at< Vec3b >(i, j) = texture.at< Vec3b >(r, c);
        }
    }
    imwrite("./resultImg/color.png", color);
    diff /= count;
    cout << "average relative error " << 100 * diff / sqrt(3 * 255 * 255) << "%" << endl;
}

#if 0

float atanf_Pade(float x)
{
    const float CORRECTED_VALUE = ((1.0f + 7.0f / 9.0f + 64.0f / 945.0f) / (1.0f + 10.0f / 9.0f + 5.0f / 21.0f) - 0.785398163f);
    float x2 = x*x;
    float x3 = x2*x;
    float x4 = x3*x;
    float x5 = x4*x;
    return (x + 7.0f / 9.0f*x3 + 64.0f / 945.0f*x5) / (1.0f + 10.0f / 9.0f*x2 + 5.0f / 21.0f*x4) - CORRECTED_VALUE*x;
}

float fast_atanf(float y, float x)
{
    bool b = fabsf(y) > fabsf(x);
    if (b) {
        float t = y;
        y = x;
        x = t;
    }

    float div = y / x;

    float v = atanf_Pade(div);

    if (div > 0 && b) return 1.570796327f - v;
    else if (div < 0 && b) return -1.570796327f - v;
    else return v;
}

float source(const float2* func, float pt_x, float pt_y)
{
    float x = pt_x - func->x;
    float y = pt_y - func->y;

    //float v1 = atan(y / x);
    float v1 = fast_atanf(y, x);

    float x2 = x*x;
    float y2 = y*y;
    float v = x*y*(logf(x2 + y2) - 3.0f) + (x2 - y2)*v1;
    y2 *= 1.570796327f;

    if (x == 0.0f || y == 0.0f)  v = y2 = 0.0f;

    return v1 > 0 ? v + y2 : v - y2;
}

float green_integral_(const float2& func, float x1, float x2, float y1, float y2)
{
    return source(&func, x1, y1) + source(&func, x2, y2) - source(&func, x1, y2) - source(&func, x2, y1);
}

#include <fstream>
void save_basis(const CPoint2d& pt, const vector<int>& index, const vector<float>& coefs, const vector<TreeNodeD>& nodes, const vector<CPoint2d>& knots)
{
    static int n = 0;
    ofstream file("./resultImg/basis/basis"+to_num<int,string>(n++));
    file << 201 << " " << 201 << endl;
    for (int ln = -100; ln <= 100; ++ln)
    {
        for (int col = -100; col <= 100; ++col)
        {
            float2 p;
            p.x= pt[0] +ln*0.1;
            p.y = pt[1] + col*0.1;
            float val = 0;
            for (size_t i = 0; i < index.size(); ++i)
            {
                val += coefs[i] * green_integral_(p, nodes[index[i]].row, nodes[index[i]].row + nodes[index[i]].width,
                    nodes[index[i]].col, nodes[index[i]].col + nodes[index[i]].width) / (nodes[index[i]].width*nodes[index[i]].width);
            }
            val *= 0.25f / 3.141592654f;
            file << ln*0.1 << " " << col*0.1 << " " << val << endl;
        }
    }
    for (size_t i = 0; i < knots.size(); ++i)
    {
        file << knots[i][0] - pt[0] << " " << knots[i][1] - pt[1] << endl;
    }
    file.close();
}

void write_obj(const cv::Mat& image, const char* filename)
{
    ofstream ofile(filename);
    ofile << "mtllib mtlfile.mtl" << endl;
    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
        {

            ofile << "v " << i << " " << j << " " << image.at<float>(i, j) << endl;
            ofile << "vt " << (float)j / (image.cols - 1) << " " << (float)(image.rows - i - 1) / (image.rows - 1) << endl;
        }
    }
    for (int i = 0; i < image.rows - 1; ++i)
    {
        for (int j = 0; j < image.cols - 1; ++j)
        {
            int id1 = i*image.cols + j + 1;
            int id2 = i*image.cols + j + 2;
            int id3 = (i + 1)*image.cols + j + 2;
            int id4 = (i + 1)*image.cols + j + 1;
            ofile << "f " << id1 << " " << id2 << " " << id3 << " " << id4 << endl;
        }
    }
    ofile.close();
}

void save_mesh(const Mat& image)
{
    Mat bgr[3];
    split(image, bgr);
    write_obj(bgr[0], "./resultImg/B.obj");
    write_obj(bgr[1], "./resultImg/G.obj");
    write_obj(bgr[2], "./resultImg/R.obj");

    ofstream mtl("./resultImg/mtifile.mtl");
    mtl << "newmtl img" << endl;
    mtl << "map_Kd name.png" << endl;
    mtl.close();
}
#endif

#include "adaptive_solver.h"
#include "region.h"
#include "auxiliary.h"
#include <tbb/tbb.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

AdaptiveSolver::AdaptiveSolver(const Region& region, int region_id, const QuadTree& tree, const cv::Mat& laplacian_image)
	:region(region), region_id(region_id), tree(tree), laplacian_image(laplacian_image), laplacian_matrix(tree.get_laplacian_solver())
{
	solved = false;
}

#ifdef DEBUG_TEST
Mat AdaptiveSolver::solve(vector<Vec3f>& color)
#else
void AdaptiveSolver::solve(vector<Vec3f>& color)
#endif
{
	if (!solved)
	{
		solver.compute(laplacian_matrix.topLeftCorner(tree.get_number_of_inner_points(), tree.get_number_of_inner_points()));
		if (solver.info() != Eigen::Success) {
			cout << "decomposition failed " << solver.info() << endl;
			exit(-1);
		}
		solved = true;
	}

	vector<TreeNodeD> regions;
	tree.get_regions(regions);

	vector<Eigen::VectorXd> X_lap(3, Eigen::VectorXd(tree.get_number_of_inner_points()));
	for (int i = 0; (int)i < regions.size(); ++i)
	{
		if (regions[i].type == INNER)
		{
			if (regions[i].row >= 0 && regions[i].row < region.row() && regions[i].col >= 0 && regions[i].col < region.col())
			{
				double area = regions[i].width*regions[i].width;
				X_lap[0](regions[i].index) = laplacian_image.at<Vec3f>(regions[i].row, regions[i].col)[0] * area;
				X_lap[1](regions[i].index) = laplacian_image.at<Vec3f>(regions[i].row, regions[i].col)[1] * area;
				X_lap[2](regions[i].index) = laplacian_image.at<Vec3f>(regions[i].row, regions[i].col)[2] * area;
			}
			else
			{
				X_lap[0](regions[i].index) = 0;
				X_lap[1](regions[i].index) = 0;
				X_lap[2](regions[i].index) = 0;
			}
		}
	}

	color.resize(tree.get_number_of_pixel_points());

	tbb::parallel_for(0, 3, 1, [&](int i)
	{
		Eigen::VectorXd X_boundary(this->tree.get_number_of_pixel_points());
		X_boundary.setZero();
		for (size_t j = 0; j < regions.size(); ++j)
		{
			if (regions[j].type == BOUNDARY)
			{
				X_boundary(regions[j].index) = laplacian_image.at<Vec3f>(regions[j].row, regions[j].col)[i];
			}
		}
		X_lap[i] -= laplacian_matrix*X_boundary;

		const Eigen::VectorXd& y = solver.solve(X_lap[i]);
		if (solver.info() != Eigen::Success)
		{
			cout << "solving failed " << solver.info() << endl;
			exit(-1);
		}

		for (int j = 0; j < y.rows(); ++j)
			color[j][i] = (float)y(j);
		for (size_t j = (size_t)y.rows(); j < color.size(); ++j)
			color[j][i] = (float)X_boundary(j);
	});

	//debug code
#ifdef DEBUG_TEST
	static Mat tmp = Mat::zeros(region.row(), region.col(), CV_32FC3);
	if (tmp.rows != region.row() || tmp.cols != region.col())
		tmp = Mat::zeros(region.row(), region.col(), CV_32FC3);
	static Mat tmp2 = Mat::zeros(region.row() + 2, region.col() + 2, CV_32FC3);
	static Mat block = Mat::zeros(region.row(), region.col(), CV_8UC3);

	for (size_t k = 0; k < regions.size(); ++k)
	{
		uchar blue, green, red;
		blue = rand() % 255;
		green = rand() % 255;
		red = rand() % 255;

		for (int i = regions[k].row; i < regions[k].row + regions[k].width; ++i)
		{
			for (int j = regions[k].col; j < regions[k].col + regions[k].width; ++j)
			{
				if (!is_image_boundary(region, 1, regions[k].row, regions[k].col))
				{
					if (regions[k].type == INNER) {
						tmp.at<Vec3f>(i, j) = color[regions[k].index];
						block.at<Vec3b>(i, j) = Vec3b(blue, green, red);
					}
				}
				if (regions[k].type == INNER) tmp2.at<Vec3f>(i + 1, j + 1) = color[regions[k].index];
			}
		}
	}
	for (int i = 0; i < tmp.rows; ++i)
	{
		for (int j = 0; j < tmp.cols; ++j)
		{
			if (!region.is_inner_of_a_region(i, j))
			{
				tmp.at<Vec3f>(i, j) = laplacian_image.at<Vec3f>(i, j);
				tmp2.at<Vec3f>(i + 1, j + 1) = laplacian_image.at<Vec3f>(i, j);
			}
		}
	}

	imwrite("./resultImg/padded_coef.png", tmp2);
	imwrite("./resultImg/block.png", block);

	/*Mat tmp2(tmp.rows, tmp.cols, CV_8UC3);
	for (int i = 0; i < tmp.rows; ++i)
	{
		for (int j = 0; j < tmp.cols; ++j)
		{
			float r = tmp.at<Vec3f>(i, j)[2];
			if (r < 0) r = 0;
			else if (r>255) r = 255;
			float g = tmp.at<Vec3f>(i, j)[1];
			if (g < 0) g = 0;
			else if (g>255) g = 255;
			float b = tmp.at<Vec3f>(i, j)[0];
			if (b < 0) b = 0;
			else if (b>255) b = 255;
			tmp2.at<Vec3b>(i, j) = Vec3b(round(b), round(g), round(r));
		}
	}*/
//	static int n = 0;
//	string name1 = string("./resultImg/fitting") + ".png";
//	cv::imwrite(name1.c_str(), tmp2);

//end debug
	return tmp;
#endif

}

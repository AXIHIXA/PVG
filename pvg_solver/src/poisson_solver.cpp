#define NOMINMAX
#include "poisson_solver.h"
#include "point_vector.h"
#include "quadtree.h"
#include "region.h"
#include "adaptive_solver.h"
#include "adaptive_evaluation.h"
#include "logger.h"
#include <algorithm>
#include <time.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <iostream>

using namespace cv;
using namespace std;

//#define OUTPUTLOG
//#define FULL_EVALUATION

#ifdef DEBUG_TEST
#include "auxiliary.h"
bool in(const Mat& img, cv::Point2f& p)
{
	return (int)p.x < img.rows && (int)p.x >= 0 && (int)p.y < img.cols && (int)p.y >= 0;
}


#endif

std::vector<std::vector<cv::Vec3f>> PoissonSolver::coefs;
std::vector<std::unique_ptr<QuadTree>> PoissonSolver::trees;
std::vector<std::unique_ptr<AdaptiveEvaluation>> PoissonSolver::evaluators;

#ifdef OUTPUTLOG
clock_t times[3] = { 0, 0, 0 };
#endif


void PoissonSolver::region_computation(int index, bool recompute, const BoundingBox<double>& box, bool simple_solver, int n_rings)
{
#ifdef OUTPUTLOG
	clock_t t = clock();
#endif
	if (recompute) trees[index].reset(new QuadTree(region, index, laplacian_image, 8, simple_solver, convert_to_laplacian_mask));

#ifdef OUTPUTLOG
	times[0] += clock() - t;
	t = clock();
#endif
	if (!trees[index]->with_inner_node()) return;

	if (recompute)
	{
		AdaptiveSolver solver(region, index, *trees[index], laplacian_image);
		solver.solve(coefs[index]);
	}
	if (simple_solver) return;

#ifdef DEBUG_TEST
	Logger::ins() << trees[index]->get_number_of_inner_points() << " " << trees[index]->get_number_of_inner_pixels() << "\n";
#endif

#ifdef OUTPUTLOG
	times[1] += clock() - t;
	t = clock();
#endif

	if (recompute) evaluators[index].reset(new AdaptiveEvaluation(*trees[index], coefs[index]));

	BoundingBox<int> box_;
	box_.row = (int)floor((box.row - original[0])*scale);
	box_.col = (int)floor((box.col - original[1])*scale);
	box_.width = std::min((int)ceil(box.width*scale) + 1, result.cols - box_.col);
	box_.height = std::min((int)ceil(box.height*scale) + 1, result.rows - box_.row);

//	if (!recompute)
	{
#ifdef FULL_EVALUATION
		critical_points[index] = evaluators[index]->full_solution(region, box_, 1.0 / scale, CPoint2d(box_.row / scale + original[0], box_.col / scale + original[1]), laplacian_image, n_rings, result);
#else
		critical_points[index] = evaluators[index]->cuda_evaluation(region, box_, 1.0 / scale, CPoint2d(box_.row / scale + original[0], box_.col / scale + original[1]), laplacian_image, n_rings, result);
#endif
	}

#ifdef OUTPUTLOG
	times[2] += clock() - t;
#endif

#ifdef DEBUG_TEST
	{
#if 0
		CPoint2i t[] = { CPoint2i(0, 1), CPoint2i(-1, 0), CPoint2i(1, 0), CPoint2i(0, -1) };
		vector<pair<CPoint2i, Vec3f>> boundary_pairs;
		Vec3f max_val(0, 0, 0);
		for (int j = 0; j < box.height; ++j)
		{
			for (int k = 0; k < box.width; ++k)
			{
				CPoint2i p(box.row + j, box.col + k);
				if (region.type(index, p) == BOUNDARY)
				{
					Vec3f val(0, 0, 0);
					for (int l = 0; l < 4; ++l)
					{
						CPoint2i pt = p + t[l];
						if (region.type(index, pt) == INNER)
						{
							val += accurate_coef.at<Vec3f>(pt[0], pt[1]) - coef.at<Vec3f>(pt[0], pt[1]); 
						}
					}
					boundary_pairs.push_back(make_pair(p, val));
					if (abs(val[0])>max_val[0]) max_val[0] = abs(val[0]);
					if (abs(val[1])>max_val[1]) max_val[1] = abs(val[1]);
					if (abs(val[2])>max_val[2]) max_val[2] = abs(val[2]);
				}
			}
		}
		cout << "max " << max_val << endl;

		for (int j = 0; j < box.height; ++j)
		{
			for (int k = 0; k < box.width; ++k)
			{
				CPoint2i p(box.row + j, box.col + k);
				if (region.type(index, p) == INNER)
				{
					Vec3f val(0, 0, 0);
					for (size_t l = 0; l < boundary_pairs.size(); ++l)
					{
						float2 pt;
						pt.x = box.row + j + 0.5f;
						pt.y = box.col + k + 0.5f;
						val += boundary_pairs[l].second *
							green_integral(pt, boundary_pairs[l].first[0], boundary_pairs[l].first[0] + 1,
							boundary_pairs[l].first[1], boundary_pairs[l].first[1] + 1);
					}
					result.at<Vec3f>(p[0], p[1]) += val;
				}
			}
		}
#endif
	}

	//interpolation
	{
#if 0
		vector<vector<TRIANGLE>> triangles = trees[index]->get_triangles();
		vector<TreeNodeD> nodes;
		trees[index]->get_regions(nodes);

		for (int j = 0; j < box.height; ++j)
		{
			for (int k = 0; k < box.width; ++k)
			{
				CPoint2f p(box.row + j + 0.5f, box.col + k + 0.5f);
				if (region.type(index, int(p[0]), int(p[1])) == INNER)
				{
					vector<int> the_index = trees[index]->get_neighbors(trees[index]->search(p));
					the_index.push_back(trees[index]->search(p));
					int id1 = -1;
					int id2 = -1;
					for (size_t m = 0; m < the_index.size(); ++m)
					{
						if (the_index[m] >= triangles.size()) continue;
						for (size_t l = 0; l < triangles[the_index[m]].size(); ++l)
						{
							vector<Point2f> pts;
							pts.push_back(triangles[the_index[m]][l].p[0]);
							pts.push_back(triangles[the_index[m]][l].p[1]);
							pts.push_back(triangles[the_index[m]][l].p[2]);
							double d = pointPolygonTest(pts, Point2f(p[0], p[1]), false);
							if (d >= 0){
								if (in(result, pts[0]) && in(result, pts[1]) && in(result, pts[2]))
								{
									id1 = the_index[m];
									id2 = l;
									break;
								}
							}
						}
					}
					if (id1 != -1)
					{
						Eigen::MatrixXd m(3, 3);
						for (int l = 0; l < 3; ++l)
						{
							m(0, l) = triangles[id1][id2].p[l].x;
							m(1, l) = triangles[id1][id2].p[l].y;
							m(2, l) = 1;
						}
						Eigen::VectorXd v(3);
						v(0) = p[0];
						v(1) = p[1];
						v(2) = 1;
						Eigen::VectorXd x = m.colPivHouseholderQr().solve(v);
						Vec3f c1 = coef.at<Vec3f>(triangles[id1][id2].p[0].x, triangles[id1][id2].p[0].y);
						Vec3f c2 = coef.at<Vec3f>(triangles[id1][id2].p[1].x, triangles[id1][id2].p[1].y);
						Vec3f c3 = coef.at<Vec3f>(triangles[id1][id2].p[2].x, triangles[id1][id2].p[2].y);
						Vec3f c = x(0)*c1 + x(1)*c2 + x(2)*c3;
						result.at<Vec3f>((int)p[0], (int)p[1]) = c;
					}
					else result.at<Vec3f>((int)p[0], (int)p[1]) = Vec3f(0, 0, 0);
				}
			}
		}
#endif
	}

#endif
}

PoissonSolver::PoissonSolver(bool recompute, const Size& canvas_size, const Mat& laplacian_image, const Region& region, const double scale, const CPoint2d& original, const vector<CPoint2d>& end_points, bool simple_solver, int n_rings, const Mat& convert_to_laplacian_mask)
	: laplacian_image(laplacian_image), region(region), scale(scale), original(original), critical_points(region.get_number_of_regions()), convert_to_laplacian_mask(convert_to_laplacian_mask)
{
	clock_t t_total = clock();

	result.create(canvas_size.height, canvas_size.width, CV_32FC3);
	result.setTo(Vec3f(-10000, -10000, -10000));

	BoundingBox<double> window_box(original[0], original[1], result.rows / scale, result.cols / scale);

	if (recompute)
	{
		coefs.clear();
		trees.clear();
		evaluators.clear();
		coefs.resize(region.get_number_of_regions());
		trees.resize(region.get_number_of_regions());
		evaluators.resize(region.get_number_of_regions());
	}
#ifdef DEBUG_TEST
	Logger::ins() << "number of regions " << region.get_number_of_regions() << "\n";
#endif

#ifdef DEBUG_TEST
	{
		cout << "number of region " << region.get_number_of_regions() << endl;
		Mat tmp_img1;
		for (int i = 0; i < region.get_number_of_regions(); ++i)
		{
			clock_t t = clock();
			QuadTree tree(region, i, laplacian_image, 1, false, Mat());

			if (!tree.with_inner_node()) continue;

			AdaptiveSolver solver(region, i, tree, laplacian_image);
			vector<Vec3f> tmp;
			tmp_img1 = solver.solve(tmp);
		}
		tmp_img1.copyTo(accurate_coef);
		imwrite("./resultImg/coef.png", accurate_coef);
		/*Mat tmp_img2;
		for (int i = 0; i < region.get_number_of_regions(); ++i)
		{
			clock_t t = clock();
			QuadTree tree(region, i, laplacian_image, 8, false, Mat());

			if (!tree.with_inner_node()) continue;

			AdaptiveSolver solver(region, i, tree, laplacian_image);
			vector<Vec3f> tmp;
			tmp_img2 = solver.solve(tmp);
		}
		tmp_img2.copyTo(coef);
		imwrite("./resultImg/fitting.png", coef);*/
	}
#endif

	vector<BoundingBox<double>> bounding_boxes(region.get_number_of_regions());
	for (int i = 0; i < region.get_number_of_regions(); ++i)
	{
		BoundingBox<int> box = region.get_boundingbox(i);
		bounding_boxes[i].width = box.width + 2;
		bounding_boxes[i].height = box.height + 2;
		bounding_boxes[i].row = box.row - 1;
		bounding_boxes[i].col = box.col - 1;

		bounding_boxes[i].intersection_boundingbox(window_box);
	}

#if defined(_DEBUG) || defined(DEBUG_TEST) || defined(QUADTREE_VORONOI_OUTPUT)
	for (int r = 0; r < region.get_number_of_regions(); ++r)
	{
		if (recompute || bounding_boxes[r].valid())
			region_computation(r, recompute, bounding_boxes[r], simple_solver, n_rings);
	}
#else
	tbb::parallel_for(0, region.get_number_of_regions(), 1, [&](int r)
	{
		if (recompute || bounding_boxes[r].valid())
			region_computation(r, recompute, bounding_boxes[r], simple_solver, n_rings);
	});
#endif

	if (simple_solver)
	{
		result.setTo(Vec3f(0, 0, 0));
		for (size_t index = 0; index < trees.size(); ++index)
		{
			vector<TreeNodeD> regions;
			trees[index]->get_regions(regions);
			for (size_t k = 0; k < regions.size(); ++k)
			{
				for (int i = regions[k].row; i < regions[k].row + regions[k].width; ++i)
				{
					for (int j = regions[k].col; j < regions[k].col + regions[k].width; ++j)
					{
						if (i >= 0 && i < result.rows && j >= 0 && j < result.cols)
						{
							if (regions[k].type == INNER) result.at<Vec3f>(i, j) = coefs[index][k];
						}
					}
				}
			}
		}
		for (int i = 0; i < result.rows; ++i)
		{
			for (int j = 0; j < result.cols; ++j)
			{
				if (!region.is_inner_of_a_region(i, j))
				{
					result.at<Vec3f>(i, j) = laplacian_image.at<Vec3f>(i, j);
				}
			}
		}
		cout << "solver total " << (float)(clock() - t_total) / CLOCKS_PER_SEC << "s\n";
		return;
	}

	if (scale > 1.0)
	{
		// fill singular pixels
		Mat point_label = Mat::zeros(result.size(), CV_8UC1);
		Mat mono_edge = Mat::zeros(result.size(), CV_8UC1);
		tbb::parallel_for(0, result.rows, 1, [&](int i)
		{
			for (int j = 0; j < result.cols; ++j)
			{
				if (result.at<Vec3f>(i, j) == Vec3f(-10000, -10000, -10000))
					point_label.at<uchar>(i, j) = 255;
				if (region.is_mono_edge_scaled_pt(i, j))
					mono_edge.at<uchar>(i, j) = 255;
			}
		});

		Region::bfs<uchar>(mono_edge, static_cast<int>(ceil(scale)));
		bitwise_or(point_label, mono_edge, point_label);

		for (size_t i = 0; i < critical_points.size(); ++i)
		{
			for (size_t j = 0; j < critical_points[i].size(); ++j)
			{
				point_label.at<uchar>(critical_points[i][j][0], critical_points[i][j][1]) = 255;
			}
		}

		Mat temp = Mat::zeros(result.rows, result.cols, CV_8UC1);
		int half_width = min((int)(6 * scale), max(result.rows, result.cols));
		BoundingBox<double> box(0, 0, result.rows, result.cols);
		for (size_t i = 0; i < end_points.size(); ++i)
		{
			CPoint2d c = scale*(end_points[i] - original);
			BoundingBox<double> box_;
			box_.row = c[0] - half_width;
			box_.col = c[1] - half_width;
			box_.height = 2 * half_width + 1;
			box_.width = 2 * half_width + 1;
			if (box_.intersection_boundingbox(box))
			{
				for (int r = 0; r < box_.height; ++r)
					for (int c = 0; c < box_.width; ++c)
						temp.at<uchar>(box_.row + r, box_.col + c) = 255;
			}
		}
		bitwise_or(point_label, temp, point_label);

		vector<CPoint2i> points;
		for (int i = 0; i < point_label.rows; ++i)
		{
			for (int j = 0; j < point_label.cols; ++j)
			{
				if (point_label.at<uchar>(i, j) != 0) points.push_back(CPoint2i(i, j));
			}
		}

		int round = min((int)floor(10 * scale), 100);
		int radius = max((int)ceil(scale / 10.0), 1);
		if (radius > 10) radius = 10;

		AdaptiveEvaluation::jacobian(result, region.get_enlarged_side_mask_source(), points, round, radius);
	}

#ifdef OUTPUTLOG
	cout << "tree time " << (float)times[0] / CLOCKS_PER_SEC << "s" << endl;
	cout << "linear equation time " << (float)times[1] / CLOCKS_PER_SEC << "s" << endl;
	cout << "rendering time " << (float)times[2] / CLOCKS_PER_SEC << "s" << endl;
#endif

	cout << "solver total " << (float)(clock() - t_total) / CLOCKS_PER_SEC << "s\n";
}

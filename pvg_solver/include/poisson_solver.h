#ifndef POISSONSOLVER_H
#define POISSONSOLVER_H
#include <memory>
#include <opencv2/core.hpp>
#include "point_vector.h"
#include "quadtree.h"

class Region;
class AdaptiveEvaluation;
template<class T> struct BoundingBox;

class PoissonSolver
{
private:
	static std::vector<std::vector<cv::Vec3f>> coefs;
	static std::vector<std::unique_ptr<QuadTree>> trees;
	static std::vector<std::unique_ptr<AdaptiveEvaluation>> evaluators;

	const double scale;
	const cv::Mat& laplacian_image;
	const cv::Mat& convert_to_laplacian_mask;
	const Region& region;
	const CPoint2d& original;

#ifdef DEBUG_TEST
	cv::Mat accurate_coef;
#endif

	cv::Mat result;
	std::vector<std::vector<CPoint2i>> critical_points;

	void region_computation(int index, bool recompute, const BoundingBox<double>& box, bool simple_solver, int n_rings);
public:
	PoissonSolver(bool recompute, const cv::Size& canvas_size, const cv::Mat& laplacian_image, const Region& region, const double scale, const CPoint2d& original, const std::vector<CPoint2d>& end_points, bool simple_solver, int n_rings, const cv::Mat& convert_to_laplacian_mask);
	const cv::Mat get_result_image() const
	{
		return result;
	}
#ifdef DEBUG_TEST
	const cv::Mat get_accurate_coef()
	{
		return accurate_coef;
	}
#endif
};

#endif

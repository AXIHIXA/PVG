#ifndef ADAPTIVE_EVALUATION_H
#define ADAPTIVE_EVALUATION_H


#include "point_vector.h"
#include "structure.h"
#include <Eigen/Sparse>
#include <opencv2/core.hpp>
#include <vector>


class Region;

class QuadTree;

class AdaptiveEvaluation
{
public:
    AdaptiveEvaluation(const int region_index, const QuadTree & quadtree, const std::vector<cv::Vec3f> & coef);

    ~AdaptiveEvaluation() = default;

    void full_solution(
            const Region & region,
            const BoundingBox<int> & rect,
            const CPoint2d & original,
            const cv::Mat & laplacian_image,
            int n_ring,
            cv::Mat & result);

private:
    static double source_double(const CPoint2d & func, double pt_x, double pt_y);

    static double green_integral_double(const CPoint2d & func, double x1, double x2, double y1, double y2);

private:
    const int region_index;
    const QuadTree & quadtree;
    std::vector<TreeNodeD> regions;
    const Eigen::SparseMatrix<double> & laplacian_matrix;

    const std::vector<cv::Vec3f> & coefs;

    std::vector<std::vector<int>> neighbors;

    const int default_rings = 4;
};

#endif

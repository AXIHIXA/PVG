#ifndef POISSONSOLVER_H
#define POISSONSOLVER_H


#include "point_vector.h"
#include "quadtree.h"
#include <memory>
#include <opencv2/core.hpp>


class Region;

class AdaptiveEvaluation;

template <class T>
struct BoundingBox;

class PoissonSolver
{
public:
    PoissonSolver(
            const cv::Size & canvas_size,
            const cv::Mat & laplacian_image,
            const Region & region,
            const double scale,
            const CPoint2d & origin,
            const std::vector<CPoint2d> & end_points,
            int n_rings,
            const cv::Mat & convert_to_laplacian_mask);

    const cv::Mat get_result_image() const
    {
        return result;
    }

private:
    void regionComputation(int index, const BoundingBox<double> & box, int n_rings);

private:
    static std::vector<std::vector<cv::Vec3f>> coefs;
    static std::vector<std::unique_ptr<QuadTree>> trees;
    static std::vector<std::unique_ptr<AdaptiveEvaluation>> evaluators;

    const double scale;
    const cv::Mat & laplacian_image;
    const cv::Mat & convert_to_laplacian_mask;
    const Region & region;
    const CPoint2d & origin;

    cv::Mat result;
    std::vector<std::vector<CPoint2i>> critical_points;
};


#endif

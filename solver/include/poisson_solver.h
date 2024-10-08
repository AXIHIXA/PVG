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
            const cv::Size & canvasSize,
            const cv::Mat & laplacianImage,
            const Region & region,
            double scale,
            const CPoint2d & origin,
            int n_rings);

    cv::Mat getResultImage() const
    {
        return result.clone();
    }

private:
    void regionComputation(int index, const BoundingBox<double> & box, int n_rings);

private:
    // solver
    static std::vector<std::unique_ptr<QuadTree>> trees;  // quadtree
    static std::vector<std::vector<cv::Vec3f>> coefs;     // control coefficient
    static std::vector<std::unique_ptr<AdaptiveEvaluation>> evaluators;

    // canvas
    const CPoint2d & origin;
    const double scale;

    // discretized primitives
    const cv::Mat & laplacianImage;
    const Region & region;

    // renerded final PVG result
    cv::Mat result;
};


#endif

#ifndef ADAPTIVESOLVER_H
#define ADAPTIVESOLVER_H

#include "point_vector.h"
#include "quadtree.h"
#include <Eigen/Sparse>
#include <opencv2/core.hpp>

class Region;

class AdaptiveSolver
{
public:
    AdaptiveSolver(const Region & region, int region_id, const QuadTree & tree, const cv::Mat & laplacian_image);

    void solve(std::vector<cv::Vec3f> & color);

private:
    bool solved;
    int region_id;
    const Region & region;
    const cv::Mat & laplacian_image;
    const QuadTree & tree;
    const Eigen::SparseMatrix<double> & laplacian_matrix;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
};

#endif

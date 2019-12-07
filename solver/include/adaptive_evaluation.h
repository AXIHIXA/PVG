#ifndef ADAPTIVE_EVALUATION_H
#define ADAPTIVE_EVALUATION_H


#include "point_vector.h"
//#include "cuda_adaptive_evaluation.h"
#include "structure.h"
#include <Eigen/Sparse>
#include <opencv2/core.hpp>
#include <vector>


class Region;

class QuadTree;

class AdaptiveEvaluation
{
public:
    AdaptiveEvaluation(const QuadTree & quadtree, const std::vector<cv::Vec3f> & coef);

    ~AdaptiveEvaluation();

//    // CUDA functions
//    std::vector<CPoint2i> cuda_evaluation(
//            const Region & region,
//            const BoundingBox<int> & rect,
//            double step,
//            const CPoint2d & original,
//            const cv::Mat & laplacian_image,
//            int n_ring,
//            cv::Mat & result);
//
//    static void jacobian(
//            cv::Mat & result,
//            const cv::Mat & enlarged_side_mask,
//            std::vector<CPoint2i> & points,
//            int round,
//            int radius);

    std::vector<CPoint2i> full_solution(
            const Region & region,
            const BoundingBox<int> & rect,
            double step,
            const CPoint2d & original,
            const cv::Mat & laplacian_image,
            int n_ring,
            cv::Mat & result);

private:
//    void transfer_variables();
//
//    void transfer_neigobors(const std::vector<std::vector<int>> & neighbor_array);
//
//    void free_neighbors();

private:
    const std::vector<cv::Vec3f> & coefs;
    const QuadTree & quadtree;
    const Eigen::SparseMatrix<double> & laplacian_matrix;

    std::vector<TreeNodeD> regions;
    std::vector<std::vector<int>> neighbors;
    std::vector<std::vector<int>> neighbors_small;
//    CudaVariable cuda_var;  //for cuda

    const int default_rings = 4;
    int neighbors_small_size;
};

#endif

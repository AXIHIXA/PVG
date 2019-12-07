#include "adaptive_solver.h"
#include "auxiliary.h"
#include "region.h"
#include "debug.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <tbb/tbb.h>


AdaptiveSolver::AdaptiveSolver(
        const Region & region,
        int region_id,
        const QuadTree & tree,
        const cv::Mat & laplacian_image
        ) :
        region(region),
        region_id(region_id),
        tree(tree),
        laplacian_image(laplacian_image),
        laplacian_matrix(tree.get_laplacian_solver()),
        solved(false)
{
}

void AdaptiveSolver::solve(std::vector<cv::Vec3f> & color)
{
    if (!solved)
    {
        solver.compute(laplacian_matrix.topLeftCorner(
                tree.get_number_of_inner_points(),
                tree.get_number_of_inner_points()));

        if (solver.info() != Eigen::Success)
        {
            st_error("decomposition failed %d", solver.info());
            abort();
        }

        solved = true;
    }

    std::vector<TreeNodeD> regions;
    tree.get_regions(regions);

    std::vector<Eigen::VectorXd> X_lap(3, Eigen::VectorXd(tree.get_number_of_inner_points()));

    for (int i = 0; (int) i < regions.size(); ++i)
    {
        if (regions[i].type == INNER)
        {
            if (regions[i].row >= 0 && regions[i].row < region.row() && regions[i].col >= 0 &&
                regions[i].col < region.col())
            {
                double area = regions[i].width * regions[i].width;
                X_lap[0](regions[i].index) = laplacian_image.at<cv::Vec3f>(regions[i].row, regions[i].col)[0] * area;
                X_lap[1](regions[i].index) = laplacian_image.at<cv::Vec3f>(regions[i].row, regions[i].col)[1] * area;
                X_lap[2](regions[i].index) = laplacian_image.at<cv::Vec3f>(regions[i].row, regions[i].col)[2] * area;
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
                X_boundary(regions[j].index) = laplacian_image.at<cv::Vec3f>(regions[j].row, regions[j].col)[i];
            }
        }

        X_lap[i] -= laplacian_matrix * X_boundary;

        const Eigen::VectorXd & y = solver.solve(X_lap[i]);

        if (solver.info() != Eigen::Success)
        {
            st_error("solving failed %d", solver.info());
            abort();
        }

        for (int j = 0; j < y.rows(); ++j)
        {
            color[j][i] = (float) y(j);
        }

        for (size_t j = (size_t) y.rows(); j < color.size(); ++j)
        {
            color[j][i] = (float) X_boundary(j);
        }
    });
}

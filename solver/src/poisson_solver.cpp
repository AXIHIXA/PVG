#define NOMINMAX


#include "adaptive_evaluation.h"
#include "adaptive_solver.h"
#include "debug.h"
#include "logger.h"
#include "point_vector.h"
#include "poisson_solver.h"
#include "quadtree.h"
#include "region.h"
#include <algorithm>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <tbb/tbb.h>
#include <ctime>


std::vector<std::vector<cv::Vec3f>> PoissonSolver::coefs;
std::vector<std::unique_ptr<QuadTree>> PoissonSolver::trees;
std::vector<std::unique_ptr<AdaptiveEvaluation>> PoissonSolver::evaluators;

static clock_t times[3] = { 0, 0, 0 };

PoissonSolver::PoissonSolver(
        const cv::Size & canvas_size,
        const cv::Mat & laplacian_image,
        const Region & region,
        const double scale,
        const CPoint2d & origin,
        const std::vector<CPoint2d> & end_points,
        int n_rings,
        const cv::Mat & convert_to_laplacian_mask
        ) :
        laplacian_image(laplacian_image),
        region(region),
        scale(scale),
        origin(origin),
        critical_points(region.get_number_of_regions()),
        convert_to_laplacian_mask(convert_to_laplacian_mask)
{
    assert(scale == 1.0);

    clock_t t_total = clock();

    result.create(canvas_size.height, canvas_size.width, CV_32FC3);
    result.setTo(cv::Vec3f(-10000, -10000, -10000));

    BoundingBox<double> window_box(
            origin[0], origin[1],
            result.rows / scale, result.cols / scale);

    coefs.clear();
    trees.clear();
    evaluators.clear();
    coefs.resize(region.get_number_of_regions());
    trees.resize(region.get_number_of_regions());
    evaluators.resize(region.get_number_of_regions());

    std::vector<BoundingBox<double>> bounding_boxes(region.get_number_of_regions());

    for (int i = 0; i < region.get_number_of_regions(); ++i)
    {
        BoundingBox<int> box = region.get_boundingbox(i);
        bounding_boxes[i].width = box.width + 2;
        bounding_boxes[i].height = box.height + 2;
        bounding_boxes[i].row = box.row - 1;
        bounding_boxes[i].col = box.col - 1;

        bounding_boxes[i].intersection_boundingbox(window_box);
    }

#if defined(QUADTREE_VORONOI_OUTPUT)
    for (int r = 0; r < region.get_number_of_regions(); ++r)
    {
        if (bounding_boxes[r].valid())
            region_computation(r, recompute, bounding_boxes[r], simple_solver, n_rings);
    }
#else
    tbb::parallel_for(0, region.get_number_of_regions(), 1, [&](int r)
    {
        if (bounding_boxes[r].valid())
        {
            regionComputation(r, bounding_boxes[r], n_rings);
        }
    });
#endif

    st_info("tree time %fs", (float) times[0] / CLOCKS_PER_SEC);
    st_info("linear equation time %fs", (float) times[1] / CLOCKS_PER_SEC);
    st_info("rendering time %fs", (float) times[2] / CLOCKS_PER_SEC);
    st_info("solver total %fs", (float) (clock() - t_total) / CLOCKS_PER_SEC);
}

void PoissonSolver::regionComputation(int index, const BoundingBox<double> & box, int n_rings)
{
    st_debug("regionComputation");

    clock_t t = clock();

    trees[index].reset(new QuadTree(region, index, laplacian_image, 8, false, convert_to_laplacian_mask));

    times[0] += clock() - t;
    t = clock();

    if (!trees[index]->with_inner_node())
    {
        return;
    }

    AdaptiveSolver solver(region, index, *trees[index], laplacian_image);
    solver.solve(coefs[index]);

    //trees[index]->get_number_of_inner_points() << " " << trees[index]->get_number_of_inner_pixels() << "\n";

    times[1] += clock() - t;
    t = clock();

    evaluators[index].reset(new AdaptiveEvaluation(*trees[index], coefs[index]));

    BoundingBox<int> box_;
    box_.row = (int) floor((box.row - origin[0]) * scale);
    box_.col = (int) floor((box.col - origin[1]) * scale);
    box_.width = std::min((int) ceil(box.width * scale) + 1, result.cols - box_.col);
    box_.height = std::min((int) ceil(box.height * scale) + 1, result.rows - box_.row);

    critical_points[index] = evaluators[index]->full_solution(
            region,
            box_,
            1.0 / scale,
            CPoint2d(box_.row / scale + origin[0], box_.col / scale + origin[1]),
            laplacian_image,
            n_rings,
            result);

    times[2] += clock() - t;
}
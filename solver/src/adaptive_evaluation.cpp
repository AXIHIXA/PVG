#include "adaptive_evaluation.h"
#include "auxiliary.h"
#include "debug.h"
#include "point_vector.h"
#include "quadtree.h"
#include "region.h"
#include "structure.h"
#include "tree.hh"
#include <Eigen/Sparse>
#include <opencv2/core.hpp>
#include <tbb/tbb.h>


using namespace std;
using namespace cv;


AdaptiveEvaluation::AdaptiveEvaluation(
        const int region_index,
        const QuadTree & quadtree,
        const std::vector<cv::Vec3f> & coefs
        ) :
        region_index(region_index),
        quadtree(quadtree),
        coefs(coefs),
        laplacian_matrix(quadtree.get_laplacian_basis())
{
    quadtree.get_regions(regions);

    // L1 nodes
    vector<TreeNodeD> nodes;
    quadtree.get_level1_nodes(nodes);

    // get neighbors
    neighbors.resize(nodes.size());

    tbb::parallel_for(0, (int) (nodes.size()), 1, [&](int i)
    {
        if (quadtree.with_inner_node(nodes[i]))
        {
            quadtree.get_neighbor_nodes(
                    nodes[i].center(),
                    neighbors[i],
                    default_rings,
                    8 * default_rings);
        }
    });
}

void AdaptiveEvaluation::full_solution(
        const Region & region,
        const BoundingBox<int> & rect,
        const CPoint2d & original,
        const cv::Mat & laplacian_image,
        int n_ring,
        cv::Mat & result)
{
    Eigen::MatrixXd v(coefs.size(), 3);

    for (size_t i = 0; i < coefs.size(); ++i)
    {
        v(i, 0) = coefs[i][0];
        v(i, 1) = coefs[i][1];
        v(i, 2) = coefs[i][2];
    }

    Eigen::MatrixXd vec = laplacian_matrix * v;
    vector<Vec3d> coef(vec.rows());

    for (size_t i = 0; i < coef.size(); ++i)
    {
        coef[i][0] = vec(i, 0);
        coef[i][1] = vec(i, 1);
        coef[i][2] = vec(i, 2);
    }

    vector<TreeNodeD> nodes;
    quadtree.get_regions(nodes);

    int * index = (int *) calloc(rect.width * rect.height, sizeof(int));

    tbb::parallel_for(0, rect.height, [&](int i)
    {
        st_debug("AdaptiveEvaluation::full_solution for region %d row %d", region_index, i);

        for (int j = 0; j < rect.width; ++j)
        {
            CPoint2d p(original[0] + (i + 0.5), original[1] + (j + 0.5));
            CPoint2d pt(p);

            if ((index[i * rect.width + j] != -1) &&
                (region.is_boundary((int) pt[0], (int) pt[1]) || region.is_singular((int) pt[0], (int) pt[1])))
            {
                index[i * rect.width + j] = -1;
                result.at<Vec3f>(rect.row + i, rect.col + j) = laplacian_image.at<Vec3f>((int) pt[0], (int) pt[1]);
            }

            if (index[i * rect.width + j] == 0)
            {
                int id = quadtree.search(pt);

                if (id >= 0 && id < quadtree.get_number_of_inner_points())
                {
                    Vec3d val(0, 0, 0);

                    for (size_t k = 0; k < coef.size(); ++k)
                    {
                        val += coef[k] *
                               green_integral_double(pt, nodes[k].row, nodes[k].row + nodes[k].width, nodes[k].col,
                                                     nodes[k].col + nodes[k].width) / (nodes[k].width * nodes[k].width);
                    }

                    val *= 0.25 / CV_PI;
                    result.at<Vec3f>(rect.row + i, rect.col + j) = val;
                }
            }
        }
    });
}

double AdaptiveEvaluation::source_double(const CPoint2d & func, double pt_x, double pt_y)
{
    double x = pt_x - func[0];
    double y = pt_y - func[1];

    double v1 = atan(y / x);

    double x2 = x * x;
    double y2 = y * y;
    double v = x * y * (log(x2 + y2) - 3.0) + (x2 - y2) * v1;
    y2 *= CV_PI / 2;

    if (x == 0.0 || y == 0.0)
    {
        v = y2 = 0.0;
    }

    return v1 > 0 ? v + y2 : v - y2;
}

double AdaptiveEvaluation::green_integral_double(const CPoint2d & func, double x1, double x2, double y1, double y2)
{
    return (source_double(func, x1, y1) + source_double(func, x2, y2) -
            source_double(func, x1, y2) - source_double(func, x2, y1));
}
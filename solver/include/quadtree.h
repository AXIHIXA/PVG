#ifndef QUADTREE_H
#define QUADTREE_H


#include "point_vector.h"
#include "structure.h"
#include "tree.hh"
#include <Eigen/Sparse>
#include <opencv2/core.hpp>
#include <vector>

//#define DEBUG_TEST
//#define QUADTREE_VORONOI_OUTPUT

#ifdef DEBUG_TEST
struct TRIANGLE
{
    cv::Point2f p[3];
};
#endif

class Region;

class QuadTree
{
public:
    // step must be 2^n
    QuadTree(
            const Region & region,
            int region_id,
            const cv::Mat & laplacian,
            int step,
            bool split_edge_neighbor,
            const cv::Mat edge_neighbor_mask);

    const Eigen::SparseMatrix<double> & get_laplacian_basis() const;

    const Eigen::SparseMatrix<double> & get_laplacian_solver() const;

    inline int get_number_of_inner_points() const
    {
        return inner_node_count;
    }

    inline int get_number_of_pixel_points() const
    {
        return pixel_node_count;
    }

    inline int get_number_of_all_points() const
    {
        return all_node_count;
    }

    bool in_range(const CPoint2f & p) const;

    int search(const CPoint2f & p) const;

    void get_regions(std::vector<TreeNodeD> & regions) const;

    void get_level1_nodes(std::vector<TreeNodeD> & nodes) const;

    void get_neighbor_nodes(const CPoint2d & p, std::vector<int> & neighbors, int n_rings, double radius) const;

    int search_level1(const CPoint2f & p) const;

    bool with_inner_node(const TreeNodeD & node) const;

    bool with_inner_node() const;

    int get_number_of_inner_pixels() const;

#ifdef DEBUG_TEST
    std::vector<int> get_neighbors(int i) const
    {
        return neighbor_nodes[i];
    }

    std::vector<std::vector<TRIANGLE>> get_triangles() const
    {
        return triangles;
    }
#endif

private:
    void construct_laplacian();

    inline bool to_split(
            const TreeNodeD & node,
            const cv::Mat & mask,
            const cv::Mat & pseudo_laplacian,
            const CPoint2i & mask_ori);

    void insert(std::vector<int> & neighbors, int row, int col, const CPoint2d & pt, double radius) const;

private:
    int region_id;
    int height;
    int width;
    int step;
    int pixel_node_count;
    int inner_node_count;
    int all_node_count;
    const Region & region;
    CPoint2i original;
    Eigen::SparseMatrix<double> laplacian_matrix_solver;
    Eigen::SparseMatrix<double> laplacian_matrix_basis;
    tree<TreeNodeD> quadtree;

    // for search acceleration
    std::vector<tree<TreeNodeD>::sibling_iterator> iterators;

#ifdef DEBUG_TEST
    // for n-ring neighbor search
    std::vector<std::vector<int>> neighbor_nodes;
    std::vector<std::vector<TRIANGLE>> triangles;
#endif
};

#endif

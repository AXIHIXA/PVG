#ifndef QUADTREE_H
#define QUADTREE_H


#include "point_vector.h"
#include "structure.h"
#include "tree.hh"
#include <Eigen/Sparse>
#include <opencv2/core.hpp>
#include <vector>


class Region;

class QuadTree
{
public:
    QuadTree(
            const Region & region,
            int region_id,
            const cv::Mat & laplacian,
            int step);

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

    inline bool in_range(const CPoint2f & p) const
    {
        return (origin[0] <= p[0] && p[0] < origin[0] + step * height &&
                origin[1] <= p[1] && p[1] < origin[1] + step * width);
    }

    int search(const CPoint2f & p) const;

    void get_regions(std::vector<TreeNodeD> & regions) const;

    void get_level1_nodes(std::vector<TreeNodeD> & nodes) const;

    void get_neighbor_nodes(const CPoint2d & p, std::vector<int> & neighbors, int n_rings, double radius) const;

    int search_level1(const CPoint2f & p) const;

    bool with_inner_node(const TreeNodeD & node) const;

    bool with_inner_node() const;

    int get_number_of_inner_pixels() const;

private:
    void construct_laplacian();

    inline bool to_split(
            const TreeNodeD & node,
            const cv::Mat & mask,
            const cv::Mat & pseudo_laplacian,
            const CPoint2i & mask_ori);

    void insert(std::vector<int> & neighbors, int row, int col, const CPoint2d & pt, double radius) const;

private:
    const Region & region;
    int region_id;
    int step;

    CPoint2i origin;
    int height;
    int width;

    int pixel_node_count;
    int inner_node_count;

    int all_node_count;

    Eigen::SparseMatrix<double> laplacian_matrix_solver;
    Eigen::SparseMatrix<double> laplacian_matrix_basis;

    tree<TreeNodeD> quadtree;

    // for search acceleration
    std::vector<tree<TreeNodeD>::sibling_iterator> iterators;
};

#endif

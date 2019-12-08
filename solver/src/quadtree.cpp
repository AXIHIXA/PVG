#include "auxiliary.h"
#include "Fade_2D.h"
#include "quadtree.h"
#include "region.h"
#include <Eigen/Sparse>
#include <opencv2/imgproc.hpp>


using namespace std;
using namespace cv;


QuadTree::QuadTree(
        const Region & region,
        int region_id,
        const cv::Mat & laplacian,
        int step
) :
        region(region),
        region_id(region_id),
        step(step)
{
    ///
    /// tree initialization
    ///

    const BoundingBox<int> & box = region.get_boundingbox(region_id);
    height = (int) ceil((box.height + 4) / (double) step);
    width = (int) ceil((box.width + 4) / (double) step);

    origin[0] = box.row - 2;
    origin[1] = box.col - 2;
    quadtree.set_head(TreeNodeD(origin[0], origin[1], -1));

    ///
    /// tree construction
    ///

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            quadtree.append_child(
                    quadtree.begin(),
                    TreeNodeD(origin[0] + i * step, origin[1] + j * step, step));
        }
    }

    ///
    /// create mask
    ///

    CPoint2i mask_ori(box.row - 2, box.col - 2);
    Mat mask = Mat::zeros(box.height + 4, box.width + 4, CV_8UC1);

    for (int ln = 0; ln < mask.rows; ++ln)
    {
        for (int col = 0; col < mask.cols; ++col)
        {
            if (region.type(region_id, mask_ori[0] + ln, mask_ori[1] + col) != OUTER
                || region.type(region_id, mask_ori[0] + ln + 1, mask_ori[1] + col) != OUTER
                || region.type(region_id, mask_ori[0] + ln - 1, mask_ori[1] + col) != OUTER
                || region.type(region_id, mask_ori[0] + ln, mask_ori[1] + col + 1) != OUTER
                || region.type(region_id, mask_ori[0] + ln, mask_ori[1] + col - 1) != OUTER
                || is_image_boundary(region, 2, mask_ori[0] + ln, mask_ori[1] + col))
            {
                mask.at<uchar>(ln, col) = 255;
            }
        }
    }

    Mat pseudo_laplacian = Mat::zeros(mask.rows, mask.cols, CV_32FC3);

    for (int ln = 0; ln < pseudo_laplacian.rows; ++ln)
    {
        for (int col = 0; col < pseudo_laplacian.cols; ++col)
        {
            PointType type = region.type(region_id, mask_ori[0] + ln, mask_ori[1] + col);
            if (type == INNER)
            {
                pseudo_laplacian.at<Vec3f>(ln, col) = laplacian.at<Vec3f>(mask_ori[0] + ln, mask_ori[1] + col);
            }
            else if (type == BOUNDARY)
            {
                pseudo_laplacian.at<Vec3f>(ln, col) = Vec3f(-1e10, -1e10, -1e10);
            }
            else
            {
                if (region.type(region_id, mask_ori[0] + ln + 1, mask_ori[1] + col) != OUTER
                    || region.type(region_id, mask_ori[0] + ln - 1, mask_ori[1] + col) != OUTER
                    || region.type(region_id, mask_ori[0] + ln, mask_ori[1] + col + 1) != OUTER
                    || region.type(region_id, mask_ori[0] + ln, mask_ori[1] + col - 1) != OUTER
                    || is_image_boundary(region, 2, mask_ori[0] + ln, mask_ori[1] + col))
                {
                    pseudo_laplacian.at<Vec3f>(ln, col) = Vec3f(-1e10, -1e10, -1e10);
                }
            }
        }
    }

    ///
    /// node splitting
    ///

    int cell_width = step;
    for (int depth = 1; cell_width > 1; ++depth)
    {
        if (quadtree.max_depth() < depth)
        {
            break;
        }

        cell_width /= 2;
        tree<TreeNodeD>::fixed_depth_iterator ite = quadtree.begin_fixed(quadtree.begin(), depth);

        while (quadtree.is_valid(ite))
        {
            if (should_split(*ite, mask, pseudo_laplacian, mask_ori))
            {
                quadtree.append_child(ite, TreeNodeD(ite->row, ite->col, cell_width));
                quadtree.append_child(ite, TreeNodeD(ite->row, ite->col + cell_width, cell_width));
                quadtree.append_child(ite, TreeNodeD(ite->row + cell_width, ite->col, cell_width));
                quadtree.append_child(ite, TreeNodeD(ite->row + cell_width, ite->col + cell_width, cell_width));
            }
            ++ite;
        }
    }

    ///
    /// label cell types
    ///

    tree<TreeNodeD>::leaf_iterator ite = quadtree.begin_leaf();
    while (quadtree.is_valid(ite))
    {
        ite->type = region.type(region_id, ite->row, ite->col);

        if (is_image_boundary(region, 1, ite->row, ite->col))
        {
            CPoint2i p = region.get_adjacent_image_border(CPoint2i(ite->row, ite->col));
            if (region.region_id(p) == region_id)
            {
                ite->type = INNER;
            }
        }
        ++ite;
    }

    ///
    /// leaf node index
    ///

    int count = 0;

    // inner nodes
    ite = quadtree.begin_leaf();

    while (quadtree.is_valid(ite))
    {
        if (ite->type == INNER)
        {
            ite->index = count++;
        }

        ++ite;
    }

    inner_node_count = count;

    // boundary nodes
    ite = quadtree.begin_leaf();

    while (quadtree.is_valid(ite))
    {
        if (ite->type == BOUNDARY)
        {
            ite->index = count++;
        }

        ++ite;
    }

    pixel_node_count = count;

    // 1-order outer nodes (all width 1)
    ite = quadtree.begin_leaf();

    while (quadtree.is_valid(ite))
    {
        if (ite->type == OUTER)
        {
            if (ite->width == 1 &&
                (region.type(region_id, ite->row + 1, ite->col) != OUTER
                 || region.type(region_id, ite->row - 1, ite->col) != OUTER
                 || region.type(region_id, ite->row, ite->col + 1) != OUTER
                 || region.type(region_id, ite->row, ite->col - 1) != OUTER
                 || is_image_boundary(region, 2, ite->row, ite->col))
                    )
            {
                ite->index = count++;
            }
            else
            {
                ite->index = -2;
            }
        }

        ++ite;
    }

    all_node_count = count;

    ///
    /// Laplacian Matrix
    ///

    construct_laplacian();

    ///
    /// prepare for search
    ///
    {
        tree<TreeNodeD>::sibling_iterator ite = quadtree.begin(quadtree.begin());

        while (quadtree.is_valid(ite))
        {
            iterators.push_back(ite);
            ite = quadtree.next_sibling(ite);
        }
    }
}

bool QuadTree::should_split(
        const TreeNodeD & node,
        const Mat & mask,
        const Mat & pseudo_laplacian,
        const CPoint2i & mask_ori)
{
    Vec3f stand(-2e10, -2e10, -2e10);

    for (int i = 0; i < node.width; ++i)
    {
        for (int j = 0; j < node.width; ++j)
        {
            CPoint2i p(node.row - mask_ori[0] + i, node.col - mask_ori[1] + j);

//            if (p[0] >= 0 && p[1] >= 0 && p[0] < mask.rows && p[1] < mask.cols)
            if (0 <= p[0] && p[0] < mask.rows && 0 <= p[1] && p[1] < mask.cols)
            {
                if (mask.at<uchar>(p[0], p[1]) != 0)
                {
                    if (stand == Vec3f(-2e10, -2e10, -2e10))
                    {
                        stand = pseudo_laplacian.at<Vec3f>(p[0], p[1]);

                        if (stand == Vec3f(-1e10, -1e10, -1e10))
                        {
                            return true;
                        }
                    }
                    else if (pseudo_laplacian.at<Vec3f>(p[0], p[1]) != stand)
                    {
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

int QuadTree::search(const CPoint2f & p) const
{
    int ind_row = (int) floor((p[0] - origin[0]) / step);
    int ind_col = (int) floor((p[1] - origin[1]) / step);

    if (ind_row < 0 || ind_row >= height || ind_col < 0 || ind_col >= width)
    {
        return -1;
    }

    int index = ind_row * width + ind_col;
    tree<TreeNodeD>::sibling_iterator it = iterators[index];

    while (quadtree.begin(it) != quadtree.end(it))
    {
        int ind_row = (int) floor((p[0] - it->row) / (it->width / 2));
        int ind_col = (int) floor((p[1] - it->col) / (it->width / 2));
        int index = ind_row * 2 + ind_col;
        it = quadtree.child(it, index);
    }

    return it->index;
}

void QuadTree::insert(std::vector<int> & neighbors, int row, int col, const CPoint2d & pt, double radius) const
{
    if (row >= 0 && col >= 0 && row < height && col < width)
    {
        int index = row * width + col;
        tree<TreeNodeD>::leaf_iterator it = quadtree.begin_leaf(iterators[index]);

        if (it == quadtree.end_leaf(iterators[index]))
        {
            if (iterators[index]->index >= 0 && iterators[index]->index < pixel_node_count)
            {
                const CPoint2d & p = iterators[index]->center();

                if (max(abs(p[0] - pt[0]), abs(p[1] - pt[1])) < radius)
                {
                    neighbors.push_back(iterators[index]->index);
                }
            }
        }
        else
        {
            while (it != quadtree.end_leaf(iterators[index]))
            {
                if (it->index >= 0 && it->index < pixel_node_count)
                {
                    const CPoint2d & p = it->center();

                    if (max(abs(p[0] - pt[0]), abs(p[1] - pt[1])) < radius)
                    {
                        neighbors.push_back(it->index);
                    }
                }

                ++it;
            }
        }
    }
}

void QuadTree::get_neighbor_nodes(const CPoint2d & p, vector<int> & neighbors, int n_rings, double radius) const
{
    int ind_row = int((p[0] - origin[0]) / step);
    int ind_col = int((p[1] - origin[1]) / step);

    const CPoint2d & pt = iterators[ind_row * width + ind_col]->center();

    neighbors.clear();
    neighbors.reserve(400);

    for (int r = 0; r <= n_rings; ++r)
    {
        for (int i = -r; i < r; ++i)
        {
            int row = ind_row + r;
            int col = ind_col + i;
            insert(neighbors, row, col, pt, radius);
        }

        for (int i = -r; i < r; ++i)
        {
            int row = ind_row - r;
            int col = ind_col + i;
            insert(neighbors, row, col, pt, radius);
        }

        for (int i = -r; i <= r; ++i)
        {
            int row = ind_row + i;
            int col = ind_col + r;
            insert(neighbors, row, col, pt, radius);
        }

        for (int i = -r + 1; i < r; ++i)
        {
            int row = ind_row + i;
            int col = ind_col - r;
            insert(neighbors, row, col, pt, radius);
        }
    }
}

inline int find(
        const vector<tuple<const GEOM_FADE2D::Point2 *, const GEOM_FADE2D::Point2 *, const GEOM_FADE2D::Point2 *>> & v,
        const GEOM_FADE2D::Point2 * ptr)
{
    for (size_t i = 0; i < v.size(); ++i)
    {
        if (get<0>(v[i]) == ptr)
        {
            return int(i);
        }
    }

    return -1;
}

void QuadTree::construct_laplacian()
{
    ///
    /// Fade2D Voronoi knots
    ///

    vector<GEOM_FADE2D::Point2> points;
    points.reserve(all_node_count);

    vector<tree<TreeNodeD>::leaf_iterator> points_list(all_node_count);
    tree<TreeNodeD>::leaf_iterator ite = quadtree.begin_leaf();

    while (quadtree.is_valid(ite))
    {
        if (ite->index >= 0)
        {
            CPoint2d c = ite->center();
            GEOM_FADE2D::Point2 p(c[0], c[1]);
            p.setCustomIndex(ite->index);
            points_list[ite->index] = ite;
            points.push_back(p);
        }
        ++ite;
    }

    ///
    /// Delaunay Triangulation
    ///

    GEOM_FADE2D::Fade_2D del_tri(static_cast<unsigned int>(points.size()));
    del_tri.insert(points);

    vector<GEOM_FADE2D::Triangle2 *> all_delaunay_triangles;
    del_tri.getTrianglePointers(all_delaunay_triangles);

    vector<GEOM_FADE2D::Point2> all_voronoi_knots(all_delaunay_triangles.size());

    for (size_t i = 0; i < all_delaunay_triangles.size(); ++i)
    {
        all_voronoi_knots[i] = all_delaunay_triangles[i]->getDual().first;
    }

    vector<vector<tuple<const GEOM_FADE2D::Point2 *, const GEOM_FADE2D::Point2 *, const GEOM_FADE2D::Point2 *>>>
            adjacent_list(pixel_node_count);

    for (size_t i = 0; i < all_delaunay_triangles.size(); ++i)
    {
        GEOM_FADE2D::Point2 * pt0 = all_delaunay_triangles[i]->getCorner(0);
        GEOM_FADE2D::Point2 * pt1 = all_delaunay_triangles[i]->getCorner(1);
        GEOM_FADE2D::Point2 * pt2 = all_delaunay_triangles[i]->getCorner(2);

        if (pt0->getCustomIndex() < pt1->getCustomIndex() && pt0->getCustomIndex() < pixel_node_count)
        {
            GEOM_FADE2D::Triangle2 * tri = del_tri.getAdjacentTriangle(pt1, pt0);

            if (tri != NULL)
            {
                GEOM_FADE2D::Point2 * v1 = &all_voronoi_knots[i];
                GEOM_FADE2D::Point2 * v2 = &all_voronoi_knots[tri - all_delaunay_triangles[0]];

                if (*v1 != *v2)
                {
                    adjacent_list[pt0->getCustomIndex()].push_back(make_tuple(pt1, v1, v2));

                    if (pt1->getCustomIndex() < pixel_node_count)
                    {
                        adjacent_list[pt1->getCustomIndex()].push_back(make_tuple(pt0, v1, v2));
                    }
                }
            }
        }

        if (pt1->getCustomIndex() < pt2->getCustomIndex() && pt1->getCustomIndex() < pixel_node_count)
        {
            GEOM_FADE2D::Triangle2 * tri = del_tri.getAdjacentTriangle(pt2, pt1);

            if (tri != NULL)
            {
                GEOM_FADE2D::Point2 * v1 = &all_voronoi_knots[i];
                int n = tri - all_delaunay_triangles[0];
                GEOM_FADE2D::Point2 * v2 = &all_voronoi_knots[tri - all_delaunay_triangles[0]];

                if (*v1 != *v2)
                {
                    adjacent_list[pt1->getCustomIndex()].push_back(make_tuple(pt2, v1, v2));

                    if (pt2->getCustomIndex() < pixel_node_count)
                    {
                        adjacent_list[pt2->getCustomIndex()].push_back(make_tuple(pt1, v1, v2));
                    }
                }
            }
        }

        if (pt2->getCustomIndex() < pt0->getCustomIndex() && pt2->getCustomIndex() < pixel_node_count)
        {
            GEOM_FADE2D::Triangle2 * tri = del_tri.getAdjacentTriangle(pt0, pt2);

            if (tri != NULL)
            {
                GEOM_FADE2D::Point2 * v1 = &all_voronoi_knots[i];
                GEOM_FADE2D::Point2 * v2 = &all_voronoi_knots[tri - all_delaunay_triangles[0]];

                if (*v1 != *v2)
                {
                    adjacent_list[pt2->getCustomIndex()].push_back(make_tuple(pt0, v1, v2));

                    if (pt0->getCustomIndex() < pixel_node_count)
                    {
                        adjacent_list[pt0->getCustomIndex()].push_back(make_tuple(pt2, v1, v2));
                    }
                }
            }
        }
    }

#ifdef QUADTREE_VORONOI_OUTPUT
    if (origin[0] > 0)
    {
        CPoint2i ori = origin + CPoint2i(1, 1);
        {
            int scale = 30;
            BoundingBox<int> box = region.get_boundingbox(region_id);
            Mat pix(scale*(box.height + 2), scale*(box.width + 2), CV_8UC3);
            pix.setTo(Vec3b(255, 255, 255));
            tree<TreeNodeD>::leaf_iterator ite = quadtree.begin_leaf();
            while (quadtree.is_valid(ite))
            {
                if (ite->type != OUTER)
                {
                    int row = scale*(ite->row - ori[0]);
                    int col = scale*(ite->col - ori[1]);
                    int width = ite->width;
                    for (int i = 0; i < width; ++i)
                    {
                        for (int j = 0; j < width; ++j)
                        {
                            cv::line(pix, Point(col + scale*i, row + scale*j), Point(col + scale*(i + 1), row + scale*j), CV_RGB(255, 153, 50));
                            cv::line(pix, Point(col + scale*(i + 1), row + scale*j), Point(col + scale*(i + 1), row + scale*(j + 1)), CV_RGB(255, 153, 50));
                            cv::line(pix, Point(col + scale*(i + 1), row + scale*(j + 1)), Point(col + scale*i, row + scale*(j + 1)), CV_RGB(255, 153, 50));
                            cv::line(pix, Point(col + scale*i, row + scale*(j + 1)), Point(col + scale*i, row + scale*j), CV_RGB(255, 153, 50));

                            Point pt(col + scale*(i + 0.5), row + scale*(j + 0.5));

                            if (ite->type == INNER) cv::circle(pix, pt, 8, CV_RGB(255, 255, 0), CV_FILLED);
                            else if (ite->type == BOUNDARY) cv::circle(pix, pt, 8, CV_RGB(0, 0, 255), CV_FILLED);
                            //	else if (ite->index > 0) cv::circle(vor, pt, 8, CV_RGB(255, 0, 0), CV_FILLED);
                        }
                    }
                }
                ++ite;
            }
            cv::imwrite("./resultImg/pixel.png", pix);
        }
        {
            int scale = 30;
            BoundingBox<int> box = region.get_boundingbox(region_id);
            Mat vor(scale*(box.height + 2), scale*(box.width + 2), CV_8UC3);
            vor.setTo(Vec3b(255, 255, 255));

            for (int i = 0; i < adjacent_list.size(); ++i)
            {
                for (int j = 0; j < adjacent_list[i].size(); ++j)
                {
                    Point p1(scale*(*get<1>(adjacent_list[i][j])).y() + 0.5, scale*(*get<1>(adjacent_list[i][j])).x() + 0.5);
                    Point p2(scale*(*get<2>(adjacent_list[i][j])).y() + 0.5, scale*(*get<2>(adjacent_list[i][j])).x() + 0.5);
                    p1.x -= scale*ori[1]; p1.y -= scale*ori[0];
                    p2.x -= scale*ori[1]; p2.y -= scale*ori[0];
                    int id = get<0>(adjacent_list[i][j])->getCustomIndex();
                    if (points_list[i]->type == BOUNDARY)
                    {
                        if (id < pixel_node_count) cv::line(vor, p1, p2, CV_RGB(255, 85, 255), 2, CV_AA);
                        else cv::line(vor, p1, p2, CV_RGB(255, 0, 85), 2, CV_AA);
                    }
                    else if (points_list[i]->type == INNER) cv::line(vor, p1, p2, CV_RGB(255, 153, 50), 2, CV_AA);
                    /**************************************/
                    //if (points_list[i]->type != OUTER) cv::line(vor, p1, p2, CV_RGB(255, 153, 50), 2, CV_AA);
                    /**************************************/

                    Point pt(scale*(*get<0>(adjacent_list[i][j])).y() + 0.5, scale*(*get<0>(adjacent_list[i][j])).x() + 0.5);
                    pt.x -= scale*ori[1]; pt.y -= scale*ori[0];
                    if (id < inner_node_count) cv::circle(vor, pt, 8, CV_RGB(255, 255, 0), CV_FILLED);
                    else if (id < pixel_node_count) cv::circle(vor, pt, 8, CV_RGB(0, 0, 255), CV_FILLED);
                    else cv::circle(vor, pt, 8, CV_RGB(255, 0, 0), CV_FILLED);
                    /**************************************/
                    /*CPoint2d p_ = points_list[i]->center();
                    p_ *= scale;
                    p_ -= scale*origin;
                    if (p_[1] > 3260 && p_[1] < 3280 && p_[0] < 4550 && p_[0] >4510)
                    {
                        cout << i << endl;
                    }
                    if (pt.x > 3260 && pt.x < 3280 && pt.y < 4550 && pt.y > 4510)
                        cv::circle(vor, pt, 8, CV_RGB(0, 255, 0), CV_FILLED);
                    else cv::circle(vor, pt, 8, CV_RGB(255, 255, 0), CV_FILLED);*/
                    /**************************************/
                }
            }

            tree<TreeNodeD>::leaf_iterator ite = quadtree.begin_leaf();
            while (quadtree.is_valid(ite))
            {
                int row = scale*(ite->row - ori[0]);
                int col = scale*(ite->col - ori[1]);
                int width = scale*ite->width;
                if (ite->index >= 0)
                {
                    cv::line(vor, Point(col, row), Point(col + width, row), CV_RGB(100, 100, 100));
                    cv::line(vor, Point(col + width, row), Point(col + width, row + width), CV_RGB(100, 100, 100));
                    cv::line(vor, Point(col + width, row + width), Point(col, row + width), CV_RGB(100, 100, 100));
                    cv::line(vor, Point(col, row + width), Point(col, row), CV_RGB(100, 100, 100));
                }
                ++ite;
            }

            cv::rectangle(vor, Point(0, 0), Point(vor.cols - 1, vor.rows - 1), CV_RGB(12, 175, 24), 6);

            static int count = 0;
            string name = string("./resultImg/voronoi") + to_num<int, string>(count++) + ".png";
            cv::imwrite(name.c_str(), vor);
        }
        {
            int scale = 30;
            BoundingBox<int> box = region.get_boundingbox(region_id);
            Mat qtree(scale*(box.height + 2), scale*(box.width + 2), CV_8UC3);
            qtree.setTo(Vec3b(255, 255, 255));

            int max_depth = quadtree.max_depth();
            for (int i = 1; i <= max_depth; ++i)
            {
                tree<TreeNodeD>::fixed_depth_iterator ite = quadtree.begin_fixed(quadtree.begin(), i);
                while (quadtree.is_valid(ite))
                {
                    int row = scale*(ite->row - ori[0]);
                    int col = scale*(ite->col - ori[1]);
                    int width = scale*ite->width;

                    if (quadtree.number_of_children(ite) == 0 && ite->index >= 0)
                    {
                        Scalar c = CV_RGB(141, 56, 201);
                        cv::line(qtree, Point(col, row), Point(col + width, row), c, 3);
                        cv::line(qtree, Point(col + width, row), Point(col + width, row + width), c, 3);
                        cv::line(qtree, Point(col + width, row + width), Point(col, row + width), c, 3);
                        cv::line(qtree, Point(col, row + width), Point(col, row), c, 3);

                        CPoint2d p = ite->center();
                        Point pt(scale*p[1] + 0.5, scale*p[0] + 0.5);
                        pt.x -= scale*ori[1]; pt.y -= scale*ori[0];

                        if (ite->type == INNER) cv::circle(qtree, pt, 8, CV_RGB(255, 255, 0), CV_FILLED);
                        else if (ite->type == BOUNDARY) cv::circle(qtree, pt, 8, CV_RGB(0, 0, 255), CV_FILLED);
                        else if (ite->index > 0) cv::circle(qtree, pt, 8, CV_RGB(255, 0, 0), CV_FILLED);
                    }

                    ++ite;
                }

                cv::rectangle(qtree, Point(0, 0), Point(qtree.cols - 1, qtree.rows - 1), CV_RGB(12, 175, 24), 6);

                static int count = 0;
                string name = string("./resultImg/tree") + to_num<int, string>(count++) + ".png";
                cv::imwrite(name.c_str(), qtree);
            }
        }
    }
#endif

    ///
    /// construct Laplacian matrix
    ///

    vector<Eigen::Triplet<double>> triplet_list_basis;
    triplet_list_basis.reserve(8 * adjacent_list.size());

    vector<Eigen::Triplet<double>> triplet_list_solver;
    triplet_list_solver.reserve(8 * adjacent_list.size());

    for (size_t i = 0; i < adjacent_list.size(); ++i)
    {
        double total_basis(0);
        double total_solver(0);

        const CPoint2d & p_ = points_list[i]->center();
        GEOM_FADE2D::Point2 pt(p_[0], p_[1]);

        for (size_t j = 0; j < adjacent_list[i].size(); ++j)
        {
            double d1 = (*get<1>(adjacent_list[i][j]) - *get<2>(adjacent_list[i][j])).length();
            double d2 = (pt - *get<0>(adjacent_list[i][j])).length();
            d1 /= d2;

            triplet_list_basis.push_back(Eigen::Triplet<double>(
                    points_list[get<0>(adjacent_list[i][j])->getCustomIndex()]->index,
                    (int) i,
                    d1));
            total_basis += d1;

            int r = points_list[get<0>(adjacent_list[i][j])->getCustomIndex()]->index;

            if (r < pixel_node_count)
            {
                total_solver += d1;

                if (r < inner_node_count)
                {
                    triplet_list_solver.push_back(Eigen::Triplet<double>(r, (int) i, -d1));
                }
            }
        }

        triplet_list_basis.push_back(Eigen::Triplet<double>((int) i, (int) i, -total_basis));

        if (i < (size_t) inner_node_count)
        {
            triplet_list_solver.push_back(Eigen::Triplet<double>((int) i, (int) i, total_solver));
        }
    }

    laplacian_matrix_basis.resize(all_node_count, pixel_node_count);
    laplacian_matrix_basis.setFromTriplets(triplet_list_basis.begin(), triplet_list_basis.end());

    laplacian_matrix_solver.resize(inner_node_count, pixel_node_count);
    laplacian_matrix_solver.setFromTriplets(triplet_list_solver.begin(), triplet_list_solver.end());
}

void QuadTree::get_regions(vector<TreeNodeD> & regions) const
{
    regions.resize(all_node_count);

    tree<TreeNodeD>::leaf_iterator ite = quadtree.begin_leaf();

    while (quadtree.is_valid(ite))
    {
        if (ite->index >= 0)
        {
            regions[ite->index] = *ite;
        }

        ++ite;
    }
}

void QuadTree::get_level1_nodes(std::vector<TreeNodeD> & nodes) const
{
    nodes.clear();
    tree<TreeNodeD>::fixed_depth_iterator ite = quadtree.begin_fixed(quadtree.begin(), 1);

    while (quadtree.is_valid(ite))
    {
        nodes.push_back(*ite);
        ++ite;
    }
}

bool QuadTree::with_inner_node(const TreeNodeD & node) const
{
    int n = search_level1(node.center());
    tree<TreeNodeD>::leaf_iterator ite = quadtree.begin_leaf(iterators[n]);

    if (ite == quadtree.end_leaf(iterators[n]))
    {
        return node.type == INNER;
    }

    while (ite != quadtree.end_leaf(iterators[n]))
    {
        if (ite->type == INNER)
        {
            return true;
        }

        ++ite;
    }

    return false;
}

bool QuadTree::with_inner_node() const
{
    tree<TreeNodeD>::leaf_iterator ite = quadtree.begin_leaf();

    while (ite != quadtree.end_leaf())
    {
        if (ite->type == INNER)
        {
            return true;
        }

        ++ite;
    }

    return false;
}

int QuadTree::search_level1(const CPoint2f & p) const
{
    unsigned int ind_row = static_cast<unsigned int>((p[0] - origin[0]) / step);
    unsigned int ind_col = static_cast<unsigned int>((p[1] - origin[1]) / step);

    return ind_row * width + ind_col;
}

int QuadTree::get_number_of_inner_pixels() const
{
    vector<TreeNodeD> nodes;
    get_regions(nodes);
    int n = 0;

    for (size_t i = 0; i < nodes.size(); ++i)
    {
        if (nodes[i].type == INNER)
        {
            n += nodes[i].width * nodes[i].width;
        }
    }

    return n;
}

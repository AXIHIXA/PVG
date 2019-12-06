#ifndef REGION_H
#define REGION_H


#include "point_vector.h"
#include "structure.h"
#include "Strokes.h"
#include <boost/multi_array.hpp>
#include <opencv2/core.hpp>


class Region
{
public:
    Region(const cv::Mat & mask, const cv::Mat & sideMask);

    void labels_generation();

    PointType type(int region_id, const CPoint2i & p) const;

    PointType type(int region_id, int ln, int col) const;

    int region_id(const CPoint2i & p) const;

    int region_id(int ln, int col) const;

    void set_side_source(const std::vector<std::vector<std::pair<cv::Vec2i, cv::Vec3f>>> & pts);

    CPoint2i find_closest_pixel(const CPoint2f & p, int id) const;

    int get_edge_id(const CPoint2f & p) const;

    void set_enlarged_mask(const cv::Mat & enlarged_mask, double scale, const BoundingBox<double> & ROI,
                           const std::vector<std::vector<std::vector<std::pair<cv::Vec2i, cv::Vec3f>>>> & points_vector);

    void add_SQ_strokes(const std::vector<SQ_Stroke> & strokes);

    int get_curve_index(const CPoint2f & p) const;

    CPoint2f get_neighbor_of_border(const CPoint2f & p) const;

    CPoint2f get_adjacent_image_border(const CPoint2i & p) const;

    CPoint2i find_nearby_point(const CPoint2f & p, const std::vector<int> & idx) const;

    template <class T>
    static void bfs(cv::Mat & mask, int n_ring)
    {
        std::vector<CPoint2i> pt_list;
        for (int i = 0; i < mask.rows; ++i)
        {
            for (int j = 0; j < mask.cols; ++j)
            {
                if (mask.at<T>(i, j) != 0)
                {
                    pt_list.push_back(CPoint2i(i, j));
                }
            }
        }

        for (int n = 0; n < n_ring; ++n)
        {
            std::vector<CPoint2i> added;
            for (int i = 0; i < pt_list.size(); ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    CPoint2i pt = pt_list[i] + offset[j];
                    if (pt[0] >= 0 && pt[0] < mask.rows && pt[1] >= 0 && pt[1] < mask.cols)
                    {
                        if (mask.at<T>(pt[0], pt[1]) == 0)
                        {
                            mask.at<T>(pt[0], pt[1]) = mask.at<T>(pt_list[i][0], pt_list[i][1]);
                            added.push_back(pt);
                        }
                    }
                }
            }
            pt_list = added;
        }
    }

    inline bool is_inner_of_a_region(int ln, int col) const
    {
        return mask.at<int>(ln, col) > 0;
    }

    inline bool is_singular(int ln, int col) const
    {
        return mask.at<int>(ln, col) == 0;
    }

    inline bool is_boundary(int ln, int col) const
    {
        return mask.at<int>(ln, col) < 0;
    }

    inline int col() const
    {
        return mask.cols;
    }

    inline int row() const
    {
        return mask.rows;
    }

    inline int get_number_of_regions() const
    {
        return number_of_regions;
    }

    inline BoundingBox<int> get_boundingbox(int index) const
    {
        return rects[index];
    }

    inline bool is_critical(const CPoint2f & p) const
    {
        int r = int(p[0]);
        int c = int(p[1]);
        if (r >= 0 && r < side_mask.rows && c >= 0 && c < side_mask.cols)
        {
            return side_mask.at<int>(r, c) != 0;
        }
        else
        {
            return false;
        }
    }

    inline const cv::Mat & get_enlarged_side_mask_source() const
    {
        return enlarged_side_mask_source;
    }

    inline bool is_mono_edge(int ln, int col) const
    {
        if (ln >= 0 && ln < mono_mask.rows && col >= 0 && col < mono_mask.cols)
        {
            return mono_mask.at<uchar>(ln, col) != 0;
        }
        else
        {
            return false;
        }
    }

    inline bool is_mono_edge_scaled_pt(int ln, int col) const
    {
        CPoint2f p = to_source_pt(CPoint2f(ln, col));
        return is_mono_edge((int) p[0], (int) p[1]);
    }

    inline const cv::Mat get_region_mask() const
    {
        return mask;
    }

    inline bool is_border(const CPoint2f & p) const
    {
        return int(p[0]) == 0 || int(p[0]) == side_mask.rows - 1 || int(p[1]) == 0 || int(p[1]) == side_mask.cols - 1;
    }

    inline std::vector<int> crossing_index(const CPoint2f & p) const
    {
        CPoint2f pt = to_scaled_pt(p);
        if ((int) pt[0] >= 0 && (int) pt[0] < enlarged_crossing_region.shape()[0] && (int) pt[1] >= 0 &&
            (int) pt[1] < enlarged_crossing_region.shape()[1])
        {
            if (!end_points_mask[(int) pt[0]][(int) pt[1]].empty())
            {
                return std::vector<int>();
            }
            else
            {
                return enlarged_crossing_region[(int) pt[0]][(int) pt[1]];
            }
        }
        else
        {
            return std::vector<int>();
        }
    }

    inline bool in_crossing(const CPoint2f & p) const
    {
        CPoint2f pt = to_scaled_pt(p);
        if ((int) pt[0] >= 0 && (int) pt[0] < enlarged_crossing_region.shape()[0] && (int) pt[1] >= 0 &&
            (int) pt[1] < enlarged_crossing_region.shape()[1])
        {
            return !enlarged_crossing_region[(int) pt[0]][(int) pt[1]].empty();
        }
        return false;
    }

private:
    CPoint2f to_source_pt(const CPoint2f & p) const;

    CPoint2f to_scaled_pt(const CPoint2f & p) const;

    void reflect_side_mask(double scale, const BoundingBox<double> & ROI);

    void set_side_enlarged(const std::vector<std::vector<std::pair<cv::Vec2i, cv::Vec3f>>> & pts);

    void bfs(boost::multi_array<std::vector<int>, 2> & mask, int n_ring);

    static bool is_end_point(const cv::Mat & mask, const CPoint2i & p);

private:
    typedef boost::multi_array<std::vector<std::pair<int, CPoint2i>>, 2> NEARPTS;

    static const CPoint2i offset[8];

    double scale;
    int number_of_regions;
    int side_mask_index;
    int enlarged_side_mask_index;
    int enlarged_side_mask_index_backup;

    NEARPTS near_pts;
    const cv::Mat mask;
    cv::Mat enlarged_side_mask_source;
    cv::Mat side_mask;
    cv::Mat enlarged_side_mask;
    cv::Mat mono_mask;
    boost::multi_array<std::vector<int>, 2> crossing_region;
    boost::multi_array<std::vector<int>, 2> enlarged_crossing_region;
    boost::multi_array<std::vector<int>, 2> end_points_mask;

    BoundingBox<double> ROI;
    std::vector<BoundingBox<int>> rects;

    std::vector<SQ_Stroke> strokes;
};


#endif

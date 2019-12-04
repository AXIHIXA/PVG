#ifndef PATHSTROKE_AUXILIARY_H
#define PATHSTROKE_AUXILIARY_H


#include "point_vector.h"
#include "structure.h"
#include <QVector>
#include <vector>

extern CPoint2i trans[];

class SQ_Stroke;

struct Frame
{
    QVector< SQ_Stroke > m_strokes;                                           // boundary color edges
    QVector< SQ_Stroke > lap_edges;                                           // Laplacian edges
    QVector< float > lap_edges_control_parameters;
    QVector< SQ_Stroke > lap_regions;                                         // Laplacian regions
    QVector< QPair< cv::Vec3f, cv::Vec3f > > lap_regions_control_parameters;
};

template < class T >
int count_neighbor(const cv::Mat & image, const CPoint2i & p, T id)
{
    int count = 0;
    for (int i = 0; i < 4; ++i)
    {
        CPoint2i pt = p + trans[i];
        if (pt[0] >= 0 && pt[0] < image.rows && pt[1] >= 0 && pt[1] < image.cols)
        {
            if (image.at< T >(pt[0], pt[1]) == id)
            {
                ++count;
            }
        }
    }
    return count;
}

cv::Vec3f evaluate_laplacian(const cv::Mat & image, const CPoint2i & p);

std::vector< std::vector< CPoint2i>> extract_curves(const cv::Mat & mono_image);

void flood_fill(cv::Mat region_mask);

bool in_region(const SQ_Stroke & stroke, const cv::Mat & region);

cv::Mat combine_region_map(const cv::Mat r1, const cv::Mat r2, const CPoint2i & corner);

CPoint2i find_nearest_points(const cv::Mat & region_mask, const cv::Mat & side_mask, const int sign, const CPoint2i & p);

void update_colors(SQ_Stroke & stroke, const cv::Mat & result, const cv::Mat & region_mask, const cv::Mat & side_mask,
                   const cv::Mat & roi, const CPoint2i & corner);

std::pair< std::vector< std::vector< cv::Vec2i>>, std::vector< std::vector< cv::Vec2i > > >
lap_edge_splited(int height, int width, const SQ_Stroke & stroke,
                 const std::vector< std::vector< std::pair< cv::Vec2i, cv::Vec3f>> > & lap_edges);

BoundingBox< double > get_boundingbox(const Frame & frame);

#endif

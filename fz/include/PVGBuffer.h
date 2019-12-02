//
// Created by ax on 11/30/19.
//

#ifndef PVG_PVGBUFFER_H
#define PVG_PVGBUFFER_H

#include "RegionFZ.h"
#include "Strokes.h"
#include "tinyxml2.h"
#include <memory>
#include <QPoint>
#include <QString>


class PVGBuffer
{
public:
    explicit PVGBuffer(double sf = 1.0, double lsf = 1.0, const QPoint & w = QPoint(0, 0));

    ~PVGBuffer() = default;

    /// parse .pvg file
    /// \param filename
    /// \param scale
    /// \return true if success, false if error occurs
    bool open(const QString & filename, double scale);

private:
    static SQ_Stroke parseStroke(const tinyxml2::XMLElement * SQ_Stroke_ele, double scale, bool parseProperty);

    void discretization();

    void zoomIn(double scale, const QPoint& cur_w00);

private:
    struct LapPoint
    {
        cv::Vec2i pt;
        cv::Vec3f lap;
    };

    // cpu capacity
    unsigned int max_threads;

    // resize magnifier
    double scaleFactor;
    double lastScaleFactor;
    QPoint w00;

    // PVG primitives
    QSize size;                                                               // size of this PVG image

    QVector< SQ_Stroke > m_strokes;                                           // boundary color edges

    QVector< SQ_Stroke > lap_edges;                                           // Laplacian edges
    QVector< float > lap_edges_control_parameters;

    QVector< SQ_Stroke > lap_regions;                                         // Laplacian regions
    QVector< QPair< cv::Vec3f,cv::Vec3f > > lap_regions_control_parameters;

    QVector< LapPoint > lap_points;

    // scaled PVG primitives
    QVector< SQ_Stroke > m_strokes_scaled;
    QVector< SQ_Stroke > lap_edges_scaled;
    QVector< SQ_Stroke > lap_regions_scaled;

    QVector< int > overwrite_id;

    // regions
    std::vector< std::unique_ptr< CRegionFZ > > regions;
    cv::Mat result;
};

#endif //PVG_PVGBUFFER_H

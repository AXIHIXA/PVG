#ifndef PVG_PVG_H
#define PVG_PVG_H


#include "region.h"
#include "RegionFZ.h"
#include "Strokes.h"
#include "tinyxml2.h"
#include <memory>
#include <QPoint>
#include <QString>


class PVG
{
public:
    ///
    /// PVG
    /// \brief constructor
    /// \param filename
    /// \param sf
    /// \param w
    ///
    explicit PVG(const QString & filename, double sf = 1.0, const QPoint & w = QPoint(0, 0));

    ~PVG() = default;

    ///
    /// save
    /// \brief save rendered PVG file to disk
    /// \param filename
    ///
    void save(const QString & filename);

    ///
    /// decompress
    /// \brief decompress .pvg file into xml and save to disk
    /// \param filename
    ///
    static void decompress(const QString & filename);

private:
    ///
    /// open
    /// \brief parse & compute .pvg file
    /// \param filename
    /// \param scale
    ///
    void open(const QString & filename, double scale);

    static SQ_Stroke parseStroke(const tinyxml2::XMLElement * SQ_Stroke_ele, double scale, bool parseProperty);

    void zoomIn(double scale, const QPoint & cur_w00);

    QPair<Region *, cv::Mat> discretization();

    void evaluation(Region * region, const cv::Mat & laplacian_image, int n_rings);

private:
    // cpu capacity
    unsigned int max_threads;

    ///
    /// PVG primitives
    ///

    // size of this PVG image
    QSize size;

    // resize magnifier
    double scaleFactor;
    QPoint w00;

    // primitives

    // boundary color edges
    QVector<SQ_Stroke> m_strokes;

    // Laplacian edges
    QVector<SQ_Stroke> lap_edges;
    QVector<float> lap_edges_control_parameters;

    // Laplacian regions
    QVector<SQ_Stroke> lap_regions;
    QVector<QPair<cv::Vec3f, cv::Vec3f> > lap_regions_control_parameters;

    // scaled PVG primitives
    QVector<SQ_Stroke> m_strokes_scaled;
    QVector<SQ_Stroke> lap_edges_scaled;
    QVector<SQ_Stroke> lap_regions_scaled;

    // CRegion_FZ regions buffer used for lap edges / regions parallel computation
    std::vector<std::unique_ptr<CRegionFZ> > regions;

    ///
    /// computed results
    ///

    // rendered final result
    cv::Mat result;
};


#endif  // PVG_PVG_H

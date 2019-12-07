#ifndef STROKES_H
#define STROKES_H


#include <opencv2/core.hpp>
#include <QtWidgets>


struct PointProperties
{
    QColor color_1;
    QColor color_2;
    bool keyframe_1;
    bool keyframe_2;
};

class SQ_Stroke
{
public:
    SQ_Stroke() = default;

    ~SQ_Stroke() = default;

    enum StrokeMode
    {
        CLOSED, OPEN
    };

    enum MDMode
    {
        MONO, DUAL
    };

    int sideEndpoint(const QPointF & p, double scale) const;

    int sideTangent(const QPointF & p) const;

    void translation(const QPointF & t);

    void scaleVertical(double s);

    void scaleHorizontal(double s);

    void updateAll();

    void initProperties(const QColor & c11, const QColor & c12, const QColor & c21, const QColor & c22, MDMode mode);

    bool setPointColor(int index, const QColor & color, int side_index);

    bool deletePointColor(int index, int side_index);

    bool resetKeyframe(int index_i, int index_j, int side_index);

    bool updateColor(int index_i, int index_j, int mode);

    void swapColor();

    qreal length() const;

    bool insertTangent(const QPointF & p, const cv::Vec3d & line);

    void clearTangents();

    void scale(double s);

    SQ_Stroke & operator=(const SQ_Stroke & s);

public:
    QVector<QPair<QPointF, cv::Vec3d>> tangents;

    QPointF dir_f;
    QPointF dir_b;
    QVector<QPointF> s_points;
    QVector<PointProperties> s_properties;

    QVector<QPointF> segs;
    QVector<PointProperties> pps;
    QVector<int> idx;

    StrokeMode s_mode;  // closed=0 open=1
    MDMode s_mdmode;    // define mono/dual line !!!DIFFERENT from s_mode!!! mono=0 dual=1
};


SQ_Stroke linear_interpolation(const SQ_Stroke & s1, const SQ_Stroke & s2, double rate);  // return rate*(s2-s1)

#endif

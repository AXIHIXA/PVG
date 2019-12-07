#pragma once

#include "psimpl.h"
#include "Strokes.h"

class CSamplSimpl
{
public:
    CSamplSimpl(void)
    {}

    ~CSamplSimpl(void)
    {}

    inline bool isEqual(QPointF a, QPointF b)
    {
        return a.x() == b.x() && a.y() == b.y();
    };

    inline void clean(SQ_Stroke & stroke)
    {
        stroke.s_points.clear();
        stroke.s_properties.clear();
        stroke.segs.clear();
        stroke.pps.clear();
    };

    void simplify(SQ_Stroke & stroke, float sample_rate)
    {
        if (stroke.s_points.size() < 2)
        { return; }

        unsigned tolerance = stroke.s_points.size() * sample_rate / 100;          // point count tolerance
        //unsigned tolerance = 1 + (100 - sample_rate)*0.2;
        tolerance = tolerance < 2 ? 2 : tolerance;

        if (10 - 1e-6 < sample_rate && sample_rate < 10 + 1e-6)
        {
            tolerance = 2;
        }
        std::vector< double > polyline;   // original polyline, assume not empty
        std::vector< double > result;      // resulting simplified polyline

        //printf("%d %f\n", stroke.s_points.size(), sample_rate);

        newStroke = stroke;
        clean(newStroke);

        bool f = false;
        QPointF last_pt;
        PointProperties last_prop;
        if (stroke.s_points[0] == stroke.s_points.back())
        {
            f = true;
            last_pt = stroke.s_points.back();
            last_prop = stroke.s_properties.back();
            stroke.s_points.pop_back();
        }

        for (int i = 0; i < stroke.s_points.size(); i++)
        {
            polyline.push_back(stroke.s_points[i].x());
            polyline.push_back(stroke.s_points[i].y());
        }

        psimpl::simplify_douglas_peucker_n< 2 >(polyline.begin(), polyline.end(),
                                                tolerance, std::back_inserter(result));
        //psimpl::simplify_douglas_peucker <2>(polyline.begin(), polyline.end(),
        //	tolerance, std::back_inserter(result));

        int j = 0;
        for (int i = 0; i < stroke.s_points.size(); i++)
        {
            if (stroke.s_points[i] == QPointF(result[j], result[j + 1]))
            {
                newStroke.s_points.push_back(QPointF(result[j], result[j + 1]));
                newStroke.s_properties.push_back(stroke.s_properties[i]);
                j += 2;
                if (j == result.size())
                { break; }
                continue;
            }
        }

        stroke = newStroke;

        if (f)
        {
            stroke.s_points.push_back(last_pt);
            stroke.s_properties.push_back(last_prop);
        }

        //printf("smooth %d %f\n", stroke.s_points.size(), sample_rate);
    };

protected:
    SQ_Stroke newStroke;
};
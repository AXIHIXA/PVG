#include "debug.h"
#include "Strokes.h"
#include "RegionFZ.h"
#include <interpolation.h>

using namespace std;

bool SQ_Stroke::resetKeyframe(int index_i, int index_j, int side_index)
{
    // set keyframe index_i and index_j to be true, points between to be false
    if (s_points.isEmpty() || s_properties.isEmpty())
    {
        st_warn("stroke or properties is empty!");
        return false;
    }

    if (side_index == 1)  // side_1
    {
        for (int i = index_i; i <= index_j; i++)
        {
            s_properties[i].keyframe_1 = i == (index_i || i == index_j);
        }
    }
    else if (side_index == 2)  // side_2
    {
        for (int i = index_i; i <= index_j; i++)
        {
            s_properties[i].keyframe_2 = i == (index_i || i == index_j);
        }
    }
    else  // both sides
    {
        for (int i = index_i; i <= index_j; i++)
        {
            if (i == index_i || i == index_j)
            {
                s_properties[i].keyframe_1 = true;
                s_properties[i].keyframe_2 = true;
            }
            else
            {
                s_properties[i].keyframe_1 = false;
                s_properties[i].keyframe_2 = false;
            }
        }
    }
    return true;
}

void SQ_Stroke::initProperties(
        const QColor & c11, const QColor & c12,
        const QColor & c21, const QColor & c22,
        MDMode mdmode)
{
    if (s_points.isEmpty())
    {
        return;
    }

    mdmode = mdmode;
    s_properties.clear();

    for (int i = 0; i < s_points.size(); i++)
    {
        if (s_points.size() <= 1)
        {
            break;
        }

        double factor = 1.0 - (double) i / (double) (s_points.size() - 1);

        int red = round(c11.red() * factor + (c12.red() * (1.0 - factor)));
        int green = round(c11.green() * factor + (c12.green() * (1.0 - factor)));
        int blue = round(c11.blue() * factor + (c12.blue() * (1.0 - factor)));
        int alpha = round(c11.alpha() * factor + (c12.alpha() * (1.0 - factor)));

        PointProperties s_pp;
        s_pp.color_1.setRed(red);
        s_pp.color_1.setGreen(green);
        s_pp.color_1.setBlue(blue);
        s_pp.color_1.setAlpha(alpha);

        red = round(c21.red() * factor + (c22.red() * (1.0 - factor)));
        green = round(c21.green() * factor + (c22.green() * (1.0 - factor)));
        blue = round(c21.blue() * factor + (c22.blue() * (1.0 - factor)));
        alpha = round(c21.alpha() * factor + (c22.alpha() * (1.0 - factor)));

        s_pp.color_2.setRed(red);
        s_pp.color_2.setGreen(green);
        s_pp.color_2.setBlue(blue);
        s_pp.color_2.setAlpha(alpha);

        if (i == 0 || i == s_points.size() - 1)
        {
            s_pp.keyframe_1 = true;
            s_pp.keyframe_2 = true;
        }
        else
        {
            s_pp.keyframe_1 = false;
            s_pp.keyframe_2 = false;
        }

        s_properties.push_back(s_pp);
    }
}

bool SQ_Stroke::updateColor(int index_i, int index_j, int mode)
{
    if (index_i == index_j)
    {
        return true;
    }

    if (index_i > index_j)
    {
        st_error("index_error");
        return false;
    }

    if (mode == 1)  // color_1
    {
        QColor c_i = s_properties[index_i].color_1;
        QColor c_j = s_properties[index_j].color_1;

        for (int i = index_i; i <= index_j; i++)
        {
            double factor = 1.0 - (double) (i - index_i) / (double) (index_j - index_i);

            int red = round(c_i.red() * factor + (c_j.red() * (1.0 - factor)));
            int green = round(c_i.green() * factor + (c_j.green() * (1.0 - factor)));
            int blue = round(c_i.blue() * factor + (c_j.blue() * (1.0 - factor)));
            int alpha = round(c_i.alpha() * factor + (c_j.alpha() * (1.0 - factor)));

            s_properties[i].color_1.setRed(red);
            s_properties[i].color_1.setGreen(green);
            s_properties[i].color_1.setBlue(blue);
            s_properties[i].color_1.setAlpha(alpha);
        }

        return true;
    }
    else if (mode == 2)  // color_2
    {
        QColor c_i = s_properties[index_i].color_2;
        QColor c_j = s_properties[index_j].color_2;

        for (int i = index_i; i <= index_j; i++)
        {
            double factor = 1.0 - (double) (i - index_i) / (double) (index_j - index_i);

            int red = round(c_i.red() * factor + (c_j.red() * (1.0 - factor)));
            int green = round(c_i.green() * factor + (c_j.green() * (1.0 - factor)));
            int blue = round(c_i.blue() * factor + (c_j.blue() * (1.0 - factor)));
            int alpha = round(c_i.alpha() * factor + (c_j.alpha() * (1.0 - factor)));

            s_properties[i].color_2.setRed(red);
            s_properties[i].color_2.setGreen(green);
            s_properties[i].color_2.setBlue(blue);
            s_properties[i].color_2.setAlpha(alpha);
        }

        return true;
    }

    return false;  // hope not happened
}

bool SQ_Stroke::setPointColor(int index, const QColor & color, int side_index)
{
    if (index < 0 || index >= s_properties.size())
    {
        st_warn("index out of range!");
        return false;
    }

    int pre_index = 0, aft_index = s_properties.size() - 1;

    if (side_index == 1)  // color_1
    {
        s_properties[index].keyframe_1 = true;
        s_properties[index].color_1 = color;

        for (int i = index - 1; i >= 0; i--)
        {
            if (s_properties[i].keyframe_1)
            {
                pre_index = i;
                break;
            }
        }

        for (int i = index + 1; i < s_properties.size(); i++)
        {
            if (s_properties[i].keyframe_1)
            {
                aft_index = i;
                break;
            }
        }

        updateColor(pre_index, index, 1);
        updateColor(index, aft_index, 1);
    }
    else if (side_index == 2)  // color_2
    {
        s_properties[index].keyframe_2 = true;
        s_properties[index].color_2 = color;

        for (int i = index - 1; i >= 0; i--)
        {
            if (s_properties[i].keyframe_2)
            {
                pre_index = i;
                break;
            }
        }

        for (int i = index + 1; i < s_properties.size(); i++)
        {
            if (s_properties[i].keyframe_2)
            {
                aft_index = i;
                break;
            }
        }

        updateColor(pre_index, index, 2);
        updateColor(index, aft_index, 2);
    }
    else  // color_1 & color_2
    {
        s_properties[index].keyframe_1 = true;
        s_properties[index].keyframe_2 = true;
        s_properties[index].color_1 = color;
        s_properties[index].color_2 = color;
    }

    return true;
}

bool SQ_Stroke::deletePointColor(int index, int side_index)
{
    if (index < 0 || index >= s_properties.size())
    {
        st_warn("WARNING: index out of range!");
        return false;
    }

    if (index == 0 || index == s_properties.size() - 1)
    {
        st_warn("WARNING: Pole point must has a color!");
        return true;
    }

    int pre_index = 0, aft_index = s_properties.size() - 1;

    if (side_index == 1)  // color_1
    {
        s_properties[index].keyframe_1 = false;

        for (int i = index - 1; i >= 0; i--)
        {
            if (s_properties[i].keyframe_1)
            {
                pre_index = i;
                break;
            }
        }

        for (int i = index + 1; i < s_properties.size(); i++)
        {
            if (s_properties[i].keyframe_1)
            {
                aft_index = i;
                break;
            }
        }

        updateColor(pre_index, aft_index, 1);
    }
    else if (side_index == 2)  // color_2
    {
        s_properties[index].keyframe_2 = false;

        for (int i = index - 1; i >= 0; i--)
        {
            if (s_properties[i].keyframe_2)
            {
                pre_index = i;
                break;
            }
        }

        for (int i = index + 1; i < s_properties.size(); i++)
        {
            if (s_properties[i].keyframe_2)
            {
                aft_index = i;
                break;
            }
        }

        updateColor(pre_index, aft_index, 2);
    }
    else  // color_1 & color_2 i think this will never happen
    {
        s_properties[index].keyframe_1 = false;
        s_properties[index].keyframe_2 = false;
    }

    return true;
}

void SQ_Stroke::swapColor()
{
    for (int i = 0; i < s_properties.size(); ++i)
    {
        std::swap(s_properties[i].color_1, s_properties[i].color_2);
        std::swap(s_properties[i].keyframe_1, s_properties[i].keyframe_2);
    }
}

qreal SQ_Stroke::length() const
{
    qreal length = 0.0;

    for (int i = 1; i < segs.size(); i++)
    {
        QPointF p1 = segs.at(i - 1);
        QPointF p2 = segs.at(i);
        qreal dis = QLineF(p1, p2).length();
        length += dis;
    }

    return length;
}

SQ_Stroke & SQ_Stroke::operator=(const SQ_Stroke & s)
{
    this->s_mode = s.s_mode;
    this->s_mdmode = s.s_mdmode;
    this->s_points = s.s_points;
    this->s_properties = s.s_properties;
    this->dir_b = s.dir_b;
    this->dir_f = s.dir_f;
    this->tangents = s.tangents;

    return *this;
}

void SQ_Stroke::scaleVertical(double s)
{
    for (int i = 0; i < s_points.size(); ++i)
    {
        s_points[i].ry() *= s;
    }
}

void SQ_Stroke::scaleHorizontal(double s)
{
    for (int i = 0; i < s_points.size(); ++i)
    {
        s_points[i].rx() *= s;
    }
}

void SQ_Stroke::scale(double s)
{
    for (int i = 0; i < s_points.size(); ++i)
    {
        s_points[i] *= s;
    }
}

void SQ_Stroke::updateAll()
{
    int index1 = 0, index2 = 0;

    while (index1 < s_properties.size())
    {
        for (index2 = index1 + 1; index2 < s_properties.size(); ++index2)
        {
            if (s_properties[index2].keyframe_1)
            {
                updateColor(index1, index2, 1);
                break;
            }
        }

        index1 = index2;
    }

    index1 = 0;

    while (index1 < s_properties.size())
    {
        for (index2 = index1 + 1; index2 < s_properties.size(); ++index2)
        {
            if (s_properties[index2].keyframe_2)
            {
                updateColor(index1, index2, 2);
                break;
            }
        }

        index1 = index2;
    }
}

void SQ_Stroke::translation(const QPointF & t)
{
    for (int i = 0; i < s_points.size(); ++i)
    {
        s_points[i] += t;
    }
}

int SQ_Stroke::sideEndpoint(const QPointF & p, double scale) const
{
    if (s_mode == CLOSED || s_points.size() < 2)
    {
        return 0;
    }

    double dis1 = (s_points.first() - p).manhattanLength();
    double dis2 = (s_points.last() - p).manhattanLength();

    if (dis1 > 3 * scale && dis2 > 3 * scale)
    {
        return 0;
    }

    cv::Vec3d p1, p2;

    if (dis1 < dis2)
    {
        p1[0] = s_points.first().x();
        p1[1] = s_points.first().y();
        p1[2] = 1;

        p2[0] = s_points.first().x() + dir_f.x();
        p2[1] = s_points.first().y() + dir_f.y();
        p2[2] = 1;
    }
    else
    {
        p1[0] = s_points.last().x();
        p1[1] = s_points.last().y();
        p1[2] = 1;

        p2[0] = s_points.last().x() + dir_b.x();
        p2[1] = s_points.last().y() + dir_b.y();
        p2[2] = 1;
    }
    cv::Vec3d line = p1.cross(p2);
    cv::Vec3d pt(p.x(), p.y(), 1);

    return (line.dot(pt) < 0) ? -1 : 1;
}

bool SQ_Stroke::insertTangent(const QPointF & p, const cv::Vec3d & line)
{
    QPair< QPointF, cv::Vec3d > pt(p, line);

    if (find(tangents.begin(), tangents.end(), pt) == tangents.end())
    {
        tangents.push_back(pt);
        return true;
    }
    else
    {
        return false;
    }
}

void SQ_Stroke::clearTangents()
{
    tangents.clear();
}

int SQ_Stroke::sideTangent(const QPointF & p) const
{
    int id = -1;
    double dis = numeric_limits< double >::infinity();

    for (int i = 0; i < tangents.size(); ++i)
    {
        double d = (p - tangents[i].first).manhattanLength();

        if (d < dis)
        {
            id = i;
            dis = d;
        }
    }

    if (id != -1)
    {
        double dot = tangents[id].second.dot(cv::Vec3d(p.x(), p.y(), 1));
        return (dot > 0) ? 1 : -1;
    }
    else
    {
        return 0;
    }
}

SQ_Stroke linear_interpolation(const SQ_Stroke & s1, const SQ_Stroke & s2, double rate)
{
    SQ_Stroke stroke;
    stroke.s_mdmode = s1.s_mdmode;
    stroke.s_mode = s1.s_mode;

    for (int i = 0; i < s1.s_points.size(); ++i)
    {
        stroke.s_points.push_back(s1.s_points[i] + rate * (s2.s_points[i] - s1.s_points[i]));
    }

    for (int i = 0; i < s1.s_properties.size(); ++i)
    {
        double r1 = s1.s_properties[i].color_1.red() +
                    rate * (s2.s_properties[i].color_1.red() - s1.s_properties[i].color_1.red());
        double g1 = s1.s_properties[i].color_1.green() +
                    rate * (s2.s_properties[i].color_1.green() - s1.s_properties[i].color_1.green());
        double b1 = s1.s_properties[i].color_1.blue() +
                    rate * (s2.s_properties[i].color_1.blue() - s1.s_properties[i].color_1.blue());

        double r2 = s1.s_properties[i].color_2.red() +
                    rate * (s2.s_properties[i].color_2.red() - s1.s_properties[i].color_2.red());
        double g2 = s1.s_properties[i].color_2.green() +
                    rate * (s2.s_properties[i].color_2.green() - s1.s_properties[i].color_2.green());
        double b2 = s1.s_properties[i].color_2.blue() +
                    rate * (s2.s_properties[i].color_2.blue() - s1.s_properties[i].color_2.blue());

        stroke.s_properties.push_back({QColor(r1, g1, b1),QColor(r2, g2, b2),
                                       s1.s_properties[i].keyframe_1, s1.s_properties[i].keyframe_2});
    }

    return stroke;
}
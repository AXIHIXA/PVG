#ifndef PARSECOLOR_HPP
#define PARSECOLOR_HPP


#include "interpolation.h"
#include "Strokes.h"
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <tbb/parallel_for.h>


class CParseColor
{
public:
    CParseColor() = default;

    CParseColor(int h, int w, const QPoint & _w00, bool tangents)
    {
        height = h;
        width = w;
        w00 = _w00;
        with_tangents = tangents;
    }

    void parse(QVector<SQ_Stroke> & m_strokes, int boundaryColor)
    {
        using namespace alglib;

        bdColor = boundaryColor;
        newStrokes = m_strokes;

        for (int i = 0; i < m_strokes.size(); i++)
        {
            if (m_strokes[i].s_points.isEmpty())
            {
                continue;
            }

            bool isPe = m_strokes[i].s_properties.isEmpty() || (bdColor == 2);

            if (m_strokes[i].s_mode == SQ_Stroke::CLOSED)
            {
                m_strokes[i].s_points.pop_back();
                m_strokes[i].s_points.push_back(m_strokes[i].s_points.at(0));
            }

            clear();

            if (isPe)
            {
                moveTo(m_strokes[i].s_points.at(0), 0, isPe, PointProperties());
            }
            else
            {
                moveTo(m_strokes[i].s_points.at(0), 0, isPe, m_strokes[i].s_properties.at(0));
            }

            if (m_strokes[i].s_points.size() >= 2)
            {
                pspline2interpolant p;
                cubicTo(m_strokes[i], p, 2);//parabolically terminated cubic spline

                if (boundaryColor == 0)
                {
                    SQ_Stroke newStroke;
                    convertToDiffusionCurve(m_strokes[i], newStroke, p);
                    newStrokes[i] = newStroke;
                }
            }

            m_strokes[i].segs = segs;
            m_strokes[i].pps = pps;
            m_strokes[i].idx = idx;
        }
    }

    inline QColor lerpColor(const QColor & c1, const QColor & c2, double e)
    {
        return QColor(
                (int) ((double) c2.red() * e + (double) c1.red() * (1 - e)),
                (int) ((double) c2.green() * e + (double) c1.green() * (1 - e)),
                (int) ((double) c2.blue() * e + (double) c1.blue() * (1 - e)),
                (int) ((double) c2.alpha() * e + (double) c1.alpha() * (1 - e))
        );
    }

private:
    void clear()
    {
        segs.clear();
        pps.clear();
        idx.clear();
    }

    void moveTo(const QPointF & p, int id, bool isPropertyEmpty, const PointProperties & pp)
    {
        if (segs.empty() || segs.back() != p)
        {
            segs.push_back(p);
        }

        idx.push_back(id);

        if (!isPropertyEmpty)
        {
            pps.push_back(pp);
        }
    }

    void lineTo(const QPointF & p, int id, bool isPropertyEmpty, const PointProperties & pp)
    {
        moveTo(p, id, isPropertyEmpty, pp);
    }

    bool maxAngle(const QPointF & lp, const QPointF & mp, const QPointF & rp)
    {
        if (bdColor == 1)
        {
            return QLineF(mp, lp).length() + QLineF(mp, rp).length() < 3;
        }

        double a = acos(QPointF::dotProduct(lp - mp, rp - mp) / (QLineF(mp, lp).length() * QLineF(mp, rp).length()));

        bool b = (QLineF(mp, lp).length() + QLineF(mp, rp).length() < 10 && a > CV_PI * 0.95) ||
                 (QLineF(mp, lp).length() + QLineF(mp, rp).length() < 2);

        return b;
    }

    inline bool in(int y, int x)
    {
        return (x >= 0) && (x < width) && (y >= 0) && (y < height);
    }

    bool intersect_segs(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4)
    {
        double x0, y0;

        double d = (y2 - y1) * (x4 - x3) - (y4 - y3) * (x2 - x1);

        if (d != 0)
        {
            x0 = ((x2 - x1) * (x4 - x3) * (y3 - y1) + (y2 - y1) * (x4 - x3) * x1 - (y4 - y3) * (x2 - x1) * x3) / d;
            y0 = ((y2 - y1) * (y4 - y3) * (x3 - x1) + (x2 - x1) * (y4 - y3) * y1 - (x4 - x3) * (y2 - y1) * y3) / (-d);

            if ((x0 - x1) * (x0 - x2) <= 0 &&
                (x0 - x3) * (x0 - x4) <= 0 &&
                (y0 - y1) * (y0 - y2) <= 0 &&
                (y0 - y3) * (y0 - y4) <= 0)
            {
                return true;
            }
        }

        return false;
    }

    bool intersect(
            const QPointF & lp,
            const QPointF & rp,
            const QPointF & w00,
            const alglib::real_1d_array & t,
            double lt,
            double rt,
            const SQ_Stroke & stroke)
    {
        int N = t.length(), idx_l, idx_r;

        if (lt > t[N - 1])
        {
            idx_l = N - 1;
        }
        else
        {
            for (int i = 0; i < N - 1; i++)
            {
                if (t[i] < lt + 1e-6 && lt < t[i + 1] + 1e-6)
                {
                    idx_l = i;
                    break;
                }
            }
        }

        if (rt > t[N - 1])
        {
            idx_r = N - 1;
        }
        else
        {
            for (int i = 0; i < N - 1; i++)
            {
                if (t[i] < rt + 1e-6 && rt < t[i + 1] + 1e-6)
                {
                    idx_r = i + 1;
                    break;
                }
            }
        }

        for (int i = idx_l; i < idx_r; i++)
        {
            double x1 = (double) (stroke.s_points[i].x() - w00.x());
            double y1 = (double) (stroke.s_points[i].y() - w00.y());
            double x2 = (double) (stroke.s_points[i + 1].x() - w00.x());
            double y2 = (double) (stroke.s_points[i + 1].y() - w00.y());

            double x3 = 0, y3 = 0, x4 = 0, y4 = height - 1;

            bool a = intersect_segs(x1, y1, x2, y2, x3, y3, x4, y4);
            x3 = 0, y3 = 0, x4 = width - 1, y4 = 0;

            bool b = intersect_segs(x1, y1, x2, y2, x3, y3, x4, y4);
            x3 = width - 1, y3 = height - 1, x4 = width - 1, y4 = 0;

            bool c = intersect_segs(x1, y1, x2, y2, x3, y3, x4, y4);
            x3 = width - 1, y3 = height - 1, x4 = 0, y4 = height - 1;

            bool d = intersect_segs(x1, y1, x2, y2, x3, y3, x4, y4);

            if ((a || b || c || d))
            {
                return true;
            }
        }

        return false;
    }

    inline void solve_cubic(const cv::Vec4d & coef, cv::Vec3d & res)
    {
        if (abs(coef[0]) > 1e-6)
        {
            cv::solveCubic(coef, res);
        }
        else if (abs(coef[1]) > 1e-6)
        {
            double delta = coef[2] * coef[2] - 4 * coef[1] * coef[3];

            if (delta >= 0)
            {
                delta = sqrt(delta);
                res[0] = (-coef[2] + delta) / (2 * coef[1]);
                res[1] = (-coef[2] - delta) / (2 * coef[1]);
            }
        }
        else if (abs(coef[2]) > 1e-6)
        {
            res[0] = -coef[3] / coef[2];
        }
    }

    //solve extrema points   ADDED BY HOU FEI
    bool solve_extremal_points(const cv::Vec4d & coefs, std::pair<double, double> & t)
    {
        cv::Vec3d quadratic_coefs(3 * coefs[0], 2 * coefs[1], coefs[2]);

        if (abs(quadratic_coefs[0]) > 1e-6)
        {
            double delta = quadratic_coefs[1] * quadratic_coefs[1] - 4 * quadratic_coefs[0] * quadratic_coefs[2];

            if (delta >= 0)
            {
                delta = sqrt(delta);
                t.first = (-quadratic_coefs[1] + delta) / (2 * quadratic_coefs[0]);
                t.second = (-quadratic_coefs[1] - delta) / (2 * quadratic_coefs[0]);
                return true;
            }
            else
            {
                return false;
            }
        }
        else if (abs(quadratic_coefs[1]) > 1e-6)
        {
            t.first = t.second = -quadratic_coefs[2] / quadratic_coefs[1];
            return true;
        }
        else
        {
            return false;
        }
    }

    inline double eval_cubic(const cv::Vec4d & v, double t)
    {
        return v[0] * t * t * t + v[1] * t * t + v[2] * t + v[3];
    }

    bool intersect_cubic(QPointF lp, QPointF rp, QPointF w00, const alglib::real_1d_array & t, double lt, double rt,
                         SQ_Stroke & stroke, const alglib::pspline2interpolant & psi)
    {
        int N = t.length(), idx_l, idx_r;

        if (lt > t[N - 1])
        {
            idx_l = N - 1;
        }
        else
        {
            for (int i = 0; i < N - 1; i++)
            {
                if (t[i] < lt + 1e-6 && lt < t[i + 1] + 1e-6)
                {
                    idx_l = i;
                    break;
                }
            }
        }

        if (rt > t[N - 1])
        {
            if (stroke.s_mode == SQ_Stroke::CLOSED)
            {
                idx_r = N;
            }
            else
            {
                idx_r = N - 1;
            }
        }
        else
        {
            for (int i = 0; i < N - 1; i++)
            {
                if (t[i] < rt + 1e-6 && rt < t[i + 1] + 1e-6)
                {
                    idx_r = i + 1;
                    break;
                }
            }
        }

        double p, q, tt;
        QPointF a, b, c, d, p0, p1, p2, p3, d0, d1, d00;

        for (int i = idx_l; i < idx_r; i++)
        {
            p = t[i];
            pspline2diff2(psi, p, p0.rx(), d0.rx(), d00.rx(), p0.ry(), d0.ry(), d00.ry());

            if (stroke.s_mode == SQ_Stroke::CLOSED && (i + 1 == stroke.s_points.size() - 1))
            {
                q = 1;
            }
            else
            {
                q = t[i + 1];
            }

            pspline2diff(psi, q, p3.rx(), d1.rx(), p3.ry(), d1.ry());

            Eigen::Matrix4d vA;
            Eigen::Vector4d vB, vXY;
            vA << 1, p, p * p, p * p * p, 1, q, q * q, q * q * q, 0, 1, 2 * p, 3 * p * p, 0, 1, 2 * q, 3 * q * q;
            vB << p0.x(), p3.x(), d0.x(), d1.x();
            vXY = vA.colPivHouseholderQr().solve(vB);
            a.rx() = vXY[0], b.rx() = vXY[1], c.rx() = vXY[2], d.rx() = vXY[3];
            vB << p0.y(), p3.y(), d0.y(), d1.y();
            vXY = vA.colPivHouseholderQr().solve(vB);
            a.ry() = vXY[0], b.ry() = vXY[1], c.ry() = vXY[2], d.ry() = vXY[3];

            cv::Vec3d roots_x0(-1, -1, -1), roots_x1(-1, -1, -1), roots_y0(-1, -1, -1), roots_y1(-1, -1, -1);
            double ll = lt > p ? lt : p, rr = rt < q ? rt : q;
            ll -= 1e-5;
            rr += 1e-5;

            cv::Vec4d coeffs_x0(d.x(), c.x(), b.x(), a.x() - w00.x());
            solve_cubic(coeffs_x0, roots_x0);

            for (int j = 0; j < 3; j++)
            {
                tt = roots_x0[j];

                if (tt == -1)
                {
                    continue;
                }

                if (tt >= ll && tt <= rr)
                {
                    double x, y;
                    pspline2calc(psi, tt, x, y);

                    if (y >= w00.y() && y <= w00.y() + height && abs(x - w00.x()) < 1e-5)
                    {
                        return true;
                    }
                }
            }

            cv::Vec4d coeffs_x1(d.x(), c.x(), b.x(), a.x() - w00.x() - width + 1);
            solve_cubic(coeffs_x1, roots_x1);

            for (int j = 0; j < 3; j++)
            {
                tt = roots_x1[j];

                if (tt == -1)
                {
                    continue;
                }

                if (tt >= ll && tt <= rr)
                {
                    double x, y;
                    pspline2calc(psi, tt, x, y);

                    if (y >= w00.y() && y <= w00.y() + height && abs(x - w00.x() - width + 1) < 1e-5)
                    {
                        return true;
                    }
                }
            }

            cv::Vec4d coeffs_y0(d.y(), c.y(), b.y(), a.y() - w00.y());
            solve_cubic(coeffs_y0, roots_y0);

            for (int j = 0; j < 3; j++)
            {
                tt = roots_y0[j];

                if (tt == -1)
                {
                    continue;
                }

                if (tt >= ll && tt <= rr)
                {
                    double x, y;
                    pspline2calc(psi, tt, x, y);

                    if (x >= w00.x() && x <= w00.x() + width && abs(y - w00.y()) < 1e-5)
                    {
                        return true;
                    }
                }
            }

            cv::Vec4d coeffs_y1(d.y(), c.y(), b.y(), a.y() - w00.y() - height + 1);
            solve_cubic(coeffs_y1, roots_y1);

            for (int j = 0; j < 3; j++)
            {
                tt = roots_y1[j];

                if (tt == -1)
                {
                    continue;
                }

                if (tt >= ll && tt <= rr)
                {
                    double x, y;
                    pspline2calc(psi, tt, x, y);

                    if (x >= w00.x() && x <= w00.x() + width && abs(y - w00.y() - height + 1) < 1e-5)
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    void evaluate_tangents(SQ_Stroke & stroke, const alglib::pspline2interpolant & psi, const alglib::real_1d_array & t)
    {
        int N = t.length(), idx_l = 0, idx_r;

        const double lt = 0, rt = 1;

        if (stroke.s_mode == SQ_Stroke::CLOSED)
        {
            idx_r = N;
        }
        else
        {
            idx_r = N - 1;
        }

        QVector<QVector<QPair<QPointF, cv::Vec3d >>> key_points(idx_r);

        tbb::parallel_for(idx_l, idx_r, [&](int i)
        {
            double p, q;
            QPointF a, b, c, d, p0, p1, p2, p3, d0, d1, d00;
            p = t[i];
            pspline2diff2(psi, p, p0.rx(), d0.rx(), d00.rx(), p0.ry(), d0.ry(), d00.ry());
            if (stroke.s_mode == SQ_Stroke::CLOSED && (i + 1 == stroke.s_points.size() - 1))
            {
                q = 1;
            }
            else
            {
                q = t[i + 1];
            }
            pspline2diff(psi, q, p3.rx(), d1.rx(), p3.ry(), d1.ry());

            Eigen::Matrix4d vA;
            Eigen::Vector4d vB, vXY;
            vA << 1, p, p * p, p * p * p, 1, q, q * q, q * q * q, 0, 1, 2 * p, 3 * p * p, 0, 1, 2 * q, 3 * q * q;
            vB << p0.x(), p3.x(), d0.x(), d1.x();
            vXY = vA.colPivHouseholderQr().solve(vB);
            a.rx() = vXY[0], b.rx() = vXY[1], c.rx() = vXY[2], d.rx() = vXY[3];
            vB << p0.y(), p3.y(), d0.y(), d1.y();
            vXY = vA.colPivHouseholderQr().solve(vB);
            a.ry() = vXY[0], b.ry() = vXY[1], c.ry() = vXY[2], d.ry() = vXY[3];

            //evaluate extremal points
            //vertical lines
            //pair<double, double> tangent_t;
            //cv::Vec4d x_curve(d.x(), c.x(), b.x(), a.x());
            //cv::Vec4d y_curve(d.y(), c.y(), b.y(), a.y());
            //bool valid = solve_extremal_points(x_curve, tangent_t);
            //if (valid)
            //{
            //	if (0 <= tangent_t.first && tangent_t.first <= 1)
            //	{
            //		if (p<tangent_t.first && q>tangent_t.first)
            //		{
            //			double x, y, dx, dy;
            //			alglib::pspline2diff(psi, tangent_t.first, x, dx, y, dy);
            //			x -= w00.x();
            //			y -= w00.y();
            //			cv::Vec3d v(cv::Vec3d(x, y, 1).cross(cv::Vec3d(x + dx, y + dy, 1)));
            //			stroke.insertTangent(QPointF(x, y), v);
            //		}
            //	}
            //	if (tangent_t.first != tangent_t.second && 0 <= tangent_t.second && tangent_t.second <= 1)
            //	{
            //		if (p<tangent_t.second && q>tangent_t.second)
            //		{
            //			double x, y, dx, dy;
            //			alglib::pspline2diff(psi, tangent_t.second, x, dx, y, dy);
            //			x -= w00.x();
            //			y -= w00.y();
            //			cv::Vec3d v(cv::Vec3d(x, y, 1).cross(cv::Vec3d(x + dx, y + dy, 1)));
            //			stroke.insertTangent(QPointF(x, y), v);
            //		}
            //	}
            //}
            ////horizontal lines
            //valid = solve_extremal_points(y_curve, tangent_t);
            //if (valid)
            //{
            //	if (0 <= tangent_t.first && tangent_t.first <= 1)
            //	{
            //		if (p<tangent_t.first && q>tangent_t.first)
            //		{
            //			double x, y, dx, dy;
            //			alglib::pspline2diff(psi, tangent_t.first, x, dx, y, dy);
            //			x -= w00.x();
            //			y -= w00.y();
            //			cv::Vec3d v(cv::Vec3d(x, y, 1).cross(cv::Vec3d(x + dx, y + dy, 1)));
            //			stroke.insertTangent(QPointF(x, y), v);
            //		}
            //	}
            //	if (tangent_t.first != tangent_t.second && 0 <= tangent_t.second && tangent_t.second <= 1)
            //	{
            //		if (p<tangent_t.second && q>tangent_t.second)
            //		{
            //			double x, y, dx, dy;
            //			alglib::pspline2diff(psi, tangent_t.second, x, dx, y, dy);
            //			x -= w00.x();
            //			y -= w00.y();
            //			cv::Vec3d v(cv::Vec3d(x, y, 1).cross(cv::Vec3d(x + dx, y + dy, 1)));
            //			stroke.insertTangent(QPointF(x, y), v);
            //		}
            //	}
            //}

            //-------------------------

            cv::Vec3d roots_x0(-1, -1, -1), roots_x1(-1, -1, -1), roots_y0(-1, -1, -1), roots_y1(-1, -1, -1);
            double ll = lt > p ? lt : p, rr = rt < q ? rt : q;
            ll -= 1e-5;
            rr += 1e-5;

            cv::Vec4d coeffs_x0(d.x(), c.x(), b.x(), a.x() - w00.x());
            solve_cubic(coeffs_x0, roots_x0);
            for (int j = 0; j < 3; j++)
            {
                double tt = roots_x0[j];
                if (tt == -1)
                {
                    continue;
                }
                if (tt >= ll && tt <= rr)
                {
                    double x, y, dx, dy;
                    pspline2diff(psi, tt, x, dx, y, dy);
                    if (y >= w00.y() && y <= w00.y() + height && abs(x - w00.x()) < 1e-5)
                    {
                        x -= w00.x();
                        y -= w00.y();
                        cv::Vec3d v(cv::Vec3d(x, y, 1).cross(cv::Vec3d(x + dx, y + dy, 1)));
                        key_points[i].push_back(qMakePair(QPointF(x, y), v));
                    }
                }
            }

            cv::Vec4d coeffs_x1(d.x(), c.x(), b.x(), a.x() - w00.x() - width + 1);
            solve_cubic(coeffs_x1, roots_x1);

            for (int j = 0; j < 3; j++)
            {
                double tt = roots_x1[j];
                if (tt == -1)
                {
                    continue;
                }
                if (tt >= ll && tt <= rr)
                {
                    double x, y, dx, dy;
                    pspline2diff(psi, tt, x, dx, y, dy);
                    if (y >= w00.y() && y <= w00.y() + height && abs(x - w00.x() - width + 1) < 1e-5)
                    {
                        x -= w00.x();
                        y -= w00.y();
                        cv::Vec3d v(cv::Vec3d(x, y, 1).cross(cv::Vec3d(x + dx, y + dy, 1)));
                        key_points[i].push_back(qMakePair(QPointF(x, y), v));
                    }
                }
            }

            cv::Vec4d coeffs_y0(d.y(), c.y(), b.y(), a.y() - w00.y());
            solve_cubic(coeffs_y0, roots_y0);
            for (int j = 0; j < 3; j++)
            {
                double tt = roots_y0[j];
                if (tt == -1)
                {
                    continue;
                }
                if (tt >= ll && tt <= rr)
                {
                    double x, y, dx, dy;
                    pspline2diff(psi, tt, x, dx, y, dy);
                    if (x >= w00.x() && x <= w00.x() + width && abs(y - w00.y()) < 1e-5)
                    {
                        x -= w00.x();
                        y -= w00.y();
                        cv::Vec3d v(cv::Vec3d(x, y, 1).cross(cv::Vec3d(x + dx, y + dy, 1)));
                        key_points[i].push_back(qMakePair(QPointF(x, y), v));
                    }
                }
            }

            cv::Vec4d coeffs_y1(d.y(), c.y(), b.y(), a.y() - w00.y() - height + 1);
            solve_cubic(coeffs_y1, roots_y1);
            for (int j = 0; j < 3; j++)
            {
                double tt = roots_y1[j];
                if (tt == -1)
                {
                    continue;
                }
                if (tt >= ll && tt <= rr)
                {
                    double x, y, dx, dy;
                    pspline2diff(psi, tt, x, dx, y, dy);
                    if (x >= w00.x() && x <= w00.x() + width && abs(y - w00.y() - height + 1) < 1e-5)
                    {
                        x -= w00.x();
                        y -= w00.y();
                        cv::Vec3d v(cv::Vec3d(x, y, 1).cross(cv::Vec3d(x + dx, y + dy, 1)));
                        key_points[i].push_back(qMakePair(QPointF(x, y), v));
                    }
                }
            }
        });
        for (int i = 0; i < key_points.size(); ++i)
        {
            for (int j = 0; j < key_points[i].size(); ++j)
            {
                stroke.insertTangent(key_points[i][j].first, key_points[i][j].second);
            }
        }
    }

    bool intersect(const QPointF & lp, const QPointF & rp, const QPointF & w00)
    {
        double x1 = (double) (lp.x() - w00.x()), y1 = (double) (lp.y() - w00.y());
        double x2 = (double) (rp.x() - w00.x()), y2 = (double) (rp.y() - w00.y());

        double x3 = 0, y3 = 0, x4 = 0, y4 = height - 1;
        bool a = intersect_segs(x1, y1, x2, y2, x3, y3, x4, y4);
        x3 = 0, y3 = 0, x4 = width - 1, y4 = 0;
        bool b = intersect_segs(x1, y1, x2, y2, x3, y3, x4, y4);
        x3 = width - 1, y3 = height - 1, x4 = width - 1, y4 = 0;
        bool c = intersect_segs(x1, y1, x2, y2, x3, y3, x4, y4);
        x3 = width - 1, y3 = height - 1, x4 = 0, y4 = height - 1;
        bool d = intersect_segs(x1, y1, x2, y2, x3, y3, x4, y4);
        return ((a || b || c || d));
    }

    //============================================================
    void
    plotCubic(const alglib::pspline2interpolant & p, const alglib::real_1d_array & t, SQ_Stroke & stroke, double lt,
              double rt, const QPointF & lp, const QPointF & rp, SQ_Stroke::StrokeMode pathMode)
    {
        using namespace alglib;

        double x, y, mt = (lt + rt) / 2;
        pspline2calc(p, mt, x, y);
        QPointF mp(x, y);
        bool a = maxAngle(lp, mp, rp);

        bool b = (!in(lp.y() - w00.y(), lp.x() - w00.x()) && !in(rp.y() - w00.y(), rp.x() - w00.x()) &&
                  !in(mp.y() - w00.y(), mp.x() - w00.x()));
        if (b)
        {
            b = (!intersect_cubic(lp, rp, w00, t, lt, rt, stroke, p));
        }
        if (b)
        {
            b = (!intersect(lp, rp, w00));
        }

        if (a || b)
        {
            int idx;
            PointProperties pp;
            bool isPe = stroke.s_properties.isEmpty();

            calcColor(pp, t, rt, stroke.s_points, stroke.s_properties, idx, pathMode, isPe);
            lineTo(rp, idx, isPe, pp);
            return;
        }
        else
        {
            plotCubic(p, t, stroke, lt, mt, lp, mp, pathMode);
            plotCubic(p, t, stroke, mt, rt, mp, rp, pathMode);
        }
    }

    void cubicTo(
            SQ_Stroke & stroke,
            alglib::pspline2interpolant & p,
            int st = 2)
    {
        using namespace alglib;

        real_2d_array xy;
        ae_int_t N = stroke.s_points.size();
        SQ_Stroke::StrokeMode pathMode = stroke.s_mode;
        real_1d_array t;
        if (stroke.s_points.size() > 3 && stroke.s_mode == SQ_Stroke::CLOSED)
        {
            ae_int_t tN = N - 1;
            xy.setlength(tN, 2);
            for (int i = 0; i < tN; i++)
            {
                xy[i][0] = stroke.s_points[i].x(), xy[i][1] = stroke.s_points[i].y();
            }
            pspline2buildperiodic(xy, tN, st, 1, p);
            pspline2parametervalues(p, tN, t);
        }
        else
        {
            xy.setlength(N, 2);
            for (int i = 0; i < N; i++)
            {
                xy[i][0] = stroke.s_points[i].x(), xy[i][1] = stroke.s_points[i].y();
            }
            pspline2build(xy, N, st, 1, p);
            pspline2parametervalues(p, N, t);
        }

        for (int i = 1; i < t.length(); ++i)
        {
            plotCubic(p, t, stroke, t(i - 1), t(i), stroke.s_points[i - 1], stroke.s_points[i], pathMode);
        }
        if (pathMode == SQ_Stroke::CLOSED)
        {
            plotCubic(p, t, stroke, t(N - 2), 1, stroke.s_points[N - 2], stroke.s_points[N - 1], pathMode);
        }

        //added by HF
        if (with_tangents && bdColor != 2)
        {
            double x, y;
            pspline2tangent(p, 0, x, y);
            stroke.dir_f = QPointF(x, y);
            pspline2tangent(p, 1, x, y);
            stroke.dir_b = QPointF(x, y);
            stroke.clearTangents();
            evaluate_tangents(stroke, p, t);
        }
    }


    void calcColor(
            PointProperties & s_pp, const alglib::real_1d_array & t, double pos, const QVector<QPointF> & ctrlPts,
            const QVector<PointProperties> & ppCtrl, int & idx, SQ_Stroke::StrokeMode pathMode, bool isPe)
    {
        int N = t.length();
        if (ppCtrl.size() == 0 || bdColor == 2)
        {
            if (pos > t[N - 1])
            {
                idx = N - 1;
            }
            else
            {
                for (int i = 0; i < N - 1; i++)
                {
                    if (t[i] < pos + 1e-6 && pos < t[i + 1] + 1e-6)
                    {
                        idx = i;
                        break;
                    }
                }
            }
        }
        else if (bdColor != 2)
        {
            if (pos > t[N - 1])
            {
                double e = fabs((pos - t[N - 1]) / (1 - t[N - 1]));
                QColor c1 = lerpColor(ppCtrl[N - 1].color_1, ppCtrl[N].color_1, e);
                QColor c2 = lerpColor(ppCtrl[N - 1].color_2, ppCtrl[N].color_2, e);
                s_pp.color_1.setRgb(c1.red(), c1.green(), c1.blue(), c1.alpha());
                s_pp.color_2.setRgb(c2.red(), c2.green(), c2.blue(), c2.alpha());
                idx = N - 1;
            }
            else
            {
                for (int i = 0; i < N - 1; i++)
                {
                    if (t[i] < pos + 1e-6 && pos < t[i + 1] + 1e-6)
                    {
                        double e = fabs((pos - t[i]) / (t[i + 1] - t[i]));
                        QColor c1 = lerpColor(ppCtrl[i].color_1, ppCtrl[i + 1].color_1, e);
                        QColor c2 = lerpColor(ppCtrl[i].color_2, ppCtrl[i + 1].color_2, e);
                        s_pp.color_1.setRgb(c1.red(), c1.green(), c1.blue(), c1.alpha());
                        s_pp.color_2.setRgb(c2.red(), c2.green(), c2.blue(), c2.alpha());
                        idx = i;
                        break;
                    }
                }
            }
        }
    }

    inline QColor diffColor(const QColor & c1, const QColor & c2)
    {
        return QColor((c2.red() - c1.red()), (c2.green() - c1.green()), (c2.blue() - c1.blue()),
                      (c2.alpha() - c1.alpha()));
    }

    inline QColor multColor(const QColor & c, const double e)
    {
        return QColor((int) e * (double) c.red(), (int) e * (double) c.green(), (int) e * (double) c.blue(),
                      (int) e * (double) c.alpha());
    }

    inline QColor multDiffColor(const QColor & c1, const QColor & c2, double e)
    {
        return QColor(
                (int) (e * (double) (c2.red() - c1.red())),
                (int) (e * (double) (c2.green() - c1.green())),
                (int) (e * (double) (c2.blue() - c1.blue())),
                (int) (e * (double) (c2.alpha() - c1.alpha()))
        );
    }

    inline QColor addColor(const QColor & c1, const QColor & c2)
    {
        return QColor(c2.red() + c1.red(), c2.green() + c1.green(), c2.blue() + c1.blue(), c2.alpha() + c1.alpha());
    }

    void convertToDiffusionCurve(SQ_Stroke & stroke, SQ_Stroke & newStroke, alglib::pspline2interpolant & psi)
    {
        using namespace alglib;
        double x, y, tl = 0.0, nl = 0.0, p, q;
        QPointF a, b, c, d, p0, p1, p2, p3, d0, d1, d00, na, nb, nc, nd;
        ae_int_t n;
        real_1d_array t;
        pspline2parametervalues(psi, n, t);
        for (int i = 1; i < stroke.s_points.size(); i++)
        {
            tl += QLineF(stroke.s_points.at(i - 1), stroke.s_points.at(i)).length();
        }
        if (!stroke.s_points.isEmpty())
        {
            newStroke.s_points.push_back(stroke.s_points.at(0));
        }
        if (!stroke.s_properties.isEmpty())
        {
            newStroke.s_properties.push_back(stroke.s_properties.at(0));
        }
        for (int i = 1; i < stroke.s_points.size(); i++)
        {
            nl += QLineF(stroke.s_points.at(i - 1), stroke.s_points.at(i)).length();
            pspline2diff2(psi, t[i - 1], p0.rx(), d0.rx(), d00.rx(), p0.ry(), d0.ry(), d00.ry());
            p = t[i - 1];
            if (stroke.s_mode == SQ_Stroke::CLOSED && i == stroke.s_points.size() - 1)
            {
                pspline2diff(psi, t[0], p3.rx(), d1.rx(), p3.ry(), d1.ry());
                q = 1;
            }
            else
            {
                pspline2diff(psi, t[i], p3.rx(), d1.rx(), p3.ry(), d1.ry());
                q = t[i];
            }
            nb = (d00 * p * p * q + d1 * p * p - d00 * p * q * q - 2 * d0 * p * q + d0 * q * q) /
                 (p * p - 2 * p * q + q * q);
            nc = (2 * d0 * p - 2 * d1 * p - d00 * p * p + d00 * q * q) / (2 * (p * p - 2 * p * q + q * q));
            nd = -(d0 - d1 - d00 * p + d00 * q) / (3 * (p * p - 2 * p * q + q * q));
            b = nb * q - nb * p - 2 * nc * p * p - 3 * nd * p * p * p + 3 * nd * p * p * q + 2 * nc * p * q;
            c = 3 * nd * p * p * p - 6 * nd * p * p * q + nc * p * p + 3 * nd * p * q * q - 2 * nc * p * q + nc * q * q;
            d = -nd * p * p * p + 3 * nd * p * p * q - 3 * nd * p * q * q + nd * q * q * q;
            p1 = (b + 3 * p0) / 3;
            p2 = (c - 3 * p0 + 6 * p1) / 3;
            newStroke.s_points.push_back(p1);
            newStroke.s_points.push_back(p2);
            newStroke.s_points.push_back(p3);
            newStroke.s_properties.push_back(stroke.s_properties.at(i));
        }
    }

private:

    QVector<QPointF> segs;
    QVector<PointProperties> pps;
    QVector<int> idx;
    QVector<SQ_Stroke> newStrokes;

    int bdColor;
    int height, width;
    QPoint w00;
    bool with_tangents;
};

#endif

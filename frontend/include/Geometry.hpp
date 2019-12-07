#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP

#include <cmath>
#include <debug.h>

class CGeometry
{
public:
    CGeometry()  = default;

    ~CGeometry() = default;

    static bool lessThan(float a, float b)
    {
        return b - a > FLIMITS;
    }

    static bool equalTo(float a, float b)
    {
        return abs(b - a) <= FLIMITS;
    }

    static bool lessThan0(float a)
    {
        return a < -FLIMITS;
    }

    static bool greaterThan0(float a)
    {
        return a > FLIMITS;
    }

    static bool equalTo0(float a)
    {
        return abs(a) <= FLIMITS;
    }

    static bool equalTo(cv::Vec3f a, cv::Vec3i b)
    {
        bool f = true;

        for (int i = 0; i < 3; i++)
        {
            f &= abs(a[i] - b[i]) < FLIMITS;
        }
    }

    static inline cv::Vec2f normVleft(int y0, int y1, int x0, int x1)
    {
        double a = y1 - y0;
        double b = x1 - x0;
        a /= sqrt(a * a + b * b);
        b /= sqrt(a * a + b * b);
        return cv::Vec2f(-b, a);
    }

    static inline cv::Vec2f normVleft(double y0, double y1, double x0, double x1)
    {
        double a = y1 - y0;
        double b = x1 - x0;
        a /= sqrt(a * a + b * b);
        b /= sqrt(a * a + b * b);
        return cv::Vec2f(-b, a);
    }

    static inline int getQuadrant(cv::Vec2f vc)
    {
        if (greaterThan0(vc[0]) && greaterThan0(vc[1]))
        {
            return 3;
        }
        if (greaterThan0(vc[0]) && lessThan0(vc[1]))
        {
            return 4;
        }
        if (lessThan0(vc[0]) && greaterThan0(vc[1]))
        {
            return 2;
        }
        if (lessThan0(vc[0]) && lessThan0(vc[1]))
        {
            return 1;
        }
        if (equalTo0(vc[0]) && greaterThan0(vc[1]))
        {
            return 7;
        }
        if (equalTo0(vc[0]) && lessThan0(vc[1]))
        {
            return 5;
        }
        if (greaterThan0(vc[0]) && equalTo0(vc[1]))
        {
            return 8;
        }
        if (lessThan0(vc[0]) && equalTo0(vc[1]))
        {
            return 6;
        }
        else
        {
            // hope not occured
            st_warn("unknown quadrant");
            return -1;
        }
    }

    static inline double calcDist(cv::Vec2f p, cv::Vec2f s0, cv::Vec2f s1)
    {
        double x0 = p[0], y0 = p[1], x1 = s0[0], y1 = s0[1], x2 = s1[0], y2 = s1[1];
        return fabs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) /
               sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    }

    static inline double calcAngle(cv::Vec2f a, cv::Vec2f b)
    {
        return acos(
                fabs(a[0] * b[0] + a[1] * b[1]) / sqrt(a[0] * a[0] + a[1] * a[1]) / sqrt(b[0] * b[0] + b[1] * b[1]));
    }

private:
    static constexpr double FLIMITS = 1e-6;
};

#endif

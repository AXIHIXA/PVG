#include "RegionFZ.h"
#include "Geometry.hpp"
#include "ParseColor.hpp"
#include "Strokes.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <queue>
#include <time.h>

using namespace std;
using namespace cv;

CRegionFZ::CRegionFZ(int h, int w, const QVector<SQ_Stroke> & strokes, double _scaleFactor, const cv::Vec2i & _w00)
        : m_strokes(strokes)
{
    wh = h, ww = w;
    height = h, width = w;
    w00 = _w00;
    scaleFactor = _scaleFactor;
    stroke_ID = -1;
    init();
}

void CRegionFZ::init()
{
    nodeMask = Mat::zeros(wh, ww, CV_8UC1);
    nm1 = Mat::zeros(wh, ww, CV_8UC1);

    sideMask = Mat::zeros(wh, ww, CV_32SC1);
    colorMask = cv::Mat::zeros(wh, ww, CV_32FC3);
    regionMask = cv::Mat::zeros(wh, ww, CV_32SC1);
    boundaryMask = cv::Mat::zeros(wh, ww, CV_32SC1);
    segmentMask = cv::Mat::zeros(wh, ww, CV_32SC1);
    segmentColorVec.clear();
}


// d: 'err'. always ZERO
// num: segmentColorVec.size() <- +1 for each seg in this stroke
// cd1, cd2: unit "normal"s of this seg
// start: 0 for 1st seg in stroke, 1 for else
// end: 1 for (seg.size - 2)-th seg, 0 for else
// ls: last side. is 0 when: ... -> plotLine -> LabelBoundry VV
// clr: false for plotLastStroke lap region, true for boundary/lap edge. passed into labelBoundry(). "color"
// li1, li2: set to -1 for this stroke (NOT this seg), may update elsewhere? 

// LabelBoundary(y0, x0, lbx, lby, d, segmentColorVec.size(), cd1, cd2, start, end, ls, clr, li1, li2)
//                y  x   lbx  lby  err num                    cd1  cd2  st     end  &lastside clr &li1 &li2    
void CRegionFZ::LabelBoundary(
        int y, int x, int lby, int lbx,
        int err, int num,
        Vec2f cd1, Vec2f cd2, bool st, bool end, bool & lastSide, bool clr, int & li1, int & li2)
{
    nodeMask.at<bool>(y, x) = true;

    int i1, i2;
    int mode = segmentColorVec[num - 1].mode;  // mono = 0, dual = 1

    ///
    /// boundary mask
    ///

    if (mode == 1)
    {
        //if (1) {
        if (lby * lbx > 0)
        {
            if (lby > 0)
            {
                if (err < 0)
                {
                    boundaryMask.at<int>(y, x) = 2;
                }
                else if (err > 0)
                {
                    boundaryMask.at<int>(y, x) = 4;
                }
                else
                {
                    if (lastSide == 0)
                    {
                        boundaryMask.at<int>(y, x) = CGeometry::getQuadrant(cd1);
                    }
                    else
                    {
                        boundaryMask.at<int>(y, x) = CGeometry::getQuadrant(cd2);
                    }
                }
            }
            else
            {
                if (err < 0)
                {
                    boundaryMask.at<int>(y, x) = 4;

                }
                else if (err > 0)
                {
                    boundaryMask.at<int>(y, x) = 2;
                }
                else
                {
                    if (lastSide == 0)
                    {
                        boundaryMask.at<int>(y, x) = CGeometry::getQuadrant(cd1);
                    }
                    else
                    {
                        boundaryMask.at<int>(y, x) = CGeometry::getQuadrant(cd2);
                    }
                }
            }
        }
        else if (lby * lbx < 0)
        {
            if (lby > 0)
            {
                if (err > 0)
                {
                    boundaryMask.at<int>(y, x) = 3;
                }
                else if (err < 0)
                {
                    boundaryMask.at<int>(y, x) = 1;
                }
                else
                {
                    if (lastSide == 0)
                    {
                        boundaryMask.at<int>(y, x) = CGeometry::getQuadrant(cd1);
                    }
                    else
                    {
                        boundaryMask.at<int>(y, x) = CGeometry::getQuadrant(cd2);
                    }
                }
            }
            else
            {
                if (err > 0)
                {
                    boundaryMask.at<int>(y, x) = 1;
                }
                else if (err < 0)
                {
                    boundaryMask.at<int>(y, x) = 3;
                }
                else
                {
                    if (lastSide == 0)
                    {
                        boundaryMask.at<int>(y, x) = CGeometry::getQuadrant(cd1);
                    }
                    else
                    {
                        boundaryMask.at<int>(y, x) = CGeometry::getQuadrant(cd2);
                    }
                }
            }
        }
        else
        {
            if (lastSide == 0)
            {
                boundaryMask.at<int>(y, x) = CGeometry::getQuadrant(cd1);
            }
            else
            {
                boundaryMask.at<int>(y, x) = CGeometry::getQuadrant(cd2);
            }
        }
    }

    if (mode == 0 && !clr)
    {
        boundaryMask.at<int>(y, x) = 1;
    }

    ///
    /// segment mask
    ///

    segmentMask.at<int>(y, x) = num;

    if (!clr)
    {
        return;
    }

    // ����ָ��9���������������
    //case 1 i1 x + 0 y + 1
    //case 1 i2 x + 1 y + 0
    //case 2 i1 x - 1 y + 0
    //case 2 i2 x + 0 y + 1
    //case 3 i1 x - 1 y + 0
    //case 3 i2 x + 0 y - 1
    //case 4 i1 x + 0 y - 1
    //case 4 i2 x + 1 y + 0
    // 9����������
    //case 5 i1 x + 1 y + 0
    //case 5 i2 x + 1 y + 0
    //case 6 i1 x + 0 y + 1
    //case 6 i2 x + 0 y + 1
    //case 7 i1 x - 1 y + 0
    //case 7 i2 x - 1 y + 0
    //case 8 i1 x + 0 y - 1
    //case 8 i2 x + 0 y - 1
    switch (boundaryMask.at<int>(y, x))  // i1 index of dx[4], dy[4]. i2 same.
    {
    case 1:  // 5 6
        i1 = 1, i2 = 3;
        break;
    case 2:  // 6 7
        i1 = 0, i2 = 1;
        break;
    case 3:  // 7 8
        i1 = 0, i2 = 2;
        break;
    case 4:  // 5 8
        i1 = 2, i2 = 3;
        break;
    case 5:  // 0
        i1 = 3, i2 = 3; // =
        break;
    case 6:  // 0
        i1 = 1, i2 = 1; // =
        break;
    case 7:  // 0
        i1 = 0, i2 = 0; // =
        break;
    case 8:  // 0
        i1 = 2, i2 = 2; // =
        break;
    default:
        break;
    }

    Vec3f c1, c2;
    calcColorFromDist(segmentColorVec[num - 1], y, x, c1, c2);


    int dx[4] = {-1, 0, 0, 1};
    int dy[4] = {0, 1, -1, 0};
    if (mode == 0)
    {// Mono Line
        colorMask.at<Vec3f>(y, x) = c1;
        vecMask.at<unsigned char>(y, x) = 1;
    }
    else if (mode == 1)
    {// Dual Line
        if (boundaryMask.at<int>(y, x) == CGeometry::getQuadrant(cd1))
        {
            colorMask.at<Vec3f>(y, x) = c1;
            lastSide = 0;
            vecMask.at<unsigned char>(y, x) = 1;
            //tmp_vec3.first.push_back(Vec2i(y, x));
            sideMask.at<int>(y, x) = -(stroke_ID + 1);
            int ny = y + dy[i1], nx = x + dx[i1];
            if (in(ny, nx) && bdmakeup(ny, nx, num, li1, li2, i1, st, end))
            {
                if (sideMask.at<int>(ny, nx) == 0 || !is_end_point(ny, nx, stroke_ID) ||
                    is_end_point(ny, nx, abs(sideMask.at<int>(ny, nx)) - 1))
                {
                    colorMask.at<Vec3f>(ny, nx) = c2;
                    boundaryMask.at<int>(ny, nx) = CGeometry::getQuadrant(cd2);
                    segmentMask.at<int>(ny, nx) = num;
                    vecMask.at<unsigned char>(ny, nx) = 2;
                    sideMask.at<int>(ny, nx) = (stroke_ID + 1);
                    //tmp_vec3.second.push_back(Vec2i(ny, nx));
                }
            }
            if (i1 != i2)
            {
                ny = y + dy[i2], nx = x + dx[i2];
                if (in(ny, nx) && bdmakeup(ny, nx, num, li1, li2, i2, st, end))
                {
                    if (sideMask.at<int>(ny, nx) == 0 || !is_end_point(ny, nx, stroke_ID) ||
                        is_end_point(ny, nx, abs(sideMask.at<int>(ny, nx)) - 1))
                    {
                        colorMask.at<Vec3f>(ny, nx) = c2;
                        boundaryMask.at<int>(ny, nx) = CGeometry::getQuadrant(cd2);
                        segmentMask.at<int>(ny, nx) = num;
                        vecMask.at<unsigned char>(ny, nx) = 2;
                        sideMask.at<int>(ny, nx) = (stroke_ID + 1);
                        //tmp_vec3.second.push_back(Vec2i(ny, nx));
                    }
                }
            }
        }
        else if (boundaryMask.at<int>(y, x) == CGeometry::getQuadrant(cd2))
        {
            colorMask.at<Vec3f>(y, x) = c2;
            lastSide = 1;
            vecMask.at<unsigned char>(y, x) = 2;
            sideMask.at<int>(y, x) = (stroke_ID + 1);

            //tmp_vec3.second.push_back(Vec2i(y, x));
            int ny = y + dy[i1], nx = x + dx[i1];
            if (in(ny, nx) && bdmakeup(ny, nx, num, li1, li2, i1, st, end))
            {
                if (sideMask.at<int>(ny, nx) == 0 || !is_end_point(ny, nx, stroke_ID) ||
                    is_end_point(ny, nx, abs(sideMask.at<int>(ny, nx)) - 1))
                {
                    colorMask.at<Vec3f>(ny, nx) = c1;
                    boundaryMask.at<int>(ny, nx) = CGeometry::getQuadrant(cd1);
                    segmentMask.at<int>(ny, nx) = num;
                    vecMask.at<unsigned char>(ny, nx) = 1;
                    sideMask.at<int>(ny, nx) = -(stroke_ID + 1);
                    //tmp_vec3.first.push_back(Vec2i(ny, nx));
                }
            }
            if (i1 != i2)
            {
                ny = y + dy[i2], nx = x + dx[i2];
                if (in(ny, nx) && bdmakeup(ny, nx, num, li1, li2, i2, st, end))
                {
                    if (sideMask.at<int>(ny, nx) == 0 || !is_end_point(ny, nx, stroke_ID) ||
                        is_end_point(ny, nx, abs(sideMask.at<int>(ny, nx)) - 1))
                    {
                        colorMask.at<Vec3f>(ny, nx) = c1;
                        boundaryMask.at<int>(ny, nx) = CGeometry::getQuadrant(cd1);
                        segmentMask.at<int>(ny, nx) = num;
                        vecMask.at<unsigned char>(ny, nx) = 1;
                        sideMask.at<int>(ny, nx) = -(stroke_ID + 1);
                        //tmp_vec3.first.push_back(Vec2i(ny, nx));
                    }
                }
            }
        }
        else
        {
            puts("ln 705");
        }

        li1 = i1, li2 = i2;
    }
}

#if 1

void CRegionFZ::plotLine(
        int oy, int y1, int ox, int x1,
        Vec3f c11, Vec3f c12, Vec3f c21, Vec3f c22,
        int mode, bool & ls, bool clr, int k)
{
    // line: segment (y0, x0) -> (y1, x1)
    // mode: 0 for mono stroke, 1 for dual
    // ls: last side. always pass in 0
    // clr: false for plotLastStroke lap region, true for boundary/lap edge. passed into labelBoundry()
    // k: stroke segment: 0, 1, 1, 1, ... , 1, 2

    if (oy == y1 && ox == x1)
    {
        return;
    }
    int x0 = ox, y0 = oy;
    Vec2f cd1, cd2;
    if (in(y0, x0))
    {
        nodeMask.at<bool>(y0, x0) = true;
        nm1.at<bool>(y0, x0) = true;
    }
    //    if (ls == 0)
    //    {
    //        if (!tmp_vec3.first.empty())
    //        {
    //            tmp_vec3.first.pop_back();
    //        }
    //    }
    //    else
    //    {
    //        if (!tmp_vec3.second.empty())
    //        {
    //            tmp_vec3.second.pop_back();
    //        }
    //    }
    if (pixel_num > 0)
    {
        pixel_num--;
    }
    segmentColorVec.push_back(SegmentFZ(y0, x0, y1, x1, c11, c12, c21, c22, mode));
    cd1 = CGeometry::normVleft(y0, y1, x0, x1), cd2 = -cd1;
    int lbx = (x1 - x0), lby = (y1 - y0);
    int dx = abs(y1 - y0), dy = -abs(x1 - x0);
    int sx = y0 < y1 ? 1 : -1, sy = x0 < x1 ? 1 : -1;
    bool start = (k > 0);
    bool end;
    int err = dx + dy, e2, d;
    for (;;)
    {
        d = lbx * (y0 - oy) - lby * (x0 - ox);
        end = (y0 == y1) && (x0 == x1) && (k < 2);
        if (in(y0, x0))
        {
            //added by HOU FEI
            if (sideMask.at<int>(y0, x0) == 0 || !is_end_point(y0, x0, stroke_ID) ||
                is_end_point(y0, x0, abs(sideMask.at<int>(y0, x0)) - 1))
            {
                // d: 'err'.
                // start: 0 for 1st seg in stroke, 1 for else
                // end: 1 for (seg.size - 2)-th seg, 0 for else
                // ls: last side. always pass in 0
                // clr: false for plotLastStroke dual line, else true. passed into labelBoundry()
                // li1, li2: set to -1 for this stroke (NOT this seg), may update elsewhere? 
                LabelBoundary(y0, x0, lbx, lby, d, segmentColorVec.size(), cd1, cd2, start, end, ls, clr, li1, li2);
                pixel_num++;
            }
        }
        start = false;
        e2 = 2 * err;

        if (e2 >= dy)
        {
            if (y0 == y1)
            {
                break;
            }
            err += dy;
            y0 += sx;
        }
        if (e2 <= dx)
        {
            if (x0 == x1)
            {
                break;
            }
            err += dx;
            x0 += sy;
        }
#if 0
        d = lbx*(y0-oy)-lby*(x0-ox);
        if ((2*err>=dy && y0==y1) || (2*err<=dx && x0==x1))
            LabelBoundary(y0,x0,lbx,lby,d,segmentColorVec.size(),cd1,cd2,strokeN,true,ls);
        else
            LabelBoundary(y0,x0,lbx,lby,d,segmentColorVec.size(),cd1,cd2,strokeN,false,ls);
#endif
    }
}

#endif

//added by HOU FEI
bool CRegionFZ::is_end_point(int y, int x, int ID)
{
    if (ID < 0)
    {
        return false;
    }
    double d1 = (m_strokes[ID].s_points.front() - QPointF(x, y)).manhattanLength();
    double d2 = (m_strokes[ID].s_points.back() - QPointF(x, y)).manhattanLength();
    if (min(d1, d2) <= 2)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void
CRegionFZ::plotColor(Vec3f & c11, Vec3f & c12, Vec3f & c21, Vec3f & c22, const QVector<PointProperties> & pps, int k)
{
    c11[0] = (float) pps[k].color_1.blue();
    c12[0] = (float) pps[k + 1].color_1.blue();
    c11[1] = (float) pps[k].color_1.green();
    c12[1] = (float) pps[k + 1].color_1.green();
    c11[2] = (float) pps[k].color_1.red();
    c12[2] = (float) pps[k + 1].color_1.red();
    c21[0] = (float) pps[k].color_2.blue();
    c22[0] = (float) pps[k + 1].color_2.blue();
    c21[1] = (float) pps[k].color_2.green();
    c22[1] = (float) pps[k + 1].color_2.green();
    c21[2] = (float) pps[k].color_2.red();
    c22[2] = (float) pps[k + 1].color_2.red();
}

void CRegionFZ::plotStrokes()
{
    //	vec3.clear();
    //	vec3.resize(m_strokes.size());

    for (int i = 0; i < m_strokes.size(); i++)
    {
        vec1.clear();
        vec2.clear();
        //        tmp_vec3.first.clear();
        //        tmp_vec3.second.clear();
        vecMask = Mat::zeros(height, width, CV_8UC1);  // vecMask cleaned!
        stroke_ID = i;
        pixel_num = 0;
        bool lastSide = 0;
        if (m_strokes[i].segs.size() <= 1)
        {
            continue;
        }
        li1 = -1, li2 = -1;
        unsigned char f;
        for (int k = 0; k < m_strokes[i].segs.size() - 1; k++)
        {
            if (k == 0)
            {
                f = 0;
            }
            else if (k == m_strokes[i].segs.size() - 2)
            {
                f = 2;
            }
            else
            {
                f = 1;
            }
            if (!in((int) m_strokes[i].segs[k].y() + 0.5 - w00[0], (int) m_strokes[i].segs[k].x() + 0.5 - w00[1]) &&
                !in((int) m_strokes[i].segs[k + 1].y() + 0.5 - w00[0],
                    (int) m_strokes[i].segs[k + 1].x() + 0.5 - w00[1]))
            {
                continue;
            }
            Vec3f c11, c12, c21, c22;
            plotColor(c11, c12, c21, c22, m_strokes[i].pps, k);
            plotLine(
                    (int) m_strokes[i].segs[k].y() + 0.5 - w00[0],
                    (int) m_strokes[i].segs[k + 1].y() + 0.5 - w00[0],
                    (int) m_strokes[i].segs[k].x() + 0.5 - w00[1],
                    (int) m_strokes[i].segs[k + 1].x() + 0.5 - w00[1],
                    c11, c12, c21, c22, m_strokes[i].s_mdmode, lastSide, true, f
            );
        }
    }
}

bool CRegionFZ::inBoundingBox(QPointF a, QPointF b)
{
    return ((int) a.y() + 0.5 >= w00[0] && (int) a.y() + 0.5 <= w00[0] + wh && (int) a.x() + 0.5 >= w00[1] &&
            (int) a.x() + 0.5 <= w00[1] + ww) ||
           ((int) b.y() + 0.5 >= w00[0] && (int) b.y() + 0.5 <= w00[0] + wh && (int) b.x() + 0.5 >= w00[1] &&
            (int) b.x() + 0.5 <= w00[1] + ww);
}

void CRegionFZ::plotLastStroke(const SQ_Stroke & ls_stroke,
                               int mode)  //mode == 0: boundary edge/laplacian edge; mode == 1: laplacian region
{
    vec1.clear();
    vec2.clear();
    vecMask = Mat::zeros(height, width, CV_8UC1);  // clean vecMask to get newest result!
    bool lastSide = 0;
    unsigned char f;

    for (int k = 0; k < ls_stroke.segs.size() - 1; k++)
    {

        if (!in((int) ls_stroke.segs[k].y() + 0.5 - w00[0], (int) ls_stroke.segs[k].x() + 0.5 - w00[1]) &&
            !in((int) ls_stroke.segs[k + 1].y() + 0.5 - w00[0], (int) ls_stroke.segs[k + 1].x() + 0.5 - w00[1]))
        {
            continue;
        }

        if (k == 0)
        {
            f = 0;
        }
        else if (k == ls_stroke.segs.size() - 2)
        {
            f = 2;
        }
        else
        {
            f = 1;
        }


        if (mode == 0)  // mode == 0: boundary edge/laplacian edge;
        {
            Vec3f c11, c12, c21, c22;
            if (!ls_stroke.pps.isEmpty())
            {
                plotColor(c11, c12, c21, c22, ls_stroke.pps, k);
            }
            plotLine(
                    (int) ls_stroke.segs[k].y() + 0.5 - w00[0],
                    (int) ls_stroke.segs[k + 1].y() + 0.5 - w00[0],
                    (int) ls_stroke.segs[k].x() + 0.5 - w00[1],
                    (int) ls_stroke.segs[k + 1].x() + 0.5 - w00[1],
                    c11, c12, c21, c22, ls_stroke.s_mdmode, lastSide, true, f
            );
        }
        else if (mode == 1)  //  mode == 1: laplacian region
        {
            //            plotLine(
            //                    (int) ls_stroke.segs[k].y() + 0.5 - w00[0],
            //                    (int) ls_stroke.segs[k + 1].y() + 0.5 - w00[0],
            //                    (int) ls_stroke.segs[k].x() + 0.5 - w00[1],
            //                    (int) ls_stroke.segs[k + 1].x() + 0.5 - w00[1],
            //                    NULL, NULL, NULL, NULL, ls_stroke.s_mdmode, lastSide, false, f
            //            );
            plotLine(
                    (int) ls_stroke.segs[k].y() + 0.5 - w00[0],
                    (int) ls_stroke.segs[k + 1].y() + 0.5 - w00[0],
                    (int) ls_stroke.segs[k].x() + 0.5 - w00[1],
                    (int) ls_stroke.segs[k + 1].x() + 0.5 - w00[1],
                    cv::Vec3f(), cv::Vec3f(), cv::Vec3f(), cv::Vec3f(),
                    ls_stroke.s_mdmode, lastSide, false, f
            );
            if ((int) ls_stroke.segs[k].y() + 0.5 - w00[0] > br[0])
            {
                br[0] = (int) ls_stroke.segs[k].y() + 0.5 - w00[0];
            }
            if ((int) ls_stroke.segs[k + 1].y() + 0.5 - w00[0] > br[0])
            {
                br[0] = (int) ls_stroke.segs[k + 1].y() + 0.5 - w00[0];
            }
            if ((int) ls_stroke.segs[k].x() + 0.5 - w00[1] > br[1])
            {
                br[1] = (int) ls_stroke.segs[k].x() + 0.5 - w00[1];
            }
            if ((int) ls_stroke.segs[k + 1].x() + 0.5 - w00[1] > br[1])
            {
                br[1] = (int) ls_stroke.segs[k + 1].x() + 0.5 - w00[1];
            }
            if ((int) ls_stroke.segs[k].y() + 0.5 - w00[0] < tl[0])
            {
                tl[0] = (int) ls_stroke.segs[k].y() + 0.5 - w00[0];
            }
            if ((int) ls_stroke.segs[k + 1].y() + 0.5 - w00[0] < tl[0])
            {
                tl[0] = (int) ls_stroke.segs[k + 1].y() + 0.5 - w00[0];
            }
            if ((int) ls_stroke.segs[k].x() + 0.5 - w00[1] < tl[1])
            {
                tl[1] = (int) ls_stroke.segs[k].x() + 0.5 - w00[1];
            }
            if ((int) ls_stroke.segs[k + 1].x() + 0.5 - w00[1] < tl[1])
            {
                tl[1] = (int) ls_stroke.segs[k + 1].x() + 0.5 - w00[1];
            }
        }
    }

    if (mode == 1)
    {
        return;
    }

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Vec3f clr = colorMask.at<Vec3f>(y, x);

            if (vecMask.at<unsigned char>(y, x) == 1)
            {
                vec1.push_back(make_pair(Vec2i(y, x), clr));
                //sideMask.at<int>(y, x) = 1;
            }
            else if (vecMask.at<unsigned char>(y, x) == 2)
            {
                vec2.push_back(make_pair(Vec2i(y, x), clr));
                //sideMask.at<int>(y, x) = 0;
            }
        }
    }
}

///
/// Xi Han: this func has nothing to do with lap edge/regions
///         only called by boundary() for DCs
///
int CRegionFZ::findRegions()
{
    queue<int> q;
    Mat vis(height, width, CV_8UC1, cv::Scalar::all(0));

    int cnt = 1, cur, curx, cury, nx, ny, p;
    int dx[4] = {-1, 0, 0, 1};
    int dy[4] = {0, 1, -1, 0};

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            //if ( (regionMask.at<int>(y,x) != 0) || (boundaryMask.at<int>(y,x) != 0) ) {
            //	continue;
            //}
            if ((regionMask.at<int>(y, x) != 0) || (sideMask.at<int>(y, x) != 0))
            {
                continue;
            }
            q.push(y * width + x);
            vis.at<bool>(y, x) = 1; // visited?
            while (!q.empty())
            {
                cur = q.front();
                q.pop();
                curx = cur % width, cury = cur / width;
                if (regionMask.at<int>(cury, curx) >= 0)
                {
                    regionMask.at<int>(cury, curx) = cnt;
                }
                //if (1) {
                //	regionMask.at<int>(cury, curx) = cnt;
                //}
                for (int i = 0; i < 4; i++)
                {
                    nx = curx + dx[i], ny = cury + dy[i];
                    if (!in(ny, nx) || vis.at<bool>(ny, nx))
                    {
                        continue;
                    }
                    p = ny * width + nx;
                    if (sideMask.at<int>(ny, nx) == 0)
                    {
                        if (colorMask.at<Vec3f>(ny, nx) != Vec3f(0, 0, 0)) // only for MONO boundary strokes
                        {
                            regionMask.at<int>(ny, nx) = -cnt;
                        }
                        q.push(p);
                        vis.at<bool>(ny, nx) = 1;
                    }
                    else
                    {
                        /*if (((i == 0) && (boundaryMask.at<int>(ny,nx) == 2 || boundaryMask.at<int>(ny,nx) == 3 || boundaryMask.at<int>(ny,nx) == 7)) ||
                             ((i == 1) && (boundaryMask.at<int>(ny,nx) == 1 || boundaryMask.at<int>(ny,nx) == 2 || boundaryMask.at<int>(ny,nx) == 6)) ||
                             ((i == 2) && (boundaryMask.at<int>(ny,nx) == 3 || boundaryMask.at<int>(ny,nx) == 4 || boundaryMask.at<int>(ny,nx) == 8)) ||
                             ((i == 3) && (boundaryMask.at<int>(ny,nx) == 1 || boundaryMask.at<int>(ny,nx) == 4 || boundaryMask.at<int>(ny,nx) == 5))) {*/
                        regionMask.at<int>(ny, nx) = -cnt;
                        vis.at<bool>(ny, nx) = 1;
                    }
                }
            }
            cnt++;
        }
    }

    int size = cnt - 1;
    setSize(size);

    //for (int y = 0; y < height; y++) {
    //	for (int x = 0; x < width; x++) {
    //		if ( (regionMask.at<int>(y,x) == 0) && (boundaryMask.at<int>(y,x) != 0) ) {
    //			regionMask.at<int>(y,x) = -(cnt+1);
    //			cnt++;
    //		}
    //	}
    //}
    return size;
}

void CRegionFZ::calcColorFromDist(const SegmentFZ s, int y, int x, Vec3f & nc1, Vec3f & nc2)
{
    double e;
    int dx = s.x1 - s.x0, dy = s.y1 - s.y0;
    if (x == s.x0 && y == s.y0)
    {
        e = 0;
    }
    else
    {
        e = sqrt((double) (s.x0 - x) * (s.x0 - x) + (s.y0 - y) * (s.y0 - y)) / sqrt((double) dx * dx + dy * dy);
    }
    nc1 = s.c11 + (s.c12 - s.c11) * e, nc2 = s.c21 + (s.c22 - s.c21) * e;
    for (int i = 0; i < 3; i++)
    {
        if (nc1[i] > 255)
        {
            nc1[i] = 255;
        }
        if (nc2[i] > 255)
        {
            nc2[i] = 255;
        }
    }
}

void CRegionFZ::findBoundary()
{
    int curr, newr, nx, ny;
    int dx[4] = {-1, 0, 0, 1};
    int dy[4] = {0, 1, -1, 0};
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            curr = abs(regionMask.at<int>(y, x));
            for (int i = 0; i < 4; i++)
            {
                nx = x + dx[i], ny = y + dy[i];
                if (!in(ny, nx))
                {
                    continue;
                }
                newr = abs(regionMask.at<int>(ny, nx));
                if (curr != newr && curr != 0 && newr != 0)
                {
                    regionMask.at<int>(y, x) = -curr;
                    regionMask.at<int>(ny, nx) = -newr;
                    if (boundaryMask.at<int>(ny, nx) == 0)
                    {
                        colorMask.at<Vec3f>(ny, nx) = colorMask.at<Vec3f>(y, x);
                        sideMask.at<int>(ny, nx) = sideMask.at<int>(y, x);
                    }
                    if (boundaryMask.at<int>(y, x) == 0)
                    {
                        colorMask.at<Vec3f>(y, x) = colorMask.at<Vec3f>(ny, nx);
                        sideMask.at<int>(y, x) = sideMask.at<int>(ny, nx);
                    }
                }
            }
        }
    }
}

Mat CRegionFZ::getRegionMask()
{
    return regionMask;
}

Mat CRegionFZ::getColorMask()
{
    return colorMask;
}

void CRegionFZ::setSize(int sz)
{
    size = sz;
}


void CRegionFZ::boundary(bool flag)
{
    //#define DEBUG_boundary
    if (flag)
    {
        return;
    }
#ifdef DEBUG_boundary
    printf("\n***** START *****\n\n",size);
    ElapasedTime g_time, totaltime;
    printf("***** image height %d width %d *****\n", h(), w());
    totaltime.start();
    g_time.start();
    //plotLastStroke(0);
    plotStrokes();
    printf("\tplotStrokes time %.3lf\n", g_time.getTime());
    g_time.start();
    findRegions();
    printf("\tfindRegions time %.3lf\n", g_time.getTime());
    g_time.start();
    findBoundary();
    printf("\tfindBoundary time %.3lf\n", g_time.getTime());
    printf("***** total time: %.3lf *****\n", totaltime.getTime());
#ifdef outFZ
    out();
    //outp(getStrokePlots(),"resultImg/last_stroke.bmp");
#endif
    printf("\n***** END *****\n\n",size);
#else
    plotStrokes();
    findRegions();
    findBoundary();
#ifdef outFZ
    out();
#endif
#endif
}

vector<vector<pair<cv::Vec2i, cv::Vec3f>>> CRegionFZ::lapEdge(unsigned int strokeID, const SQ_Stroke & ls_stroke)
{
    init();
    vector<vector<pair<Vec2i, Vec3f> > > r;
    plotLastStroke(ls_stroke, 0);
    r.push_back(vec1);  // pixels w/ c1
    r.push_back(vec2);  // pixels w/ c2
#ifdef outFZ
    out_lapEdge();
#endif
    return r;
}

void CRegionFZ::findRegions_floodFill(cv::Mat * image, cv::Mat * mask, cv::Vec2i seedPoint, unsigned char newVal,
                                      cv::Rect * rect)
{
    int h = image->size().height + 1, w = image->size().width + 1;

    int cur, curx, cury, nx, ny, p;
    int y = seedPoint[0], x = seedPoint[1];
    Mat vis = cv::Mat::zeros(h, w, CV_8UC1);
    int dx[4] = {-1, 0, 0, 1};
    int dy[4] = {0, 1, -1, 0};
    queue<int> q;
    q.push(y * w + x);
    vis.at<bool>(y, x) = 1;
    while (!q.empty())
    {
        cur = q.front();
        q.pop();
        curx = cur % w, cury = cur / w;
        mask->at<unsigned char>(cury, curx) = newVal;
        for (int i = 0; i < 4; i++)
        {
            nx = curx + dx[i], ny = cury + dy[i];
            if (!(rect->tl().y <= ny && ny <= rect->br().y && rect->tl().x <= nx && nx <= rect->br().x))
            {
                continue;
            }
            if (vis.at<bool>(ny, nx))
            {
                continue;
            }
            p = ny * w + nx;
            vis.at<bool>(ny, nx) = 1;
            if (ny == 0 || nx == 0)
            {
                q.push(p);
            }
            else if (image->at<unsigned char>(ny - 1, nx - 1) == 0)
            {
                q.push(p);
            }
        }
    }
}

Mat CRegionFZ::findRegions_lapRegion_bounding_box()
{
    int cnt = 1, nx, ny;

    cv::Mat mask = cv::Mat::zeros(height + 2, width + 2, CV_8UC1);
    cv::Mat bmask = cv::Mat::zeros(height, width, CV_8UC1);  // boundaryMask

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            bmask.at<unsigned char>(y, x) = (unsigned char) boundaryMask.at<int>(y, x);
        }
    }

    if (tl[0] < 0)
    {
        tl[0] = 0;
    }
    if (tl[1] < 0)
    {
        tl[1] = 0;
    }
    if (br[0] >= height)
    {
        br[0] = height - 1;
    }
    if (br[1] >= width)
    {
        br[1] = width - 1;
    }

    cv::Rect bb(tl[1], tl[0], br[1] - tl[1] + 3, br[0] - tl[0] + 3);
    findRegions_floodFill(&bmask, &mask, cv::Vec2i(tl[0], tl[1]), 255, &bb);  // flood mask posi bmask == 0 with 255

    int dx[4] = {-1, 0, 0, 1};
    int dy[4] = {0, 1, -1, 0};

    if (br[0] - tl[0] >= 0 && br[1] - tl[1] >= 0)
    {
        Mat regionMask_(br[0] - tl[0] + 1, br[1] - tl[1] + 1, CV_8UC1);

        for (int y = 0; y < regionMask_.rows; y++)
        {
            for (int x = 0; x < regionMask_.cols; x++)
            {
                regionMask_.at<uchar>(y, x) = !mask.at<uchar>(tl[0] + y + 1, tl[1] + x + 1);
            }
        }

        return regionMask_;
    }
    else
    {
        return Mat();
    }
}

pair<Mat, Rect> CRegionFZ::lapRegion(unsigned int strokeID, const SQ_Stroke & ls_stroke)
{
    init();

    tl[0] = height - 1, tl[1] = width - 1;
    br[0] = 0, br[1] = 0;

    plotLastStroke(ls_stroke, 1);  // only sets boundryMask, segMask for sample pixels for this call
    Mat mask = findRegions_lapRegion_bounding_box();

#ifdef outFZ
    imwrite("resultImg/_region_mask_lapRegion.bmp", regionMask);
#endif

    return make_pair(
            mask,
            Rect(tl[1], tl[0], br[1] - tl[1] + 1, br[0] - tl[0] + 1));
}


#ifdef outFZ

void CRegionFZ::out()
{
    //printf("***** number of regions: %d *****\n",size);

    Mat img(height, width, CV_32SC3);
    Mat img_c(height, width, CV_32SC3);
    Mat img_c_n(height, width, CV_32SC3);
    Mat img_s(height, width, CV_32SC3);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int mk = regionMask.at<int>(y, x);
            int s = sideMask.at<int>(y, x);

            Vec3i mk_c, mk_c_n;

            for (int i = 0; i < 3; i++)
            {
                mk_c[i] = mk_c_n[i] = (int) colorMask.at<Vec3f>(y, x)[i];

                if (nodeMask.at<bool>(y, x))
                {
                    mk_c_n[i] /= 1.5;
                }

                if (nm1.at<bool>(y, x))
                {
                    mk_c_n[i] /= 2;
                }

                img.at<Vec3i>(y, x)[i] = (int) ((double) abs(mk) / (double) size * 255.0);

                if (mk < 0)
                {
                    //img.at<Vec3i>(y, x)[i] -= 255 / size / 2;
                    img.at<Vec3i>(y, x)[i] = 255;
                }
            }

            if (mk_c == Vec3i(0, 0, 0) && mk > 0)
            {
                img_c.at<Vec3i>(y, x) = Vec3i(255, 255, 255);
                img_c_n.at<Vec3i>(y, x) = Vec3i(255, 255, 255);
            }
            else
            {
                img_c.at<Vec3i>(y, x) = mk_c;
                img_c_n.at<Vec3i>(y, x) = mk_c_n;
                //img.at<Vec3i>(y, x) = mk_c;
            }

            if (s < 0)
            {
                img_s.at<Vec3i>(y, x) = Vec3i(255, 255, 255);
            }
            else if (s > 0)
            {
                img_s.at<Vec3i>(y, x) = Vec3i(127, 127, 127);
            }
            else
            {
                img_s.at<Vec3i>(y, x) = Vec3i(0, 0, 0);
            }
        }
    }

    imwrite("resultImg/_region_mask.bmp", img);
    imwrite("resultImg/_color_mask.bmp", img_c);
    imwrite("resultImg/_color_mask_with_nodes.bmp", img_c_n);
    imwrite("resultImg/_side_mask.bmp", img_s);
}

void CRegionFZ::out_lapEdge()
{
    //printf("***** number of regions: %d *****\n",size);
    Mat img(height, width, CV_32SC1), img_c(height, width, CV_32SC3), img_c_n(height, width, CV_32SC3);
    int mk;
    Vec3i mk_c, mk_c_n;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            mk = regionMask.at<int>(y, x);
            for (int i = 0; i < 3; i++)
            {
                mk_c[i] = mk_c_n[i] = (int) colorMask.at<Vec3f>(y, x)[i];
                if (nodeMask.at<bool>(y, x))
                {
                    mk_c_n[i] /= 1.5;
                }
                if (nm1.at<bool>(y, x))
                {
                    mk_c_n[i] /= 2;
                }
            }
            //if (mk_c == Vec3i(-1, -1, -1)) {
            if (mk_c == Vec3i(0, 0, 0) && mk > 0)
            {
                img_c.at<Vec3i>(y, x) = Vec3i(255, 255, 255);
                img_c_n.at<Vec3i>(y, x) = Vec3i(255, 255, 255);
            }
            else
            {
                img_c.at<Vec3i>(y, x) = mk_c;
                img_c_n.at<Vec3i>(y, x) = mk_c_n;
            }
        }
    }
    imwrite("resultImg/_color_mask_lapEdge.bmp", img_c);
    imwrite("resultImg/_color_mask_with_nodes_lapEdge.bmp", img_c_n);
}

void CRegionFZ::outp(vector<Vec2i> pt, string fn)
{
    Mat img(height, width, CV_32SC1);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            img.at<int>(y, x) = 255;
        }
    }
    for (int i = 0; i < pt.size(); i++)
    {
        img.at<int>(pt[i][0], pt[i][1]) = 0;
    }
    imwrite(fn, img);
}

void CRegionFZ::outp(vector<vector<Vec2i> > ptv, string fn)
{
    Mat img(height, width, CV_32SC1);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            img.at<int>(y, x) = 255;
        }
    }
    int cl = 0;
    for (int j = 0; j < ptv.size(); j++)
    {
        vector<Vec2i> pt = ptv[j];
        for (int i = 0; i < pt.size(); i++)
        {
            img.at<int>(pt[i][0], pt[i][1]) = cl;
            if (nodeMask.at<bool>(pt[i][0], pt[i][1]))
            {
                img.at<int>(pt[i][0], pt[i][1]) += 20;
            }
            if (nm1.at<bool>(pt[i][0], pt[i][1]))
            {
                img.at<int>(pt[i][0], pt[i][1]) += 20;
            }
        }
        cl += 255 / ptv.size();
    }
    imwrite(fn, img);
}

void CRegionFZ::outp(std::vector<std::pair<std::vector<cv::Vec2i>, std::vector<cv::Vec2i> > > vec, string fn)
{
    Mat img(height, width, CV_32SC1);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            img.at<int>(y, x) = 255;
        }
    }
    int cl1 = 0, cl2 = 127;
    for (int j = 0; j < vec.size(); j++)
    {
        vector<Vec2i> pt1 = vec[j].first, pt2 = vec[j].second;
        for (int i = 0; i < pt1.size(); i++)
        {
            img.at<int>(pt1[i][0], pt1[i][1]) = cl1;
        }
        for (int i = 0; i < pt2.size(); i++)
        {
            img.at<int>(pt2[i][0], pt2[i][1]) = cl2;
        }
    }
    imwrite(fn, img);
}

#endif

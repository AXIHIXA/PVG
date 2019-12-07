#ifndef REGIONFZ_H
#define REGIONFZ_H

#include <opencv2/core.hpp>
#include <QWidget>
#include <vector>

class SQ_Stroke;

struct PointProperties;

struct SegmentFZ
{
    int x0, y0, x1, y1, mode;
    cv::Vec3f c11, c12, c21, c22;

    SegmentFZ(int y0, int x0, int y1, int x1, const cv::Vec3f & c11, const cv::Vec3f & c12, const cv::Vec3f & c21,
              const cv::Vec3f & c22, int mode) :
            x0(x0), y0(y0), x1(x1), y1(y1), c11(c11), c12(c12), c21(c21), c22(c22), mode(mode)
    {
    };

    ~SegmentFZ()
    {
    };
};

class CRegionFZ
{
public:

    CRegionFZ(int h, int w, const QVector<SQ_Stroke> & strokes, double scaleFactor, const cv::Vec2i & w00);

    CRegionFZ(const CRegionFZ &) = delete;

    CRegionFZ & operator=(const CRegionFZ &) = delete;

    void boundary(bool flag);

    cv::Mat getRegion();

    cv::Mat getColor();

    //m_boundary == 1
    std::vector<std::vector<std::pair<cv::Vec2i, cv::Vec3f >>>

    lapEdge(unsigned int strokeID, const SQ_Stroke & ls_stroke);

    //m_boundary == 2
    std::pair<cv::Mat, cv::Rect> lapRegion(unsigned int strokeID, const SQ_Stroke & ls_stroke);

    void setSize(int sz);

    cv::Mat get_sideMask()
    {
        return sideMask;
    }

    void set_scale(double _scaleFactor)
    {
        scaleFactor = _scaleFactor;
    }

    void set_w00(const cv::Vec2i & _w00)
    {
        w00 = _w00;
    }

    void set_dim(int h, int w)
    {
        wh = height = h;
        ww = width = w;
    }

    int w() const
    {
        return width;
    }

    int h() const
    {
        return height;
    }

private:
    const QVector<SQ_Stroke> & m_strokes;
    cv::Mat nodeMask;
    cv::Mat nm1;
    cv::Mat vecMask;

private:
    void init();

    void calcColorFromDist(const SegmentFZ s, int y, int x, cv::Vec3f & nc1, cv::Vec3f & nc2);

    void LabelBoundary(
            int y, int x, int lbx, int lby,
            int err, int num,
            cv::Vec2f cd1, cv::Vec2f cd2,
            bool st, bool end, bool & lastSide, bool clr,
            int & li1, int & li2);

    void plotLine(
            int y0, int y1, int x0, int x1,
            cv::Vec3f c11, cv::Vec3f c12, cv::Vec3f c21, cv::Vec3f c22,
            int mode, bool & lastSide, bool clr, int k);

    void plotColor(cv::Vec3f & c11, cv::Vec3f & c12, cv::Vec3f & c21, cv::Vec3f & c22,
                   const QVector<PointProperties> & pps, int k);

    inline bool in(int y, int x);

    inline bool bdmakeup(int y, int x);

    inline bool bdmakeup(int y, int x, int num, int li1, int li2, int i, bool st, bool end);

    inline bool bdmakeup(int y, int x, int num);

    bool inBoundingBox(QPointF a, QPointF b);

    bool is_end_point(int y, int x, int ID);

private:
    int height, wh;
    int width, ww;
    int size;
    int li1, li2;
    int stroke_ID, pixel_num;

    double scaleFactor;

    bool isDualLine;

    cv::Vec2i w00;
    std::vector<SegmentFZ> segmentColorVec;
    cv::Mat boundaryMask;
    cv::Mat segmentMask;

    cv::Mat regionMask;
    cv::Mat colorMask;

    cv::Mat sideMask;

    std::vector<std::pair<cv::Vec2i, cv::Vec3f> > vec1, vec2;

    //    std::pair <std::vector< cv::Vec2i >, std::vector< cv::Vec2i >> tmp_vec3;
    cv::Vec2i tl, br;

    int findRegions();

    cv::Mat findRegions_lapRegion_bounding_box();

    void
    findRegions_floodFill(cv::Mat * image, cv::Mat * mask, cv::Vec2i seedPoint, unsigned char newVal, cv::Rect * rect);

    void findBoundary();

    void plotStrokes();

    void plotLastStroke(const SQ_Stroke & ls_stroke,
                        int mode);//mode == 0: boundary edge/laplacian edge; mode == 1: laplacian region
    void out();

    void out_lapEdge();

    void outp(std::vector<cv::Vec2i> pt, std::string fn);

    void outp(std::vector<std::vector<cv::Vec2i >> pt, std::string fn);

    void outp(std::vector<std::pair<std::vector<cv::Vec2i>, std::vector<cv::Vec2i> > > vec, std::string fn);
};

inline bool CRegionFZ::in(int y, int x)
{
    return (x >= 0) && (x < width) && (y >= 0) && (y < height);
}

#ifdef nocoverFZ
inline bool CRegionFZ::bdmakeup(int y, int x)
{
    return
        ( colorMask.at<cv::Vec3f>(y,x) == cv::Vec3f(0,0,0) ) &&
        ( vecMask.at<unsigned char>(y,x) == 0 );
}
#endif

inline bool CRegionFZ::bdmakeup(int y, int x, int num, int li1, int li2, int i, bool st, bool end)
{
    if (end)
    {
        return false;
    }
    bool A = ((li1 == i) || (li2 == i));
    bool B = (((colorMask.at<cv::Vec3f>(y, x) == cv::Vec3f(0, 0, 0)) && (vecMask.at<unsigned char>(y, x) == 0)) ||
              (segmentMask.at<int>(y, x) != num &&
               segmentMask.at<int>(y, x) != num - 1)); // B: not sampled in seg, or,
    if (st)
    {
        return A && B;
    }
    else
    {
        return B;
    }
}

inline bool CRegionFZ::bdmakeup(int y, int x, int num)
{
    return
            (((colorMask.at<cv::Vec3f>(y, x) == cv::Vec3f(0, 0, 0)) && (vecMask.at<unsigned char>(y, x) == 0)) ||
             (segmentMask.at<int>(y, x) != num && segmentMask.at<int>(y, x) != num - 1));
}

#endif

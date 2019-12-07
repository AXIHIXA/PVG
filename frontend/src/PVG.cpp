#include "auxiliary.h"
#include "debug.h"
#include "pathstroke_auxiliary.h"
#include "poisson_solver.h"
#include "PVG.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ParseColor.hpp>
#include <QDebug>
#include <QFile>
#include <tbb/tbb.h>
#include <thread>


PVG::PVG(const QString & filename, double sf, const QPoint & w)
{
    // cpu capacity
    max_threads = std::min(std::thread::hardware_concurrency(), 16U);
    st_info("max_threads = %u", max_threads);
    regions.resize(max_threads);

    // resize magnifier
    scaleFactor = sf;
    w00 = w;

    // open & solve
    open(filename, scaleFactor);
    QPair<Region *, cv::Mat> res = discretization();
    //cv::imwrite("resultImg/lapimg.bmp", res.second);
    evaluation(res.first, res.second, 4);
}

void PVG::save(const QString & filename)
{
    cv::imwrite(filename.toStdString(), result);
}

void PVG::decompress(const QString & filename)
{
    if (filename.isEmpty())
    {
        st_error("empty filename");
        abort();
    }

    QByteArray byteArray;
    QFile fin(filename);

    if (!fin.open(QIODevice::ReadOnly))
    {
        st_error("reading file error");
        abort();
    }

    byteArray = fin.readAll();
    fin.close();

    if (byteArray.isEmpty())
    {
        st_error("reading file error");
        abort();
    }

    byteArray = qUncompress(byteArray);
    tinyxml2::XMLDocument xmlDoc;
    xmlDoc.Parse(byteArray.toStdString().c_str());

    QString xmlFilename = filename;
    xmlFilename.replace(".pvg", ".xml");
    xmlDoc.SaveFile(xmlFilename.toStdString().c_str());
}

void PVG::open(const QString & filename, double scale)
{
    if (filename.isEmpty())
    {
        st_error("empty filename");
        abort();
    }

    ///
    /// update scale factor
    ///

    scaleFactor = scale;

    ///
    /// read compressed file
    ///

    QByteArray byteArray;
    QFile fin(filename);

    if (!fin.open(QIODevice::ReadOnly))
    {
        st_error("reading file error");
        abort();
    }

    byteArray = fin.readAll();
    fin.close();

    if (byteArray.isEmpty())
    {
        st_error("reading file error");
        abort();
    }

    ///
    /// unconpress to xml
    ///

    byteArray = qUncompress(byteArray);
    tinyxml2::XMLDocument xmlDoc;
    xmlDoc.Parse(byteArray.toStdString().c_str());

#ifdef SAVE_DECOMPRESSED_PVG_XML
    QString xmlFilename = filename;
    xmlFilename.replace(".pvg", ".xml");
    xmlDoc.SaveFile(xmlFilename.toStdString().c_str());
#endif

    tinyxml2::XMLNode * rootNode = xmlDoc.FirstChild();

    ///
    /// read canvas dimension
    ///

    tinyxml2::XMLElement * dimensionElement = rootNode->FirstChildElement("dim");
    const char * dim = dimensionElement->GetText();
    std::vector<int> num = split_to_num<int>(dim);

    if (num.empty())
    {
        st_error("PVG file dimension corrupted");
        abort();
    }

    size = QSize(int(scale * num[0]), int(scale * num[1]));

    ///
    /// read m_strokes
    ///

    m_strokes.clear();

    tinyxml2::XMLElement * strokeElement = rootNode->FirstChildElement("m_strokes");

    for (tinyxml2::XMLElement * se = strokeElement->FirstChildElement(); se; se = se->NextSiblingElement())
    {
        m_strokes.push_back(parseStroke(se, scale, true));
    }

    ///
    /// read lap_edges
    ///

    lap_edges.clear();
    tinyxml2::XMLElement * lapEdgeElement = rootNode->FirstChildElement("lap_edges");

    for (tinyxml2::XMLElement * se = lapEdgeElement->FirstChildElement(); se; se = se->NextSiblingElement())
    {
        lap_edges.push_back(parseStroke(se, scale, true));
    }

    ///
    /// read lap_regions
    ///

    lap_regions.clear();
    tinyxml2::XMLElement * lapRegionElement = rootNode->FirstChildElement("lap_regions");

    for (tinyxml2::XMLElement * se = lapRegionElement->FirstChildElement(); se; se = se->NextSiblingElement())
    {
        lap_regions.push_back(parseStroke(se, scale, false));
    }

    ///
    /// read lap_edges_control_parameters
    ///

    lap_edges_control_parameters.clear();
    tinyxml2::XMLElement * lap_edges_control_parameters_ele =
            rootNode->FirstChildElement("lap_edges_control_parameters");
    std::vector<float> num1 = split_to_num<float>(lap_edges_control_parameters_ele->GetText());

    for (size_t i = 0; i < num1.size(); i++)
    {
        lap_edges_control_parameters.push_back(num1[i]);
    }

    ///
    /// read lap_regions_control_parameters
    ///

    lap_regions_control_parameters.clear();
    tinyxml2::XMLElement * lap_regions_control_parameters_ele =
            rootNode->FirstChildElement("lap_regions_control_parameters");
    std::vector<float> num2 = split_to_num<float>(lap_regions_control_parameters_ele->GetText());

    if (num2.size() == 6 * lap_regions.size())
    {
        for (size_t i = 0; i < num2.size(); i += 6)
        {
            lap_regions_control_parameters.push_back(
                    qMakePair(
                            cv::Vec3f(num2[i], num2[i + 1], num2[i + 2]),
                            cv::Vec3f(num2[i + 3], num2[i + 4], num2[i + 5])));
        }
    }
    else
    {
        for (size_t i = 0; i < num2.size(); i += 2)
        {
            lap_regions_control_parameters.push_back(
                    qMakePair(
                            cv::Vec3f(num2[i], num2[i], num2[i]),
                            cv::Vec3f(num2[i + 1], num2[i + 1], num2[i + 1])));
        }
    }

    ///
    /// filter useless lines
    ///

    for (int i = 0; i < m_strokes.size(); i++)
    {
        if (m_strokes[i].s_points.size() <= 1)
        {
            m_strokes.remove(i);
            i--;
        }
    }

    for (int i = 0; i < lap_edges.size(); i++)
    {
        if (lap_edges[i].s_points.size() <= 1)
        {
            lap_edges.remove(i);
            lap_edges_control_parameters.remove(i);
            i--;
        }
    }

    for (int i = 0; i < lap_regions.size(); i++)
    {
        if (lap_regions[i].s_points.size() <= 1)
        {
            lap_regions.remove(i);
            lap_regions_control_parameters.remove(i);
            i--;
        }
    }
}

SQ_Stroke PVG::parseStroke(const tinyxml2::XMLElement * SQ_Stroke_ele, double scale, bool parseProperty)
{
    SQ_Stroke stroke;

    const tinyxml2::XMLElement * s_points_ele = SQ_Stroke_ele->FirstChildElement("s_points");
    const char * str1 = s_points_ele->GetText();
    std::vector<double> num1 = split_to_num<double>(str1);

    for (size_t i = 0; i < num1.size(); i += 2)
    {
        stroke.s_points.push_back(QPointF(scale * num1[i], scale * num1[i + 1]));
    }

    if (parseProperty)
    {
        const tinyxml2::XMLElement * s_properties_ele = SQ_Stroke_ele->FirstChildElement("s_properties");

        if (s_properties_ele != nullptr)
        {
            const char * str2 = s_properties_ele->GetText();
            std::vector<int> num2 = split_to_num<int>(str2);

            for (size_t i = 0; i < num2.size(); i += 8)
            {
                PointProperties prop;
                prop.color_1 = QColor(num2[i], num2[i + 1], num2[i + 2]);
                prop.color_2 = QColor(num2[i + 3], num2[i + 4], num2[i + 5]);
                prop.keyframe_1 = (num2[i + 6] == 1);
                prop.keyframe_2 = (num2[i + 7] == 1);
                stroke.s_properties.push_back(prop);
            }
        }
        else
        {
            // for compatible with old files
            for (int i = 0; i < stroke.s_points.size(); i++)
            {
                PointProperties prop;
                prop.color_1 = QColor(127, 127, 127);
                prop.color_2 = QColor(127, 127, 127);
                prop.keyframe_1 = (i == 0 || i == stroke.s_points.size() - 1);
                prop.keyframe_2 = (i == 0 || i == stroke.s_points.size() - 1);
                stroke.s_properties.push_back(prop);
            }
        }
    }

    const tinyxml2::XMLElement * s_mode_ele = SQ_Stroke_ele->FirstChildElement("s_mode");
    const char * str3 = s_mode_ele->GetText();
    std::vector<int> num3 = split_to_num<int>(str3);
    stroke.s_mode = SQ_Stroke::StrokeMode(num3[0]);

    const tinyxml2::XMLElement * s_mdmode_ele = SQ_Stroke_ele->FirstChildElement("s_mdmode");
    const char * str4 = s_mdmode_ele->GetText();
    std::vector<int> num4 = split_to_num<int>(str4);
    stroke.s_mdmode = SQ_Stroke::MDMode(num4[0]);

    return stroke;
}

void PVG::zoomIn(double scale, const QPoint & cur_w00)
{
    m_strokes_scaled = m_strokes;
    lap_edges_scaled = lap_edges;
    lap_regions_scaled = lap_regions;

    for (int i = 0; i < m_strokes_scaled.size(); i++)
    {
        for (int j = 0; j < m_strokes_scaled[i].s_points.size(); j++)
        {
            m_strokes_scaled[i].s_points[j] *= scale;
        }
    }

    for (int i = 0; i < lap_edges_scaled.size(); i++)
    {
        for (int j = 0; j < lap_edges_scaled[i].s_points.size(); j++)
        {
            lap_edges_scaled[i].s_points[j] *= scale;
        }
    }

    for (int i = 0; i < lap_regions_scaled.size(); i++)
    {
        for (int j = 0; j < lap_regions_scaled[i].s_points.size(); j++)
        {
            lap_regions_scaled[i].s_points[j] *= scale;
        }
    }

    CParseColor path(size.height(), size.width(), cur_w00, true);
    path.parse(m_strokes_scaled, 0);
    path.parse(lap_edges_scaled, 1);
    path.parse(lap_regions_scaled, 2);
}

#ifdef QUADTREE_VORONOI_OUTPUT
template <class T>
static std::vector<std::vector<CPoint2i> >
flood_fill(const cv::Mat & image, const T & zero, int n_neighbors, int region_id, const Region & region);
#endif

QPair<Region *, cv::Mat> PVG::discretization()
{
    ///
    /// color interpolation of control points
    ///

    for (int i = 0; i < m_strokes.size(); i++)
    {
        m_strokes[i].updateAll();
    }

    for (int i = 0; i < lap_edges.size(); i++)
    {
        lap_edges[i].updateAll();
    }

    static std::unique_ptr<Region> region;
    static cv::Mat laplacian_image;

#ifdef QUADTREE_VORONOI_OUTPUT
    cv::Mat the_laplacian_image = cv::Mat::zeros(size.height(), size.width(), CV_32FC3);
#endif

    zoomIn(1.0, QPoint(0, 0));

    cv::Mat region_mask;
    cv::Mat side_mask;

    tbb::parallel_invoke
    (
        [&]()
        {
            for (int i = 0; i < max_threads; i++)
            {
                regions[i].reset(new CRegionFZ(size.height(), size.width(), QVector<SQ_Stroke>(), 1, cv::Vec2i(0, 0)));
            }
        },

        [&]()
        {
            CRegionFZ reg(size.height(), size.width(), m_strokes_scaled, 1.0, cv::Vec2i(0, 0));
            reg.boundary(false);

            laplacian_image = reg.getColor();
            region_mask = reg.getRegion();
            side_mask = reg.get_sideMask();

            region.reset(new Region(region_mask, side_mask));
        }
    );

    ///
    /// Laplacian edge generation
    ///

    cv::Mat laplacian_edge_mask(laplacian_image.rows, laplacian_image.cols, CV_32SC1);
    int initial_num = lap_edges_scaled.size() + 1;
    laplacian_edge_mask.setTo(initial_num);

    for (size_t i = 0; i < regions.size(); ++i)
    {
        regions[i]->set_scale(1.0);
        regions[i]->set_w00(cv::Vec2i(0, 0));
    }

    std::vector<std::vector<std::vector<std::pair<cv::Vec2i, cv::Vec3f> > > >
            lap_edge_points(lap_edges_scaled.size());

    std::vector<std::thread> threads(max_threads);

    for (int i = 0; i < max_threads; i++)
    {
        threads[i] = std::thread(
                [&](int g)
                {
                     for (unsigned int r = g; r < lap_edges_scaled.size(); r += max_threads)
                     {
                         lap_edge_points[r] = regions[g]->lapEdge(r, lap_edges_scaled[r]);
                     }
                },
                i);
    }

    for (size_t i = 0; i < threads.size(); i++)
    {
        threads[i].join();
    }

    std::vector<std::vector<std::vector<std::pair<cv::Vec2i, cv::Vec3f>>>> lap_curve_pts;

    for (int r = 0; r < lap_edges_scaled.size(); r++)
    {
        const std::vector<std::vector<std::pair<cv::Vec2i, cv::Vec3f>>> & points_all = lap_edge_points[r];

        std::vector<std::vector<std::pair<cv::Vec2i, cv::Vec3f>>> points(2);

        for (size_t i = 0; i < points_all[0].size(); i++)
        {
            if (region->is_inner_of_a_region(points_all[0][i].first[0], points_all[0][i].first[1]))
            {
                laplacian_edge_mask.at<int>(points_all[0][i].first[0], points_all[0][i].first[1]) = r + 1;
                points[0].push_back(points_all[0][i]);
            }
        }

        for (size_t i = 0; i < points_all[1].size(); i++)
        {
            if (region->is_inner_of_a_region(points_all[1][i].first[0], points_all[1][i].first[1]))
            {
                laplacian_edge_mask.at<int>(points_all[1][i].first[0], points_all[1][i].first[1]) = -(r + 1);
                points[1].push_back(points_all[1][i]);
            }
        }

        lap_curve_pts.push_back(points);
        region->set_side_source(points);

        // Laplacian value

        double sum1 = 0;

        for (size_t i = 0; i < points[0].size(); i++)
        {
            int n = count_neighbor<int>(laplacian_edge_mask, points[0][i].first, -(r + 1));
            float lap_val = n * (255.0f - points[0][i].second[0]) * lap_edges_control_parameters[r] / 255.0f;
            sum1 += lap_val;
            laplacian_image.at<cv::Vec3f>(points[0][i].first[0], points[0][i].first[1]) +=
                    cv::Vec3f(lap_val, lap_val, lap_val);

#ifdef QUADTREE_VORONOI_OUTPUT
            lap_val = (255.0f - points[0][i].second[0]) * lap_edges_control_parameters[r] / 255.0f;
            the_laplacian_image.at< cv::Vec3f >(points[0][i].first[0], points[0][i].first[1]) +=
                    cv::Vec3f(lap_val, lap_val, lap_val);
#endif
        }

#ifndef QUADTREE_VORONOI_OUTPUT
        double sum2 = 0;
        std::vector<float> values(points[1].size());

        for (size_t i = 0; i < points[1].size(); i++)
        {
            int n = count_neighbor<int>(laplacian_edge_mask, points[1][i].first, (r + 1));
            float lap_val = -n * (255.0f - points[1][i].second[0]) * lap_edges_control_parameters[r] / 255.0f;
            sum2 += lap_val;
            values[i] = lap_val;
        }

        if (sum2 != 0)
        {
            double rate = -sum1 / sum2;
            for (size_t i = 0; i < points[1].size(); i++)
            {
                values[i] *= rate;
                laplacian_image.at<cv::Vec3f>(points[1][i].first[0], points[1][i].first[1]) +=
                        cv::Vec3f(values[i], values[i], values[i]);
            }
        }
#endif

    }

    ///
    /// Laplacian region generation
    ///

    std::vector<std::pair<cv::Mat, cv::Rect>> lap_region_mask(lap_regions_scaled.size());
    std::vector<cv::Mat> distance_maps(lap_regions_scaled.size());
    std::vector<int> areas(lap_regions_scaled.size(), 0);

    for (int i = 0; i < max_threads; ++i)
    {
        threads[i] = std::thread([&](int g)
                                 {
                                     for (int r = g; r < lap_regions_scaled.size(); r += max_threads)
                                     {
                                         lap_region_mask[r] = regions[g]->lapRegion(r, lap_regions_scaled[r]);
                                     }
                                 },
                                 i);
    }

    for (size_t i = 0; i < threads.size(); ++i)
    {
        threads[i].join();
    }

    tbb::parallel_for(0, lap_regions_scaled.size(), [&](int r)
    {
        if (lap_region_mask[r].second.x > 0 && lap_region_mask[r].second.y > 0)
        {
            cv::Mat region_mask_tmp = cv::Mat::zeros(lap_region_mask[r].first.rows + 2,
                                                     lap_region_mask[r].first.cols + 2,
                                                     CV_8UC1);

            lap_region_mask[r].first.copyTo(
                    region_mask_tmp(
                            cv::Rect(1, 1, lap_region_mask[r].first.cols, lap_region_mask[r].first.rows)));

            cv::Mat distance_map_tmp = cv::Mat::zeros(region_mask_tmp.rows, region_mask_tmp.cols, CV_32FC1);
            cv::distanceTransform(region_mask_tmp, distance_map_tmp, CV_DIST_L1, 3);

            distance_maps[r] = distance_map_tmp(
                    cv::Rect(1, 1, lap_region_mask[r].first.cols, lap_region_mask[r].first.rows));

            for (int i = 0; i < lap_region_mask[r].second.height; ++i)
            {
                for (int j = 0; j < lap_region_mask[r].second.width; ++j)
                {
                    if (lap_region_mask[r].first.at<uchar>(i, j) != 0)
                    {
                        if (region->is_inner_of_a_region(lap_region_mask[r].second.y + i,
                                                         lap_region_mask[r].second.x + j))
                        {
                            areas[r]++;
                        }
                        else
                        {
                            lap_region_mask[r].first.at<uchar>(i, j) = 0;
                        }
                    }
                }
            }
        }
    });

    for (int r = 0; r < lap_regions_scaled.size(); ++r)
    {
        if (lap_region_mask[r].second.x > 0 && lap_region_mask[r].second.y > 0)
        {
            float max_val = -1.0f;

            for (int i = 0; i < distance_maps[r].rows; i++)
            {
                for (int j = 0; j < distance_maps[r].cols; j++)
                {
                    if (max_val < distance_maps[r].at<float>(i, j))
                    {
                        max_val = distance_maps[r].at<float>(i, j);
                    }
                }
            }

            std::vector<CPoint2i> pt_in;
            std::vector<CPoint2i> pt_out;

            for (int i = 0; i < lap_region_mask[r].first.rows; ++i)
            {
                for (int j = 0; j < lap_region_mask[r].first.cols; ++j)
                {
                    if (lap_region_mask[r].first.at<uchar>(i, j) != 0)
                    {
                        double d = distance_maps[r].at<float>(i, j) / max_val;

#ifndef QUADTREE_VORONOI_OUTPUT
                        if (d < 0.05 || (distance_maps[r].at<float>(i, j) <= 1.0f && pt_out.size() < areas[r] - 1))
                        {
                            pt_out.push_back(CPoint2i(lap_region_mask[r].second.y + i, lap_region_mask[r].second.x + j));
                        }
                        else
                        {
                            pt_in.push_back(CPoint2i(lap_region_mask[r].second.y + i, lap_region_mask[r].second.x + j));
                        }
#else
                        pt_in.push_back(
                                CPoint2i(lap_region_mask[r].second.y + i, lap_region_mask[r].second.x + j));
#endif
                    }
                }
            }

            // Laplacian value

            cv::Vec3f lap_in = 20.0f / pt_in.size() * lap_regions_control_parameters[r].first;
            cv::Vec3f lap_out = -(float)pt_in.size() / pt_out.size()*lap_in;

            cv::Vec3f sum(0, 0, 0);

            for (size_t i = 0; i < pt_in.size(); ++i)
            {
                laplacian_image.at<cv::Vec3f>(pt_in[i][0], pt_in[i][1]) += lap_in;

#ifdef QUADTREE_VORONOI_OUTPUT
                the_laplacian_image.at<cv::Vec3f>(pt_in[i][0], pt_in[i][1]) += lap_in;
#endif
            }

            for (size_t i = 0; i < pt_out.size(); ++i)
            {
                laplacian_image.at<cv::Vec3f>(pt_out[i][0], pt_out[i][1]) += lap_out;
            }

            cv::Vec3f offset = 20.0f / areas[r] * lap_regions_control_parameters[r].second;

            for (int i = 0; i < lap_region_mask[r].first.rows; ++i)
            {
                for (int j = 0; j < lap_region_mask[r].first.cols; ++j)
                {
                    if (lap_region_mask[r].first.at<uchar>(i, j) != 0)
                    {
                        laplacian_image.at<cv::Vec3f>(lap_region_mask[r].second.y + i, lap_region_mask[r].second.x + j) += offset;

#ifdef QUADTREE_VORONOI_OUTPUT
                        the_laplacian_image.at<cv::Vec3f>(lap_region_mask[r].second.y + i, lap_region_mask[r].second.x + j) += offset;
#endif
                    }
                }
            }
        }
    }

#ifdef QUADTREE_VORONOI_OUTPUT
    std::vector<std::vector<CPoint2i> > points =
            flood_fill<cv::Vec3f>(the_laplacian_image, cv::Vec3f(0, 0, 0), 8, 1, *region);
    cv::Mat tmp(the_laplacian_image.size(), CV_8UC3);
    tmp.setTo(cv::Vec3b(255, 255, 255));

    for (size_t i = 0; i < points.size(); ++i)
    {
        cv::Vec3b c(rand() % 256, rand() % 256, rand() % 256);

        for (size_t j = 0; j < points[i].size(); ++j)
        {
            tmp.at<cv::Vec3b>(points[i][j][0], points[i][j][1]) = c;
        }
    }

    imwrite("./resultImg/laplacian_cell.bmp", tmp);
    tmp.release();
#endif

    return QPair<Region *, cv::Mat>(region.get(), laplacian_image);
}

void PVG::evaluation(Region * region, const cv::Mat laplacian_image, int n_rings)
{
    ///
    /// process enlarged image
    ///

    std::vector<CPoint2d> end_points;

    if (scaleFactor != 1.0)
    {
        region->labels_generation();

        BoundingBox<double> ROI;
        ROI.row = w00.y() / scaleFactor;
        ROI.col = w00.x() / scaleFactor;
        ROI.width = size.width() / scaleFactor;
        ROI.height = size.height() / scaleFactor;

        zoomIn(scaleFactor, w00);

        if (regions[0]->w() == size.width() && regions[0]->h() == size.height())
        {
            for (size_t i = 0; i < regions.size(); ++i)
            {
                regions[i]->set_scale(scaleFactor);
                regions[i]->set_w00(cv::Vec2i(w00.y(), w00.x()));
            }
        }
        else
        {
            for (size_t i = 0; i < regions.size(); ++i)
            {
                regions[i].reset(
                        new CRegionFZ(
                                size.height(),
                                size.width(),
                                QVector<SQ_Stroke>(),
                                scaleFactor,
                                cv::Vec2i(w00.y(), w00.x())));
            }
        }

        std::vector<std::vector<std::vector<std::pair<cv::Vec2i, cv::Vec3f> > > >
                points_vector(lap_edges_scaled.size());

        std::vector<std::thread> threads(max_threads);

        for (int i = 0; i < max_threads; i++)
        {
            threads[i] = std::thread(
                    [&](int g)
                    {
                         for (int r = g; r < lap_edges_scaled.size(); r += max_threads)
                         {
                             points_vector[r] = regions[g]->lapEdge(r, lap_edges_scaled[r]);
                         }
                    },
                    i);
        }

        for (auto i = 0; i < threads.size(); ++i)
        {
            threads[i].join();
        }

        CRegionFZ scaled_reg(
                size.height(),
                size.width(),
                m_strokes_scaled,
                scaleFactor,
                cv::Vec2i(w00.y(), w00.x()));

        scaled_reg.boundary(false);

        region->set_enlarged_mask(scaled_reg.get_sideMask(), scaleFactor, ROI, points_vector);

        for (int i = 0; i < m_strokes.size(); ++i)
        {
            if (m_strokes[i].s_mode == SQ_Stroke::OPEN)
            {
                QPointF p = m_strokes[i].s_points.front();
                end_points.push_back(CPoint2d(p.y(), p.x()));
                p = m_strokes[i].s_points.back();
                end_points.push_back(CPoint2d(p.y(), p.x()));
            }
        }

        for (int i = 0; i < lap_edges.size(); ++i)
        {
            if (lap_edges[i].s_mode == SQ_Stroke::OPEN)
            {
                QPointF p = lap_edges[i].s_points.front();
                end_points.push_back(CPoint2d(p.y(), p.x()));
                p = lap_edges[i].s_points.back();
                end_points.push_back(CPoint2d(p.y(), p.x()));
            }
        }

        std::vector<SQ_Stroke> strokes(m_strokes_scaled.begin(), m_strokes_scaled.end());
        strokes.insert(strokes.end(), lap_edges_scaled.begin(), lap_edges_scaled.end());
        for (size_t i = 0; i < strokes.size(); ++i)
        {
            strokes[i].translation(QPointF(-w00.x(), -w00.y()));
        }

        region->add_SQ_strokes(strokes);
    }

    cv::Mat convert_to_laplacian_mask = cv::Mat();

    // Poisson solver
    PoissonSolver poissonSolver(
        cv::Size(size.width(), size.height()),
        laplacian_image,
        *region,
        scaleFactor,
        CPoint2d(w00.y() / scaleFactor, w00.x() / scaleFactor),
        end_points,
        n_rings,
        convert_to_laplacian_mask);

    result = poissonSolver.get_result_image();
}

#ifdef QUADTREE_VORONOI_OUTPUT
template <class T>
static std::vector<std::vector<CPoint2i> >
flood_fill(const cv::Mat & image, const T & zero, int n_neighbors, int region_id, const Region & region)
{
    cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
    std::vector<std::vector<CPoint2i> > region_points;

    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
        {
            if (mask.at<uchar>(i, j) == 0 && region.region_id(i, j) == region_id)
            {
                std::queue<CPoint2i> pt_list;
                region_points.emplace_back(std::vector<CPoint2i>(1, CPoint2i(i, j)));
                mask.at<uchar>(i, j) = 255;
                pt_list.push(CPoint2i(i, j));

                while (!pt_list.empty())
                {
                    CPoint2i p = pt_list.front();
                    pt_list.pop();

                    for (int k = 0; k < n_neighbors; ++k)
                    {
                        CPoint2i pt = p + trans[k];

                        if (pt[0] >= 0 && pt[0] < mask.rows && pt[1] >= 0 && pt[1] < mask.cols)
                        {
                            if (mask.at<uchar>(pt[0], pt[1]) == 0 &&
                                image.at<T>(pt[0], pt[1]) == image.at<T>(p[0], p[1]) &&
                                region.region_id(pt[0], pt[1]) == region_id)
                            {
                                region_points.back().push_back(pt);
                                mask.at<uchar>(pt[0], pt[1]) = 255;
                                pt_list.push(pt);
                            }
                        }
                    }
                }
            }
        }
    }

    return region_points;
}
#endif

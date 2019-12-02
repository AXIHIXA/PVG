//
// Created by ax on 11/30/19.
//

#include "auxiliary.h"
#include "debug.h"
#include "global.h"
#include "PVGBuffer.h"
#include <ParseColor.hpp>
#include <QFile>
#include <tbb/tbb.h>
#include <thread>


PVGBuffer::PVGBuffer(double sf, double lsf, const QPoint & w)
{
    // cpu capacity
    max_threads = std::min(std::thread::hardware_concurrency(), 16U);
    regions.resize(max_threads);

    // resize magnifier
    scaleFactor = sf;
    lastScaleFactor = lsf;
    w00 = w;
}

bool PVGBuffer::open(const QString & filename, double scale)
{
    if (filename.isEmpty())
    {
        st_error("empty filename");
        return false;
    }

    ///
    /// update scale factor
    ///

    lastScaleFactor = scaleFactor;
    scaleFactor = scale;

    ///
    /// read compressed file
    ///

    QByteArray byteArray;
    QFile fin(filename);

    if (!fin.open(QIODevice::ReadOnly))
    {
        st_error("reading file error");
        return false;
    }

    byteArray = fin.readAll();
    fin.close();

    if (byteArray.isEmpty())
    {
        st_error("reading file error");
        return false;
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
    std::vector< int > num = pvgaux::split_to_num< int >(dim);

    if (num.empty())
    {
        return false;
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
    tinyxml2::XMLElement * lap_edges_control_parameters_ele = rootNode->FirstChildElement(
            "lap_edges_control_parameters");
    std::vector< float > num1 = pvgaux::split_to_num< float >(lap_edges_control_parameters_ele->GetText());

    for (size_t i = 0; i < num1.size(); i++)
    {
        lap_edges_control_parameters.push_back(num1[i]);
    }

    ///
    /// read lap_regions_control_parameters
    ///

    lap_regions_control_parameters.clear();
    tinyxml2::XMLElement * lap_regions_control_parameters_ele = rootNode->FirstChildElement(
            "lap_regions_control_parameters");
    std::vector< float > num2 = pvgaux::split_to_num< float >(lap_regions_control_parameters_ele->GetText());

    if (num2.size() == 6 * lap_regions.size())
    {
        for (size_t i = 0; i < num2.size(); i += 6)
        {
            lap_regions_control_parameters.push_back(qMakePair(cv::Vec3f(num2[i], num2[i + 1], num2[i + 2]),
                                                               cv::Vec3f(num2[i + 3], num2[i + 4], num2[i + 5])));
        }
    }
    else
    {
        for (size_t i = 0; i < num2.size(); i += 2)
        {
            lap_regions_control_parameters.push_back(
                    qMakePair(cv::Vec3f(num2[i], num2[i], num2[i]), cv::Vec3f(num2[i + 1], num2[i + 1], num2[i + 1])));
        }
    }

    ///
    /// read lap_points
    ///

    lap_points.clear();
    tinyxml2::XMLElement * lap_points_ele = rootNode->FirstChildElement("lap_points");

    if (lap_points_ele != nullptr)
    {
        std::vector< double > num3 = pvgaux::split_to_num< double >(lap_points_ele->GetText());

        for (size_t i = 0; i < num3.size(); i += 5)
        {
            lap_points.push_back({CPoint2i((int) num3[i], (int) num3[i + 1]),
                                  cv::Vec3f((float) num3[i + 2], (float) num3[i + 3], (float) num3[i + 4])});
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

    discretization();

    return true;
}

SQ_Stroke PVGBuffer::parseStroke(const tinyxml2::XMLElement * SQ_Stroke_ele, double scale, bool parseProperty)
{
    SQ_Stroke stroke;

    const tinyxml2::XMLElement * s_points_ele = SQ_Stroke_ele->FirstChildElement("s_points");
    const char * str1 = s_points_ele->GetText();
    std::vector< double > num1 = pvgaux::split_to_num< double >(str1);

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
            std::vector< int > num2 = pvgaux::split_to_num< int >(str2);

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
    std::vector< int > num3 = pvgaux::split_to_num< int >(str3);
    stroke.s_mode = SQ_Stroke::StrokeMode(num3[0]);

    const tinyxml2::XMLElement * s_mdmode_ele = SQ_Stroke_ele->FirstChildElement("s_mdmode");
    const char * str4 = s_mdmode_ele->GetText();
    std::vector< int > num4 = pvgaux::split_to_num< int >(str4);
    stroke.s_mdmode = SQ_Stroke::MDMode(num4[0]);

    return stroke;
}

void PVGBuffer::discretization()
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

    //static std::unique_ptr<Region> region;
    static cv::Mat laplacian_image;

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

             //region.reset(new Region(region_mask, side_mask));

             overwrite_id.clear();

             for (int i = 0; i < lap_points.size(); i++)
             {
                 int id = abs(side_mask.at<int>(lap_points[i].pt[0], lap_points[i].pt[1]));

                 if (id != 0)
                 {
                     id -= 1;

                     if (overwrite_id.indexOf(id) == -1)
                     {
                         overwrite_id.push_back(id);
                     }
                 }
             }
         }
     );
}

void PVGBuffer::zoomIn(double scale, const QPoint& cur_w00)
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
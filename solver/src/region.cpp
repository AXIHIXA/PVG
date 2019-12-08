#include "region.h"
#include <tbb/tbb.h>


using namespace std;
using namespace cv;


const CPoint2i Region::offset[] = \
{
    CPoint2i(0, 1),
    CPoint2i(0, -1),
    CPoint2i(1, 0),
    CPoint2i(-1, 0),
    CPoint2i(-1, -1),
    CPoint2i(-1, 1),
    CPoint2i(1, -1),
    CPoint2i(1, 1)
};

Region::Region(const cv::Mat& mask, const cv::Mat& sideMask) :mask(mask)
{
	sideMask.copyTo(side_mask);

	number_of_regions = -1;

	for (int i = 0; i < mask.rows; ++i)
	{
		for (int j = 0; j < mask.cols; ++j)
		{
			int id = abs(mask.at<int>(i, j));

			if (id > number_of_regions)
            {
			    number_of_regions = id;
            }
		}
	}

	vector<CPoint2i> min(number_of_regions, CPoint2i(std::max(mask.rows, mask.cols), std::max(mask.rows, mask.cols)));
	vector<CPoint2i> max(number_of_regions, CPoint2i(0, 0));

	for (int i = 0; i < mask.rows; ++i)
	{
		for (int j = 0; j < mask.cols; ++j)
		{
			int id = abs(mask.at<int>(i, j));
			if (id != 0)
			{
				--id;
				if (i < min[id][0]) min[id][0] = i;
				if (i > max[id][0]) max[id][0] = i;
				if (j < min[id][1]) min[id][1] = j;
				if (j > max[id][1]) max[id][1] = j;
			}
		}
	}

	rects.resize(number_of_regions);

	for (size_t i = 0; i < rects.size(); ++i)
	{
		rects[i].row = min[i][0];
		rects[i].col = min[i][1];
		rects[i].height = max[i][0] - min[i][0] + 1;
		rects[i].width = max[i][1] - min[i][1] + 1;
	}

	side_mask_index = -1;
	enlarged_side_mask_index = -1;

	for (int i = 0; i < side_mask.rows; ++i)
	{
		for (int j = 0; j < side_mask.cols; ++j)
		{

			int id = abs(side_mask.at<int>(i, j));
			if (id > side_mask_index) side_mask_index = id;
		}
	}

	++side_mask_index;

	enlarged_side_mask_index = side_mask_index;
	enlarged_side_mask_index_backup = enlarged_side_mask_index;

	mono_mask = Mat::zeros(mask.size(), CV_8UC1);

	for (int i = 0; i < mask.rows; ++i)
	{
		for (int j = 0; j < mask.cols; ++j)
		{
			if (mask.at<int>(i, j) < 0 && sideMask.at<int>(i, j) == 0)
			{
				mono_mask.at<uchar>(i, j) = 255;
			}
		}
	}

	crossing_region.resize(boost::extents[mask.rows][mask.cols]);

    tbb::parallel_for(0, side_mask.rows, [&](int i)
    {
        for (int j = 0; j < side_mask.cols; ++j)
        {
            if (side_mask.at<int>(i, j) != 0)
            {
                crossing_region[i][j].push_back(side_mask.at<int>(i, j));
            }
        }
    });
}

void Region::labels_generation()
{
	if (near_pts.size() != 0)
    {
	    return;
    }

	vector<vector<CPoint2i>> points_left(side_mask_index);
	vector<vector<CPoint2i>> points_right(side_mask_index);

	for (size_t i = 0; i < crossing_region.shape()[0]; ++i)
	{
		for (size_t j = 0; j < crossing_region.shape()[1]; ++j)
		{
			for (size_t k = 0; k < crossing_region[i][j].size(); ++k)
			{
				int id = crossing_region[i][j][k];
				if (id < 0)
					points_left[-id].push_back(CPoint2i((int)i, (int)j));
				else if (id > 0)
					points_right[id].push_back(CPoint2i((int)i, (int)j));
			}
		}
	}

	vector<int> pts_id;
	vector<vector<CPoint2i>> points;

	for (int i = 0; i < (int)points_left.size(); ++i)
	{
		pts_id.push_back(-i);
		points.push_back(points_left[i]);
	}

	for (int i = 0; i < (int)points_right.size(); ++i)
	{
		pts_id.push_back(i);
		points.push_back(points_right[i]);
	}

	for (int m = 0; m < 2; ++m)
	{
		for (size_t n = 0; n < points.size(); ++n)
		{
			vector<CPoint2i> added;

			for (size_t i = 0; i < points[n].size(); ++i)
			{
				for (int k = 0; k < 4; ++k)
				{
					CPoint2i p = points[n][i] + offset[k];

					if (p[0] >= 0 && p[0] < side_mask.rows && p[1] >= 0 && p[1] < side_mask.cols)
					{
						if (side_mask.at<int>(p[0], p[1]) == 0)
						{
							vector<int>& v = crossing_region[p[0]][p[1]];

							if (find(v.begin(), v.end(), -pts_id[n]) == v.end() && find(v.begin(), v.end(), pts_id[n]) == v.end())
							{
								v.push_back(pts_id[n]);
								added.push_back(p);
							}
						}
					}
				}
			}

			points[n] = added;
		}
	}

	for (size_t i = 0; i < crossing_region.shape()[0]; ++i)
	{
		for (size_t j = 0; j < crossing_region.shape()[1]; ++j)
		{
			if (crossing_region[i][j].size() <= 1) crossing_region[i][j].clear();
			else sort(crossing_region[i][j].begin(), crossing_region[i][j].end());
		}
	}

	Mat tmp;
	side_mask.copyTo(tmp);

	for (int i = 0; i < side_mask.rows; ++i)
	{
		for (int j = 0; j < side_mask.cols; ++j)
		{
			if (tmp.at<int>(i, j) != 0 && !is_end_point(tmp, CPoint2i(i, j)))
			{
				for (int k = 0; k < 4; ++k)
				{
					CPoint2i p(i + offset[k][0], j + offset[k][1]);

					if (p[0] >= 0 && p[0] < side_mask.rows && p[1] >= 0 && p[1] < side_mask.cols)
					{
						if (side_mask.at<int>(p[0], p[1]) == 0 &&
						        this->region_id(p) == this->region_id(i, j) &&
						        !this->is_boundary(p[0], p[1]))
						{
							side_mask.at<int>(p[0], p[1]) = side_mask.at<int>(i, j);
						}
					}
				}
			}
		}
	}

	near_pts.resize(boost::extents[side_mask.rows][side_mask.cols]);
	vector<vector<CPoint2i>> pt_list_left(side_mask_index);
	vector<vector<CPoint2i>> pt_list_right(side_mask_index);

	for (int i = 0; i < side_mask.rows; ++i)
	{
		for (int j = 0; j < side_mask.cols; ++j)
		{
			int n = side_mask.at<int>(i, j);

			if (n < 0)
			{
				pt_list_left[-n].push_back(CPoint2i(i, j));
				near_pts[i][j].emplace_back(make_pair(n, CPoint2i(i, j)));
			}
			else if (n > 0)
			{
				pt_list_right[n].push_back(CPoint2i(i, j));
				near_pts[i][j].emplace_back(make_pair(n, CPoint2i(i, j)));
			}
		}
	}

	Mat mask_tmp(side_mask.rows, side_mask.cols, CV_32SC2);

	for (int i = 0; i < (int)pt_list_left.size(); ++i)
	{
		mask_tmp.setTo(Vec2i(-1, -1));

		for (int j = 0; j < pt_list_left[i].size(); ++j)
		{
			mask_tmp.at<Vec2i>(pt_list_left[i][j][0], pt_list_left[i][j][1]) = pt_list_left[i][j];
		}

		for (int n = 0; n < 2; ++n)
		{
			vector<CPoint2i> added;

			for (size_t j = 0; j < pt_list_left[i].size(); ++j)
			{
				for (int k = 0; k < 4; ++k)
				{
					CPoint2i pt = pt_list_left[i][j] + offset[k];

					if (pt[0] >= 0 && pt[0] < side_mask.rows && pt[1] >= 0 && pt[1] < side_mask.cols)
					{
						if (mask_tmp.at<Vec2i>(pt[0], pt[1]) == Vec2i(-1, -1) && is_critical(pt))
						{
							near_pts[pt[0]][pt[1]].emplace_back(
							        make_pair(-i,
							                mask_tmp.at<Vec2i>(
							                        pt_list_left[i][j][0],
							                        pt_list_left[i][j][1])));
							added.push_back(pt);
							mask_tmp.at<Vec2i>(pt[0], pt[1]) = mask_tmp.at<Vec2i>(pt_list_left[i][j][0], pt_list_left[i][j][1]);
						}
					}
				}
			}
			pt_list_left[i] = added;
		}
	}

	for (int i = 0; i < (int)pt_list_right.size(); ++i)
	{
		mask_tmp.setTo(Vec2i(-1, -1));

		for (int j = 0; j < pt_list_right[i].size(); ++j)
		{
			mask_tmp.at<Vec2i>(pt_list_right[i][j][0], pt_list_right[i][j][1]) = pt_list_right[i][j];
		}

		for (int n = 0; n < 2; ++n)
		{
			vector<CPoint2i> added;

			for (size_t j = 0; j < pt_list_right[i].size(); ++j)
			{
				for (int k = 0; k < 4; ++k)
				{
					CPoint2i pt = pt_list_right[i][j] + offset[k];

					if (pt[0] >= 0 && pt[0] < side_mask.rows && pt[1] >= 0 && pt[1] < side_mask.cols)
					{
						if (mask_tmp.at<Vec2i>(pt[0], pt[1]) == Vec2i(-1, -1) && is_critical(pt))
						{
							near_pts[pt[0]][pt[1]].push_back(make_pair(i, mask_tmp.at<Vec2i>(pt_list_right[i][j][0], pt_list_right[i][j][1])));
							added.push_back(pt);
							mask_tmp.at<Vec2i>(pt[0], pt[1]) = mask_tmp.at<Vec2i>(pt_list_right[i][j][0], pt_list_right[i][j][1]);
						}
					}
				}
			}

			pt_list_right[i] = added;
		}
	}
}

CPoint2i Region::find_closest_pixel(const CPoint2f& p, int id) const
{
	CPoint2i pi((int)p[0], (int)p[1]);

	if (pi[0] >= 0 && pi[0] < near_pts.shape()[0] && pi[1] >= 0 && pi[1] < near_pts.shape()[1])
	{
		for (size_t i = 0; i < near_pts[pi[0]][pi[1]].size(); ++i)
		{
			if (near_pts[pi[0]][pi[1]][i].first == id) return near_pts[pi[0]][pi[1]][i].second;
		}
	}

	return pi;
}

CPoint2f Region::to_source_pt(const CPoint2f& p) const
{
	CPoint2f pt;
	pt[0] = (float)(ROI.row + p[0] / scale);
	pt[1] = (float)(ROI.col + p[1] / scale);
	return pt;
}

CPoint2f Region::to_scaled_pt(const CPoint2f& p) const
{
	CPoint2f pt;
	pt[0] = (float)((p[0] - ROI.row)*scale);
	pt[1] = (float)((p[1] - ROI.col)*scale);
	return pt;
}

void Region::set_side_source(const std::vector<std::vector<std::pair<cv::Vec2i, cv::Vec3f>>> & pts)
{
	// Xi Han: mark points on lap edge that:
	// 1. in DC region (regionMask > 0)
	// 2. not on DC curve (sideMask == 0)
	// case I: vecMask = 1 (sample pt DC): -side_mask_index
	// case II: vecMask = 2 (neibor pt DC): side_mask_index

    for (size_t i = 0; i < pts[0].size(); i++)
	{
		if (side_mask.at<int>(pts[0][i].first[0], pts[0][i].first[1]) == 0)
		{
			side_mask.at<int>(pts[0][i].first[0], pts[0][i].first[1]) = -side_mask_index;
			crossing_region[pts[0][i].first[0]][pts[0][i].first[1]].push_back(-side_mask_index);
		}
	}

	for (size_t i = 0; i < pts[1].size(); i++)
	{
		if (side_mask.at<int>(pts[1][i].first[0], pts[1][i].first[1]) == 0)
		{
			side_mask.at<int>(pts[1][i].first[0], pts[1][i].first[1]) = side_mask_index;
			crossing_region[pts[1][i].first[0]][pts[1][i].first[1]].push_back(side_mask_index);
		}
	}

	++side_mask_index;
}

void Region::set_side_enlarged(const vector<vector<pair<Vec2i, Vec3f>>>& pts)
{
	for (int i = 0; i < pts[0].size(); ++i)
	{
		if (enlarged_side_mask.at<int>(pts[0][i].first[0], pts[0][i].first[1]) == 0)
		{
			enlarged_side_mask.at<int>(pts[0][i].first[0], pts[0][i].first[1]) = -enlarged_side_mask_index;
		}
	}

	for (int i = 0; i < pts[1].size(); ++i)
	{
		if (enlarged_side_mask.at<int>(pts[1][i].first[0], pts[1][i].first[1]) == 0)
		{
			enlarged_side_mask.at<int>(pts[1][i].first[0], pts[1][i].first[1]) = enlarged_side_mask_index;
		}
	}

	++enlarged_side_mask_index;
}

void Region::set_enlarged_mask(const Mat& enlarged_mask, double scale, const BoundingBox<double>& ROI, const vector<vector<vector<pair<Vec2i, Vec3f>>>>& points_vector)
{
	enlarged_side_mask_index = enlarged_side_mask_index_backup;
	enlarged_side_mask = enlarged_mask;

	for (size_t i = 0; i < points_vector.size(); ++i)
    {
	    set_side_enlarged(points_vector[i]);
    }

	this->scale = scale;
	this->ROI = ROI;

	enlarged_side_mask.copyTo(enlarged_side_mask_source);

	// fill left and right region
	bfs<int>(enlarged_side_mask, (int)ceil(4 * scale));

	enlarged_crossing_region.resize(boost::extents[0][0]);
	enlarged_crossing_region.resize(boost::extents[enlarged_side_mask.rows][enlarged_side_mask.cols]);

	int round = (int)ceil(4 * scale);

	vector<vector<CPoint2i>> points_left(enlarged_side_mask_index);
	vector<vector<CPoint2i>> points_right(enlarged_side_mask_index);

	for (int i = 0; i < enlarged_side_mask_source.rows; ++i)
	{
		for (int j = 0; j < enlarged_side_mask_source.cols; ++j)
		{
			int id = enlarged_side_mask_source.at<int>(i, j);

			if (id < 0)
            {
			    points_left[-id].push_back(CPoint2i(i, j));
            }
			else if (id > 0)
            {
			    points_right[id].push_back(CPoint2i(i, j));
            }
		}
	}

	vector<int> pts_id;
	vector<vector<CPoint2i>> points;

	for (int i = 0; i < (int)points_left.size(); ++i)
	{
		pts_id.push_back(-i);
		points.push_back(points_left[i]);
	}

	for (int i = 0; i < (int)points_right.size(); ++i)
	{
		pts_id.push_back(i);
		points.push_back(points_right[i]);
	}

	for (size_t n = 0; n < points.size(); ++n)
	{
		for (size_t i = 0; i < points[n].size(); ++i)
		{
			enlarged_crossing_region[points[n][i][0]][points[n][i][1]].push_back(pts_id[n]);
		}
	}

	for (int m = 0; m < round; ++m)
	{
		for (size_t n = 0; n < points.size(); ++n)
		{
			vector<CPoint2i> added;

			for (size_t i = 0; i < points[n].size(); ++i)
			{
				for (int k = 0; k < 4; ++k)
				{
					CPoint2i p = points[n][i] + offset[k];

					if (p[0] >= 0 && p[0] < enlarged_side_mask.rows && p[1] >= 0 && p[1] < enlarged_side_mask.cols)
					{
						if (enlarged_side_mask_source.at<int>(p[0], p[1]) == 0)
						{
							vector<int>& v = enlarged_crossing_region[p[0]][p[1]];

							if (find(v.begin(), v.end(), -pts_id[n]) == v.end() &&
							    find(v.begin(), v.end(), pts_id[n]) == v.end())
							{
								v.push_back(pts_id[n]);
								added.push_back(p);
							}
						}
					}
				}
			}

			points[n] = added;
		}
	}

	for (size_t i = 0; i < enlarged_crossing_region.shape()[0]; ++i)
	{
		for (size_t j = 0; j < enlarged_crossing_region.shape()[1]; ++j)
		{
			if (enlarged_crossing_region[i][j].size() <= 1) enlarged_crossing_region[i][j].clear();
			else sort(enlarged_crossing_region[i][j].begin(), enlarged_crossing_region[i][j].end());
		}
	}
}

bool Region::is_end_point(const cv::Mat& mask, const CPoint2i& p)
{
	int count = 0;
	int id = mask.at<int>(p[0], p[1]);

	for (int i = 0; i < 8; ++i)
	{
		CPoint2i pt = p + offset[i];

		if (pt[0] >= 0 && pt[0] < mask.rows && pt[1] >= 0 && pt[1] < mask.cols)
		{
			if (mask.at<int>(pt[0], pt[1]) == id) ++count;
		}
	}

	return count == 1;
}

PointType Region::type(int region_id, const CPoint2i& p) const
{
	++region_id;

	if (p[0] >= 0 && p[0] < mask.rows && p[1] >= 0 && p[1] < mask.cols)
	{
		if (abs(mask.at<int>(p[0], p[1])) == region_id)
		{
			if (mask.at<int>(p[0], p[1]) < 0)
            {
			    return BOUNDARY;
            }
			else
            {
			    return INNER;
            }
		}
	}

	return OUTER;
}

PointType Region::type(int region_id, int ln, int col) const
{
	++region_id;

	if (ln >= 0 && ln < mask.rows && col >= 0 && col < mask.cols)
	{
		if (abs(mask.at<int>(ln, col)) == region_id)
		{
			if (mask.at<int>(ln, col) < 0)
            {
			    return BOUNDARY;
            }
			else
            {
			    return INNER;
            }
		}
	}

	return OUTER;
}

int Region::region_id(const CPoint2i& p) const
{
	if (p[0] >= 0 && p[1] >= 0 && p[0] < mask.rows && p[1] < mask.cols)
	{
		return abs(mask.at<int>(p[0], p[1])) - 1;
	}
	else
    {
	    return -1;
    }
}

int Region::region_id(int ln, int col) const
{
	if (ln >= 0 && col >= 0 && ln < mask.rows && col < mask.cols)
	{
		return abs(mask.at<int>(ln, col)) - 1;
	}
	else
    {
	    return -2;
    }
}

void Region::add_SQ_strokes(const std::vector<SQ_Stroke>& strokes)
{
	this->strokes = strokes;

	end_points_mask.resize(boost::extents[0][0]);
	end_points_mask.resize(boost::extents[enlarged_side_mask.rows][enlarged_side_mask.cols]);

	int r = static_cast<int>(3 * scale);

	BoundingBox<int> image_box(0, 0, enlarged_side_mask.rows, enlarged_side_mask.cols);

	for (int i = 0; i < strokes.size(); ++i)
	{
		QPointF p1 = strokes[i].s_points.front();
		BoundingBox<int> box1(p1.y() - r, p1.x() - r, 2 * r, 2 * r);
		box1.intersection_boundingbox(image_box);

		for (int x = 0; x < box1.width; ++x)
		{
			for (int y = 0; y < box1.height; ++y)
			{
				QPoint pt(box1.col + x, box1.row + y);
				if (pt.x() >= 0 && pt.x() < enlarged_side_mask.cols && pt.y() >= 0 && pt.y() < enlarged_side_mask.rows)
				{
					int id = abs(enlarged_side_mask.at<int>(pt.y(), pt.x()));
					if (id == 0 || id == i + 1) end_points_mask[pt.y()][pt.x()].push_back(i);
				}
			}
		}

		QPointF p2 = strokes[i].s_points.back();
		BoundingBox<int> box2(p2.y() - r, p2.x() - r, 2 * r, 2 * r);
		box2.intersection_boundingbox(image_box);

		for (int x = 0; x < box2.width; ++x)
		{
			for (int y = 0; y < box2.height; ++y)
			{
				QPoint pt(box2.col + x, box2.row + y);

				if (0 <= pt.x() && pt.x() < enlarged_side_mask.cols &&
                    0 <= pt.y() && pt.y() < enlarged_side_mask.rows)
				{
					int id = abs(enlarged_side_mask.at<int>(pt.y(), pt.x()));

					if (id == 0 || id == i + 1)
                    {
					    end_points_mask[pt.y()][pt.x()].emplace_back(i);
                    }
				}
			}
		}
	}
}

CPoint2f Region::get_neighbor_of_border(const CPoint2f& p) const
{
	int r = int(p[0]);
	int c = int(p[1]);

	CPoint2f pt(p);

	if (r == 0)
    {
	    pt[0] = p[0] + 1;
    }
	else if (r == side_mask.rows - 1)
    {
	    pt[0] = p[0] - 1;
    }

	if (c == 0)
    {
	    pt[1] = p[1] + 1;
    }
	else if (c == side_mask.cols - 1)
    {
	    pt[1] = p[1] - 1;
    }

	return pt;
}

CPoint2f Region::get_adjacent_image_border(const CPoint2i& p) const
{
	CPoint2i pt(p);

	if (p[0] == -1)
    {
	    pt[0] = p[0] + 1;
    }
	else if (pt[0] == side_mask.rows)
    {
	    pt[0] = p[0] - 1;
    }

	if (p[1] == -1)
    {
	    pt[1] = p[1] + 1;
    }
	else if (p[1] == side_mask.cols)
    {
	    pt[1] = p[1] - 1;
    }

	return pt;
}

int Region::get_curve_index(const CPoint2f& p) const
{
	int id = -1;
	double dis = numeric_limits<double>::infinity();

	for (int i = 0; i < end_points_mask[int(p[0])][int(p[1])].size(); ++i)
	{
		double d1, d2;
		int idx = end_points_mask[int(p[0])][int(p[1])][i];

		QPointF p1 = QPointF(p[1], p[0]) - strokes[idx].s_points.front();
		Vec3d v1(p1.y(), p1.x(), 0);
		Vec3d t1(strokes[idx].dir_f.y(), strokes[idx].dir_f.x(), 0);

		if (v1.dot(t1) > 0)
        {
		    d1 = numeric_limits<double>::infinity();
        }
		else
        {
		    d1 = norm(v1);
        }

		QPointF p2 = QPointF(p[1], p[0]) - strokes[idx].s_points.back();
		Vec3d v2(p2.y(), p2.x(), 0);
		Vec3d t2(-strokes[idx].dir_b.y(), -strokes[idx].dir_b.x(), 0);

		if (v2.dot(t2) > 0)
        {
		    d2 = numeric_limits<double>::infinity();
        }
		else
        {
		    d2 = norm(v2);
        }

		double d = min(d1, d2);

		if (d < dis)
		{
			id = idx;
			dis = d;
		}
	}

	return id + 1;
}

int Region::get_edge_id(const CPoint2f& p) const
{
	CPoint2f pt = to_scaled_pt(p);

	if (0 <= (int) pt[0] && (int) pt[0] < enlarged_side_mask.rows &&
	    0 <= (int) pt[1] && (int) pt[1] < enlarged_side_mask.cols)
	{
		int id = get_curve_index(pt);

		if (id > 0)
		{
			int n = strokes[id - 1].sideEndpoint(QPointF(pt[1], pt[0]), scale);

			if (n > 0)
            {
			    return id;
            }
			else if (n < 0)
            {
			    return -id;
            }
		}

		id = enlarged_side_mask.at<int>(static_cast<int>(pt[0]), static_cast<int>(pt[1]));

		if (id != 0)
        {
		    return id;
        }

		if (0 <= (int) p[0] && (int) p[0] < side_mask.rows &&
		    0 <= (int) p[1] && (int) p[1] < side_mask.cols)
		{
			id = abs(side_mask.at<int>(static_cast<int>(p[0]), static_cast<int>(p[1])));

			if (id > 0)
			{
				int n = strokes[id - 1].sideTangent(QPointF(pt[1], pt[0]));

				if (n > 0)
                {
				    return id;
                }
				else if (n < 0)
                {
				    return -id;
                }
			}
		}
	}
	return 0;
}

void Region::bfs(boost::multi_array<std::vector<int>, 2>& mask, int n_ring)
{
	std::vector<vector<CPoint2i>> pt_list(enlarged_side_mask_index);

	for (int i = 0; i < mask.shape()[0]; ++i)
	{
		for (int j = 0; j < mask.shape()[1]; ++j)
		{
			for (size_t k = 0; k < mask[i][j].size(); ++k)
            {
			    pt_list[mask[i][j][k]].push_back(CPoint2i(i, j));
            }
		}
	}

	Mat tmp((int)mask.shape()[0], (int)mask.shape()[1], CV_8UC1);

	for (size_t i = 0; i < pt_list.size(); ++i)
	{
		tmp.setTo(0);

		for (int j = 0; j < pt_list[i].size(); ++j)
        {
		    tmp.at<uchar>(pt_list[i][j][0], pt_list[i][j][1]) = 255;
        }

		std::vector<CPoint2i> added;

		for (int n = 0; n < n_ring; ++n)
		{
			for (int j = 0; j < pt_list[i].size(); ++j)
			{
				for (int k = 0; k < 4; ++k)
				{
					CPoint2i pt = pt_list[i][j] + offset[j];

					if (pt[0] >= 0 && pt[0] < mask.shape()[0] && pt[1] >= 0 && pt[1] < mask.shape()[1])
					{
						if (tmp.at<uchar>(pt[0], pt[1]) == 0)
						{
							tmp.at<uchar>(pt[0], pt[1]) = 255;
							mask[pt[0]][pt[1]].push_back((int)i);
							added.push_back(pt);
						}
					}
				}
			}

			pt_list[i] = added;
		}
	}
}

CPoint2i Region::find_nearby_point(const CPoint2f& p, const std::vector<int>& idx) const
{
	CPoint2i pt = (CPoint2i)p;

	for (int r = 0; r <= 4; ++r)
	{
		for (int i = -r; i < r; ++i)
		{
			int row = pt[0] + r;
			int col = pt[1] + i;

			if (row >= 0 && row < crossing_region.shape()[0] && col >= 0 && col < crossing_region.shape()[1])
			{
				if (crossing_region[row][col] == idx) return CPoint2i(row, col);
			}
		}

		for (int i = -r; i < r; ++i)
		{
			int row = pt[0] - r;
			int col = pt[1] + i;

			if (row >= 0 && row < crossing_region.shape()[0] && col >= 0 && col < crossing_region.shape()[1])
			{
				if (crossing_region[row][col] == idx) return CPoint2i(row, col);
			}
		}

		for (int i = -r; i <= r; ++i)
		{
			int row = pt[0] + i;
			int col = pt[1] + r;

			if (row >= 0 && row < crossing_region.shape()[0] && col >= 0 && col < crossing_region.shape()[1])
			{
				if (crossing_region[row][col] == idx) return CPoint2i(row, col);
			}
		}

		for (int i = -r + 1; i < r; ++i)
		{
			int row = pt[0] + i;
			int col = pt[1] - r;

			if (row >= 0 && row < crossing_region.shape()[0] && col >= 0 && col < crossing_region.shape()[1])
			{
				if (crossing_region[row][col] == idx) return CPoint2i(row, col);
			}
		}
	}

	return CPoint2i(-1, -1);
}

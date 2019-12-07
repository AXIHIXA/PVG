#include "pathstroke_auxiliary.h"
#include "Strokes.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <queue>

using namespace std;
using namespace cv;


CPoint2i trans[] =
{
    CPoint2i(-1, 0),
    CPoint2i(1, 0),
    CPoint2i(0, -1),
    CPoint2i(0, 1),
    CPoint2i(1, 1),
    CPoint2i(1, -1),
    CPoint2i(-1, 1),
    CPoint2i(-1, -1)
};

cv::Vec3f evaluate_laplacian(const cv::Mat& image, const CPoint2i& p)
{
	using namespace cv;
	Vec3f lap(0, 0, 0);
	for (int i = 0; i < 4; ++i)
	{
		CPoint2i pt = p + trans[i];
		if (pt[0] >= 0 && pt[0] < image.rows && pt[1] >= 0 && pt[1] < image.cols)
		{
			lap += image.at<Vec3f>(p[0], p[1]) - image.at<Vec3f>(pt[0], pt[1]);
		}
	}
	return lap;
}

vector<vector<CPoint2i>> extract_curves(const cv::Mat& mono_image)
{
	using namespace cv;
	vector<vector<CPoint2i>> all_points;
	Mat mask = Mat::zeros(mono_image.size(), CV_8UC1);

	for (int ln = 0; ln < mono_image.rows; ++ln)
	{
		for (int col = 0; col < mono_image.cols; ++col)
		{
			if (mono_image.at<uchar>(ln, col) != 0 && mask.at<uchar>(ln, col) == 0)
			{
				mask.at<uchar>(ln, col) = 255;
				vector<CPoint2i> points;
				points.push_back(CPoint2i(ln, col));

				while (true)
				{
					vector<CPoint2i> added;
					added.push_back(points.back());
					bool flag = false;
					for (int n = 1; n <= 1; ++n)
					{
						vector<CPoint2i> new_added;
						for (size_t i = 0; i < added.size(); ++i)
						{
							for (int j = 0; j < 8; ++j)
							{
								CPoint2i pt = added[i] + trans[j];
								if (max(abs(pt[0] - points.back()[0]), abs(pt[1] - points.back()[1])) == n)
								{
									if (pt[0] >= 0 && pt[0] < mono_image.rows && pt[1] >= 0 && pt[1] < mono_image.cols)
									{
										if (mask.at<uchar>(pt[0], pt[1]) == 0)
										{
											if (mono_image.at<uchar>(pt[0], pt[1]) != 0)
											{
												flag = true;
												points.push_back(pt);
												mask.at<uchar>(pt[0], pt[1]) = 255;
												break;
											}
											else
											{
												new_added.push_back(pt);
											}
										}
									}
								}
							}
							if (flag) break;
						}
						if (flag) break;
						added = new_added;
					}
					if (!flag) break;
				}
				all_points.push_back(vector<CPoint2i>());
				for (size_t i = 0; i < points.size(); ++i)
					all_points.back().push_back(points[i]);
			}
		}
	}

	return all_points;
}

void flood_fill(cv::Mat region_mask)
{
	using namespace cv;
	Mat region_mask_copy;
	region_mask.copyTo(region_mask_copy);
	int count = 1;
	Mat mask = Mat::zeros(region_mask.size(), CV_8UC1);
	for (int i = 0; i < region_mask.rows; ++i)
	{
		for (int j = 0; j < region_mask.cols; ++j)
		{
			if (mask.at<uchar>(i, j) == 0 && region_mask.at<int>(i, j) > 0)
			{
				queue<CPoint2i> visit_list;
				visit_list.push(CPoint2i(i, j));
				mask.at<uchar>(i, j) = 255;
				region_mask.at<int>(i, j) = count;
				while (!visit_list.empty())
				{
					CPoint2i p = visit_list.front();
					visit_list.pop();
					for (int k = 0; k < 4; ++k)
					{
						CPoint2i pt = p + trans[k];
						if (pt[0] >= 0 && pt[0] < region_mask.rows && pt[1] >= 0 && pt[1] < region_mask.cols)
						{
							if (mask.at<uchar>(pt[0], pt[1]) == 0 && (region_mask.at<int>(p[0], p[1])>0 || region_mask.at<int>(pt[0], pt[1])>0))
							{
								visit_list.push(pt);
								mask.at<uchar>(pt[0], pt[1]) = 255;
								if (region_mask.at<int>(pt[0], pt[1]) > 0) region_mask.at<int>(pt[0], pt[1]) = count;
								else if (region_mask.at<int>(pt[0], pt[1]) < 0) region_mask.at<int>(pt[0], pt[1]) = -count;
							}
						}
					}
				}
				++count;
			}
		}
	}

	for (int i = 0; i < mask.rows; ++i)
	{
		for (int j = 0; j < mask.cols; ++j)
		{
			if (mask.at<uchar>(i, j) == 0) region_mask.at<int>(i, j) = 0;
		}
	}
}

bool in_region(const SQ_Stroke& stroke, const cv::Mat& region)
{
	for (int i = 0; i < stroke.s_points.size(); ++i)
	{
		CPoint2i p(stroke.s_points[i].y(), stroke.s_points[i].x());
		if (region.at<uchar>(p[0], p[1]) != 0) return true;
	}
	return false;
}

cv::Mat combine_region_map(const cv::Mat r1, const cv::Mat r2, const CPoint2i& corner)
{
	cv::Mat mask;
	r1.copyTo(mask);
	for (int i = 0; i < r2.rows; ++i)
	{
		for (int j = 0; j < r2.cols; ++j)
		{
			if (r2.at<int>(i, j) < 0)
			{
				CPoint2i p = corner + CPoint2i(i, j);
				if (p[0] >= 0 && p[0] < r1.rows && p[1] >= 0 && p[1] < r1.cols)
				{
					mask.at<int>(p[0], p[1]) = -1;
				}
			}
		}
	}
	flood_fill(mask);
	return mask;
}

CPoint2i find_nearest_points(const cv::Mat& region_mask, const cv::Mat& side_mask, const int sign, const CPoint2i& p)
{
	queue<CPoint2i> pt_list;
	pt_list.push(p);
	int count = 0;
	while (!pt_list.empty() && count < 100)
	{
		CPoint2i p = pt_list.front();
		pt_list.pop();
		if (p[0] >= 0 && p[0] < region_mask.rows && p[1] >= 0 && p[1] < region_mask.cols)
		{
			if (region_mask.at<int>(p[0], p[1]) < 0 && sign*side_mask.at<int>(p[0], p[1]) >= 0)
			{
				return p;
			}
		}
		for (int i = 0; i < 8; ++i)
		{
			pt_list.push(p + trans[i]);
		}
		++count;
	}
	return CPoint2i(-1, -1);
}

void update_colors(SQ_Stroke& stroke, const cv::Mat& result, const cv::Mat& region_mask, const cv::Mat& side_mask, const cv::Mat& roi, const CPoint2i& corner)
{
	if (stroke.s_mdmode == 0)
	{
		for (int i = 0; i < stroke.s_points.size(); ++i)
		{
			//if (stroke.s_properties[i].keyframe_1)
			{
				CPoint2i p(stroke.s_points[i].y(), stroke.s_points[i].x());
				if (roi.at<uchar>(p[0], p[1]) != 0)
				{
					CPoint2i pt = find_nearest_points(region_mask, side_mask, 0, p - corner);
					if (pt != CPoint2i(-1, -1))
					{
						pt += corner;
						cv::Vec3f c = result.at<cv::Vec3f>(pt[0], pt[1]);
						if (c[0] < 0) c[0] = 0;
						else if (c[0]>255) c[0] = 255;
						if (c[1] < 0) c[1] = 0;
						else if (c[1]>255) c[1] = 255;
						if (c[2] < 0) c[2] = 0;
						else if (c[2]>255) c[2] = 255;
						QColor qc(c[2], c[1], c[0]);
						stroke.s_properties[i].color_1 = stroke.s_properties[i].color_2 = qc;
						stroke.s_properties[i].keyframe_1 = stroke.s_properties[i].keyframe_2 = true;
					}
				}
			}
		}
	}
	else
	{
		for (int i = 0; i < stroke.s_points.size(); ++i)
		{
			//if (stroke.s_properties[i].keyframe_1)
			{
				CPoint2i p(stroke.s_points[i].y(), stroke.s_points[i].x());
				if (roi.at<uchar>(p[0], p[1]) != 0)
				{
					CPoint2i pt = find_nearest_points(region_mask, side_mask, -1, p - corner);
					if (pt != CPoint2i(-1, -1))
					{
						pt += corner;
						cv::Vec3f c = result.at<cv::Vec3f>(pt[0], pt[1]);
						if (c[0] < 0) c[0] = 0;
						else if (c[0]>255) c[0] = 255;
						if (c[1] < 0) c[1] = 0;
						else if (c[1]>255) c[1] = 255;
						if (c[2] < 0) c[2] = 0;
						else if (c[2]>255) c[2] = 255;
						QColor qc(c[2], c[1], c[0]);
						stroke.s_properties[i].color_1 = qc;
						stroke.s_properties[i].keyframe_1 = true;
					}
				}
			}

			//if (stroke.s_properties[i].keyframe_2)
			{
				CPoint2i p(stroke.s_points[i].y(), stroke.s_points[i].x());
				if (roi.at<uchar>(p[0], p[1]) != 0)
				{
					CPoint2i pt = find_nearest_points(region_mask, side_mask, 1, p - corner);
					if (pt != CPoint2i(-1, -1))
					{
						pt += corner;
						cv::Vec3f c = result.at<cv::Vec3f>(pt[0], pt[1]);
						if (c[0] < 0) c[0] = 0;
						else if (c[0]>255) c[0] = 255;
						if (c[1] < 0) c[1] = 0;
						else if (c[1]>255) c[1] = 255;
						if (c[2] < 0) c[2] = 0;
						else if (c[2]>255) c[2] = 255;
						QColor qc(c[2], c[1], c[0]);
						stroke.s_properties[i].color_2 = qc;
						stroke.s_properties[i].keyframe_2 = true;
					}
				}
			}
		}
	}
}

vector<CPoint2i> connect(vector<vector<CPoint2i>> points)
{
	vector<CPoint2i> point_list(points.back());
	points.pop_back();
	while (!points.empty())
	{
		double min_dis = numeric_limits<double>::infinity();
		size_t id = 0;
		for (size_t i = 1; i < points.size(); ++i)
		{
			double d1 = norm(point_list.front() - points[i].front());
			double d2 = norm(point_list.back() - points[i].front());
			double d3 = norm(point_list.front() - points[i].back());
			double d4 = norm(point_list.back() - points[i].back());
			double d = std::min({ d1, d2, d3, d4 });
			if (d < min_dis)
			{
				id = i;
				min_dis = d;
			}
		}
		double d1 = norm(point_list.front() - points[id].front());
		double d2 = norm(point_list.back() - points[id].front());
		double d3 = norm(point_list.front() - points[id].back());
		double d4 = norm(point_list.back() - points[id].back());
		double d = std::min({ d1, d2, d3, d4 });
		if (d == d1)
		{
			reverse(point_list.begin(), point_list.end());
			point_list.insert(point_list.cend(), points[id].cbegin(), points[id].cend());
		}
		else if (d == d2)
		{
			point_list.insert(point_list.cend(), points[id].cbegin(), points[id].cend());
		}
		else if (d == d3)
		{
			reverse(point_list.begin(), point_list.end());
			point_list.insert(point_list.cend(), points[id].crbegin(), points[id].crend());
		}
		else if (d == d4)
		{
			point_list.insert(point_list.cend(), points[id].crbegin(), points[id].crend());
		}
		points.erase(points.begin() + id);
	}
	return point_list;
}

vector<int> extract_splitters(const QVector<QPointF>& points, const vector<CPoint2i>& curve)
{
	vector<int> spliters;
	for (int i = 0; i < points.size(); ++i)
	{
		CPoint2d p(points[i].y(), points[i].x());
		int min_id = -1;
		double min_dis = numeric_limits<double>::infinity();
		for (int i = 0; i < curve.size(); ++i)
		{
			double d = norm(CPoint2d(curve[i]) - p);
			if (d < min_dis)
			{
				min_dis = d;
				min_id = i;
			}
		}
		spliters.push_back(min_id);
	}
	return spliters;
}

pair<vector<vector<Vec2i>>, vector<vector<Vec2i>>> lap_edge_splited(int height, int width, const SQ_Stroke& stroke, const vector<vector<pair<Vec2i, Vec3f>>>& lap_edges)
{
	Mat mask = Mat::zeros(height, width, CV_8UC1);
	for (int i = 0; i < lap_edges[0].size(); ++i)
		mask.at<uchar>(lap_edges[0][i].first[0], lap_edges[0][i].first[1]) = 255;
	vector<vector<CPoint2i>> pts = extract_curves(mask);
	vector<CPoint2i> curve1 = connect(pts);
	
	mask.setTo(0);
	for (int i = 0; i < lap_edges[1].size(); ++i)
		mask.at<uchar>(lap_edges[1][i].first[0], lap_edges[1][i].first[1]) = 255;
	pts = extract_curves(mask);
	vector<CPoint2i> curve2 = connect(pts);

	vector<int> spliters1, spliters2;
	if (stroke.s_mode == SQ_Stroke::OPEN)
	{
		CPoint2d p(stroke.s_points[0].y(), stroke.s_points[0].x());
		double d1 = norm(CPoint2d(curve1.front()) - p);
		double d2 = norm(CPoint2d(curve1.back()) - p);
		if (d1 > d2) reverse(curve1.begin(), curve1.end());

		d1 = norm(CPoint2d(curve2.front()) - p);
		d2 = norm(CPoint2d(curve2.back()) - p);
		if (d1 > d2) reverse(curve2.begin(), curve2.end());
		spliters1 = extract_splitters(stroke.s_points, curve1);
		spliters2 = extract_splitters(stroke.s_points, curve2);
		spliters1.front() = 0;
		spliters2.front() = 0;
		spliters1.back() = curve1.size() - 1;
		spliters2.back() = curve2.size() - 1;
	}
	else
	{
		spliters1 = extract_splitters(stroke.s_points, curve1);
		spliters2 = extract_splitters(stroke.s_points, curve2);
		vector<CPoint2i> pts1;
		pts1.push_back(curve1[spliters1[0]]);
		for (size_t i = (spliters1[0] + 1) % curve1.size(); i != spliters1[0]; i = (i + 1) % curve1.size())
			pts1.push_back(curve1[i]);
		curve1 = pts1;
		vector<CPoint2i> pts2;
		pts2.push_back(curve2[spliters2[0]]);
		for (size_t i = (spliters2[0] + 1) % curve2.size(); i != spliters2[0]; i = (i + 1) % curve2.size())
			pts2.push_back(curve2[i]);
		curve2 = pts2;

		int offset1 = spliters1[0];
		for (size_t i = 0; i < spliters1.size(); ++i)
			spliters1[i] = (spliters1[i] + curve1.size() - offset1) % curve1.size();

		int offset2 = spliters1[0];
		for (size_t i = 0; i < spliters1.size(); ++i)
			spliters2[i] = (spliters2[i] + curve2.size() - offset2) % curve2.size();

		if (spliters1.size() >= 3)
		{
			if (spliters1[1] > spliters1[2]) reverse(curve1.begin() + 1, curve1.end());
		}
		if (spliters2.size() >= 3)
		{
			if (spliters2[1] > spliters2[2]) reverse(curve2.begin() + 1, curve2.end());
		}
		spliters1 = extract_splitters(stroke.s_points, curve1);
		spliters2 = extract_splitters(stroke.s_points, curve2);
	}

	pair<vector<vector<Vec2i>>, vector<vector<Vec2i>>> lap_splited_edges;
	for (size_t i = 1; i < spliters1.size(); ++i)
	{
		lap_splited_edges.first.push_back(vector<Vec2i>());
		for (size_t id = spliters1[i - 1]; id != spliters1[i]; id = (id + 1) % curve1.size())
			lap_splited_edges.first.back().push_back(curve1[id]);
	}
	lap_splited_edges.first.back().push_back(curve1[spliters1.back()]);
	for (size_t i = 1; i < spliters2.size(); ++i)
	{
		lap_splited_edges.second.push_back(vector<Vec2i>());
		for (size_t id = spliters2[i - 1]; id != spliters2[i]; id = (id + 1) % curve2.size())
			lap_splited_edges.second.back().push_back(curve2[id]);
	}
	lap_splited_edges.second.back().push_back(curve2[spliters2.back()]);
	return lap_splited_edges;
}

void update_min_max(const SQ_Stroke& stroke, double & min_r, double & min_c, double & max_r, double & max_c)
{
	for (int i = 0; i < stroke.s_points.size(); ++i)
	{
		if (stroke.s_points[i].x() < min_c) min_c = stroke.s_points[i].x();
		if (stroke.s_points[i].x() > max_c) max_c = stroke.s_points[i].x();
		if (stroke.s_points[i].y() < min_r) min_r = stroke.s_points[i].y();
		if (stroke.s_points[i].y() > max_r) max_r = stroke.s_points[i].y();
	}
}

BoundingBox<double> get_boundingbox(const Frame & frame)
{
	double min_r = 1e20;
	double min_c = 1e20;
	double max_r = -1e20;
	double max_c = -1e20;

	for (int i = 0; i < frame.m_strokes.size(); ++i)
		update_min_max(frame.m_strokes[i], min_r, min_c, max_r, max_c);

	for (int i = 0; i < frame.lap_edges.size(); ++i)
		update_min_max(frame.lap_edges[i], min_r, min_c, max_r, max_c);

	for (int i = 0; i < frame.lap_regions.size(); ++i)
		update_min_max(frame.lap_regions[i], min_r, min_c, max_r, max_c);

	return BoundingBox< double >(min_r, min_c, max_r - min_r, max_c - min_c);
}

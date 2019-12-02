#ifndef STRUCTURE_H
#define STRUCTURE_H
#include "point_vector.h"

enum PointType{ INNER, BOUNDARY, OUTER };

template<typename T>
struct TreeNode
{
	PointType type;
	int index;
	T row;
	T col;
	T width;
	TreeNode(T row, T col, T width):
		row(row),col(col),width(width)
	{
	}

	TreeNode():row(0),col(0),width(0)
	{
	}
	CPoint2d center() const
	{
		return CPoint2d(row + width / 2.0, col + width / 2.0);
	}
};

typedef TreeNode<int> TreeNodeD;

template<typename T>
struct BoundingBox
{
	T row;
	T col;
	T height;
	T width;
	BoundingBox(const T row, const T col, const T height, const T width)
		: row(row), col(col), height(height), width(width)
	{}
	BoundingBox():row(0),col(0),height(0),width(0)
	{}
	T area() const
	{
		return height*width;
	}
	bool intersection_boundingbox(const BoundingBox<T>& box)
	{
		T row_max = std::max(row, box.row);
		T col_max = std::max(col, box.col);
		T row_min = std::min(row + height, box.row + box.height);
		T col_min = std::min(col + width, box.col + box.width);

		row = row_max;
		col = col_max;
		height = row_min - row_max;
		width = col_min - col_max;

		if (height <= 0 || width <= 0)
		{
			row = col = width = height = 0;
			return false;
		}
		else return true;
	}
	bool valid() const
	{
		return width != 0 && height != 0;
	}
};

#endif

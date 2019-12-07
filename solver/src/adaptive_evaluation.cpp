#include "adaptive_evaluation.h"
#include "auxiliary.h"
#include "debug.h"
#include "point_vector.h"
#include "quadtree.h"
#include "region.h"
#include "structure.h"
#include "tree.hh"
#include <Eigen/Sparse>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <tbb/tbb.h>


using namespace std;
using namespace cv;


AdaptiveEvaluation::AdaptiveEvaluation(
        const QuadTree & quadtree,
        const std::vector<cv::Vec3f> & coefs
) :
        quadtree(quadtree),
        coefs(coefs),
        laplacian_matrix(quadtree.get_laplacian_basis())
{
    quadtree.get_regions(regions);

    //	cuda_var.coefs = nullptr;
    //	cuda_var.regions = nullptr;
    //	cuda_var.index = nullptr;
    //	cuda_var.points = nullptr;
    //
    //	cuda_var.laplacian_matrix.cscValA = nullptr;
    //	cuda_var.laplacian_matrix.cscColPtrA = nullptr;
    //	cuda_var.laplacian_matrix.cscRowIndA = nullptr;
    //
    //	cuda_var.neighbors.data = nullptr;
    //	cuda_var.neighbors.row_ptr = nullptr;
    //	cuda_var.neighbors.row_width = nullptr;

    neighbors_small_size = -1;

    vector<TreeNodeD> nodes;
    quadtree.get_level1_nodes(nodes);
    neighbors.resize(nodes.size());

    tbb::parallel_for(0, (int) (nodes.size()), 1, [&](int i)
    {
        if (quadtree.with_inner_node(nodes[i]))
        {
            quadtree.get_neighbor_nodes(nodes[i].center(), neighbors[i], default_rings, 8 * default_rings);
        }
    });

    //	transfer_variables();
}

AdaptiveEvaluation::~AdaptiveEvaluation()
{
    //	if (cuda_var.coefs != nullptr) checkCudaErrors(cudaFree(cuda_var.coefs));
    //	if (cuda_var.regions != nullptr) checkCudaErrors(cudaFree(cuda_var.regions));
    //
    //	if (cuda_var.neighbors.data != nullptr)  checkCudaErrors(cudaFree(cuda_var.neighbors.data));
    //	if (cuda_var.neighbors.row_width != nullptr)  checkCudaErrors(cudaFree(cuda_var.neighbors.row_width));
    //	if (cuda_var.neighbors.row_ptr != nullptr)  checkCudaErrors(cudaFree(cuda_var.neighbors.row_ptr));
    //
    //	if (cuda_var.laplacian_matrix.cscValA != nullptr)  checkCudaErrors(cudaFree(cuda_var.laplacian_matrix.cscValA));
    //	if (cuda_var.laplacian_matrix.cscRowIndA != nullptr)  checkCudaErrors(cudaFree(cuda_var.laplacian_matrix.cscRowIndA));
    //	if (cuda_var.laplacian_matrix.cscColPtrA != nullptr)  checkCudaErrors(cudaFree(cuda_var.laplacian_matrix.cscColPtrA));
}

//void AdaptiveEvaluation::transfer_variables()
//{
//	//create coefs
//	float3* coefs_host = (float3*)malloc(sizeof(float3)*coefs.size());
//	for (size_t i = 0; i < coefs.size(); ++i)
//	{
//		coefs_host[i].x = (float)coefs[i][0];
//		coefs_host[i].y = (float)coefs[i][1];
//		coefs_host[i].z = (float)coefs[i][2];
//	}
//	checkCudaErrors(cudaMalloc((void**)&cuda_var.coefs, sizeof(float3)*coefs.size()));
//	checkCudaErrors(cudaMemcpy(cuda_var.coefs, coefs_host, sizeof(float3)*coefs.size(), cudaMemcpyHostToDevice));
//	free(coefs_host);
//	//create regions
//	TreeNode<float>* regions_host = (TreeNode<float>*)malloc(sizeof(TreeNode<float>)*regions.size());
//	for (size_t i = 0; i < regions.size(); ++i)
//	{
//		regions_host[i].type = regions[i].type;
//		regions_host[i].index = regions[i].index;
//		regions_host[i].row = (float)regions[i].row;
//		regions_host[i].col = (float)regions[i].col;
//		regions_host[i].width = (float)regions[i].width;
//	}
//	checkCudaErrors(cudaMalloc((void**)&cuda_var.regions, sizeof(TreeNode<float>)*regions.size()));
//	checkCudaErrors(cudaMemcpy(cuda_var.regions, regions_host, sizeof(TreeNode<float>)*regions.size(), cudaMemcpyHostToDevice));
//	free(regions_host);
//
//	//create Laplacian matrix
//	cuda_var.laplacian_matrix.nnz = laplacian_matrix.nonZeros();
//	float* cscValA_host = (float*)malloc(sizeof(float)*laplacian_matrix.nonZeros());
//	int* cscRowIndA_host = (int*)malloc(sizeof(int)*laplacian_matrix.nonZeros());
//	int* cscColPtrA_host = (int*)malloc(sizeof(int)*(laplacian_matrix.cols() + 1));
//
//	int count = 0;
//	for (int i = 0; i < laplacian_matrix.cols(); ++i)
//	{
//		cscColPtrA_host[i] = count;
//		for (Eigen::SparseMatrix<double>::InnerIterator it(laplacian_matrix, i); it; ++it)
//		{
//			cscValA_host[count] = (float)it.value();
//			cscRowIndA_host[count] = it.row();
//			++count;
//		}
//	}
//	cscColPtrA_host[laplacian_matrix.cols()] = count;
//	checkCudaErrors(cudaMalloc((void**)&cuda_var.laplacian_matrix.cscValA, sizeof(float)*laplacian_matrix.nonZeros()));
//	checkCudaErrors(cudaMalloc((void**)&cuda_var.laplacian_matrix.cscRowIndA, sizeof(int)*laplacian_matrix.nonZeros()));
//	checkCudaErrors(cudaMalloc((void**)&cuda_var.laplacian_matrix.cscColPtrA, sizeof(int)*(laplacian_matrix.cols() + 1)));
//
//	checkCudaErrors(cudaMemcpy(cuda_var.laplacian_matrix.cscValA, cscValA_host, sizeof(float)*laplacian_matrix.nonZeros(), cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(cuda_var.laplacian_matrix.cscRowIndA, cscRowIndA_host, sizeof(int)*laplacian_matrix.nonZeros(), cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(cuda_var.laplacian_matrix.cscColPtrA, cscColPtrA_host, sizeof(int)*(laplacian_matrix.cols() + 1), cudaMemcpyHostToDevice));
//
//	free(cscValA_host);
//	free(cscRowIndA_host);
//	free(cscColPtrA_host);
//}

//void AdaptiveEvaluation::transfer_neigobors(const vector<vector<int>>& neighbor_array)
//{
//	free_neighbors();
//	//create neighbors
//	cuda_var.neighbors.n_rows = (int)neighbor_array.size();
//	size_t ele_num = 0;
//	for (size_t i = 0; i < neighbor_array.size(); ++i)
//		ele_num += neighbor_array[i].size();
//
//	int* raw;
//	checkCudaErrors(cudaMallocHost((void**)&raw, sizeof(int)*max(ele_num, neighbor_array.size())));
//	for (size_t i = 0; i < neighbor_array.size(); ++i)
//		raw[i] = (int)neighbor_array[i].size();
//	checkCudaErrors(cudaMalloc((void**)&cuda_var.neighbors.row_width, sizeof(int)*neighbor_array.size()));
//	checkCudaErrors(cudaMemcpy(cuda_var.neighbors.row_width, raw, sizeof(int)*neighbor_array.size(), cudaMemcpyHostToDevice));
//	int count = 0;
//	for (size_t i = 0; i < neighbor_array.size(); ++i)
//	{
//		raw[i] = count;
//		count += (int)neighbor_array[i].size();
//	}
//	checkCudaErrors(cudaMalloc((void**)&cuda_var.neighbors.row_ptr, sizeof(int)*neighbor_array.size()));
//	checkCudaErrors(cudaMemcpy(cuda_var.neighbors.row_ptr, raw, sizeof(int)*neighbor_array.size(), cudaMemcpyHostToDevice));
//
//	count = 0;
//	for (size_t i = 0; i < neighbor_array.size(); ++i)
//	{
//		for (size_t j = 0; j < neighbor_array[i].size(); ++j)
//		{
//			raw[count++] = neighbor_array[i][j];
//		}
//	}
//	checkCudaErrors(cudaMalloc((void**)&cuda_var.neighbors.data, sizeof(int)*ele_num));
//	checkCudaErrors(cudaMemcpy(cuda_var.neighbors.data, raw, sizeof(int)*ele_num, cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaFreeHost(raw));
//}

//void AdaptiveEvaluation::free_neighbors()
//{
//	if (cuda_var.neighbors.data != nullptr)
//	{
//		checkCudaErrors(cudaFree(cuda_var.neighbors.data));
//		cuda_var.neighbors.data = nullptr;
//	}
//	if (cuda_var.neighbors.row_width != nullptr)
//	{
//		checkCudaErrors(cudaFree(cuda_var.neighbors.row_width));
//		cuda_var.neighbors.row_width = nullptr;
//	}
//	if (cuda_var.neighbors.row_ptr != nullptr)
//	{
//		checkCudaErrors(cudaFree(cuda_var.neighbors.row_ptr));
//		cuda_var.neighbors.row_ptr = nullptr;
//	}
//}

//vector<CPoint2i> AdaptiveEvaluation::cuda_evaluation(const Region& region, const BoundingBox<int>& rect, double step, const CPoint2d& original, const Mat& laplacian_image, int n_rings, Mat& result)
//{
//	if (n_rings != default_rings)
//	{
//		if (neighbors_small_size != n_rings)
//		{
//			neighbors_small_size = n_rings;
//			vector<TreeNodeD> nodes;
//			quadtree.get_level1_nodes(nodes);
//			neighbors_small.clear();
//			neighbors_small.resize(nodes.size());
//
//			tbb::parallel_for(0, (int)(nodes.size()), [&](int i)
//			{
//				if (quadtree.with_inner_node(nodes[i]))
//				{
//					quadtree.get_neighbor_nodes(nodes[i].center(), neighbors_small[i], n_rings, 8 * n_rings);
//				}
//			});
//		}
//		transfer_neigobors(neighbors_small);
//	}
//	else transfer_neigobors(neighbors);
//
//	int* index = (int*)calloc(rect.width*rect.height, sizeof(int));
//	float2* points = (float2*)malloc(sizeof(float2)*rect.width*rect.height);
//
//	float3* image_host = (float3*)malloc(sizeof(float3)*rect.width*rect.height);
//
//	for (int i = 0; i < rect.width*rect.height; ++i)
//		image_host[i] = { -10000, -10000, -10000 };
//
//	Mat critical_points_mat = Mat::zeros(rect.height, rect.width, CV_8UC1);
//
//	tbb::parallel_for(0, rect.height, [&](int i)
//	{
//		for (int j = 0; j < rect.width; ++j)
//		{
//			CPoint2f p((float)(original[0] + (i + 0.5)*step), (float)(original[1] + (j + 0.5)*step));
//			CPoint2f pt(p);
//
//			if (step != 1.0 && region.is_critical(p))
//			{
//				if (region.in_crossing(p))
//				{
//					vector<int> idx = region.crossing_index(p);
//					CPoint2i p_ = region.find_nearby_point(p, idx);
//					if (p_ != CPoint2i(-1, -1) && !idx.empty())
//					{
//						pt[0] = p_[0] + 0.5f;
//						pt[1] = p_[1] + 0.5f;
//
//						critical_points_mat.at<uchar>(i, j) = 255;
//					}
//					else {
//						index[i*rect.width + j] = -1;
//					}
//				}
//				else
//				{
//					int edge_id = region.get_edge_id(p);
//					if (edge_id != 0)
//					{
//						CPoint2i p_ = region.find_closest_pixel(p, edge_id);
//						pt[0] = p_[0] + 0.5f;
//						pt[1] = p_[1] + 0.5f;
//
//						critical_points_mat.at<uchar>(i, j) = 255;
//					}
//					else index[i*rect.width + j] = -1;
//				}
//			}
//
//			if ((step == 1.0 || index[i*rect.width + j] != -1) && (region.is_boundary((int)pt[0], (int)pt[1]) || region.is_singular((int)pt[0], (int)pt[1])))
//			{
//				index[i*rect.width + j] = -1;
//				image_host[i*rect.width + j].x = laplacian_image.at<Vec3f>((int)pt[0], (int)pt[1])[0];
//				image_host[i*rect.width + j].y = laplacian_image.at<Vec3f>((int)pt[0], (int)pt[1])[1];
//				image_host[i*rect.width + j].z = laplacian_image.at<Vec3f>((int)pt[0], (int)pt[1])[2];
//			}
//
//			if (index[i*rect.width + j] == 0)
//			{
//				int id = quadtree.search(pt);
//				if (id >= 0 && id < quadtree.get_number_of_pixel_points())
//				{
//					index[i*rect.width + j] = quadtree.search_level1(pt);
//				}
//				else
//				{
//					index[i*rect.width + j] = -1;
//				}
//
//				points[i*rect.width + j].x = pt[0];
//				points[i*rect.width + j].y = pt[1];
//			}
//		}
//	});
//
//	cuda_var.width = rect.width;
//	cuda_var.height = rect.height;
//	checkCudaErrors(cudaMalloc((void**)&cuda_var.index, sizeof(int)*rect.width*rect.height));
//	checkCudaErrors(cudaMalloc((void**)&cuda_var.points, sizeof(float2)*rect.width*rect.height));
//	checkCudaErrors(cudaMemcpy(cuda_var.index, index, sizeof(int)*rect.width*rect.height, cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpy(cuda_var.points, points, sizeof(float2)*rect.width*rect.height, cudaMemcpyHostToDevice));
//	free(index);
//	free(points);
//
//	float3* image_device;
//	checkCudaErrors(cudaMalloc((void**)&image_device, sizeof(float3)*rect.width*rect.height));
//	checkCudaErrors(cudaMemcpy(image_device, image_host, sizeof(float3)*rect.width*rect.height, cudaMemcpyHostToDevice));
//
//	//	clock_t t = clock();
//	cuda_evaluation_func(cuda_var, image_device);
//	checkCudaErrors(cudaDeviceSynchronize());
//	//	Logger::ins() << "CUDA Time " << (float)(clock() - t) / CLOCKS_PER_SEC << "\n";
//	checkCudaErrors(cudaMemcpy(image_host, image_device, sizeof(float3)*rect.width*rect.height, cudaMemcpyDeviceToHost));
//	checkCudaErrors(cudaFree(image_device));
//	checkCudaErrors(cudaFree(cuda_var.index));
//	checkCudaErrors(cudaFree(cuda_var.points));
//
//
//	for (int i = 0; i < rect.height; ++i)
//	{
//		for (int j = 0; j < rect.width; ++j)
//		{
//			float3* p = image_host + i*rect.width + j;
//			if (p->x != -10000 || p->y != -10000 || p->z != -10000)
//			{
//				result.at<Vec3f>(rect.row + i, rect.col + j) = Vec3f(p->x, p->y, p->z);
//			}
//		}
//	}
//
//	free(image_host);
//
//	vector<CPoint2i> critical_points;
//	for (int i = 0; i < critical_points_mat.rows; ++i)
//	{
//		for (int j = 0; j < critical_points_mat.cols; ++j)
//		{
//			if (critical_points_mat.at<uchar>(i, j) != 0)
//			{
//				critical_points.push_back(CPoint2i(rect.row + i, rect.col + j));
//			}
//		}
//	}
//
//	return critical_points;
//}

//void AdaptiveEvaluation::jacobian(cv::Mat& result, const cv::Mat& enlarged_side_mask, std::vector<CPoint2i>& points, int round, int radius)
//{
//	float3* image_host = (float3*)malloc(sizeof(float3)*result.rows*result.cols);
//	for (int i = 0; i < result.rows; ++i)
//	{
//		for (int j = 0; j < result.cols; ++j)
//		{
//			image_host[i*result.cols + j].x = result.at<Vec3f>(i, j)[0];
//			image_host[i*result.cols + j].y = result.at<Vec3f>(i, j)[1];
//			image_host[i*result.cols + j].z = result.at<Vec3f>(i, j)[2];
//		}
//	}
//	float3* image_device;
//	checkCudaErrors(cudaMalloc((void**)&image_device, sizeof(float3)*result.rows*result.cols));
//	checkCudaErrors(cudaMemcpy(image_device, image_host, sizeof(float3)*result.rows*result.cols, cudaMemcpyHostToDevice));
//
//	int* enlarged_side_host = (int*)malloc(sizeof(int)*enlarged_side_mask.rows*enlarged_side_mask.cols);
//	for (int i = 0; i < enlarged_side_mask.rows; ++i)
//	{
//		for (int j = 0; j < enlarged_side_mask.cols; ++j)
//		{
//			if (enlarged_side_mask.at<int>(i, j) != 0)
//				enlarged_side_host[i*enlarged_side_mask.cols + j] = 1;
//			else enlarged_side_host[i*enlarged_side_mask.cols + j] = 0;
//		}
//	}
//	int* region_index;
//	checkCudaErrors(cudaMalloc((void**)&region_index, sizeof(int)*enlarged_side_mask.rows*enlarged_side_mask.cols));
//	checkCudaErrors(cudaMemcpy(region_index, enlarged_side_host, sizeof(int)*enlarged_side_mask.rows*enlarged_side_mask.cols, cudaMemcpyHostToDevice));
//	free(enlarged_side_host);
//
//	int size_critical_points = (int)points.size();
//	int2* critical_points_host = (int2*)malloc(sizeof(int2)*points.size());
//	for (size_t i = 0; i < points.size(); ++i)
//	{
//		critical_points_host[i].x = points[i][0];
//		critical_points_host[i].y = points[i][1];
//	}
//	int2* critical_points_device;
//	checkCudaErrors(cudaMalloc((void**)&critical_points_device, sizeof(int2)*points.size()));
//	checkCudaErrors(cudaMemcpy(critical_points_device, critical_points_host, sizeof(int2)*points.size(), cudaMemcpyHostToDevice));
//	free(critical_points_host);
//
//	jacobian_func(&image_device, region_index, critical_points_device, (int)points.size(), result.cols, result.rows, round, radius);
//
//	checkCudaErrors(cudaMemcpy(image_host, image_device, sizeof(float3)*result.rows*result.cols, cudaMemcpyDeviceToHost));
//	checkCudaErrors(cudaFree(image_device));
//	checkCudaErrors(cudaFree(region_index));
//	checkCudaErrors(cudaFree(critical_points_device));
//
//	for (int i = 0; i < result.rows; ++i)
//	{
//		for (int j = 0; j < result.cols; ++j)
//		{
//			float3* p = image_host + i*result.cols + j;
//			{
//				result.at<Vec3f>(i,j) = Vec3f(p->x, p->y, p->z);
//			}
//		}
//	}
//	free(image_host);
//}

static double source_double(const CPoint2d & func, double pt_x, double pt_y)
{
    double x = pt_x - func[0];
    double y = pt_y - func[1];

    double v1 = atan(y / x);

    double x2 = x * x;
    double y2 = y * y;
    double v = x * y * (log(x2 + y2) - 3.0) + (x2 - y2) * v1;
    y2 *= CV_PI / 2;

    if (x == 0.0 || y == 0.0)
    {
        v = y2 = 0.0;
    }

    return v1 > 0 ? v + y2 : v - y2;
}

static double green_integral_double(const CPoint2d & func, double x1, double x2, double y1, double y2)
{
    return source_double(func, x1, y1) + source_double(func, x2, y2) - source_double(func, x1, y2) -
           source_double(func, x2, y1);
}

vector<CPoint2i> AdaptiveEvaluation::full_solution(
        const Region & region,
        const BoundingBox<int> & rect,
        double step,
        const CPoint2d & original,
        const cv::Mat & laplacian_image,
        int n_ring,
        cv::Mat & result)
{
    st_debug("AdaptiveEvaluation::full_solution");

    Eigen::MatrixXd v(coefs.size(), 3);

    for (size_t i = 0; i < coefs.size(); ++i)
    {
        v(i, 0) = coefs[i][0];
        v(i, 1) = coefs[i][1];
        v(i, 2) = coefs[i][2];
    }

    Eigen::MatrixXd vec = laplacian_matrix * v;
    vector<Vec3d> coef(vec.rows());

    for (size_t i = 0; i < coef.size(); ++i)
    {
        coef[i][0] = vec(i, 0);
        coef[i][1] = vec(i, 1);
        coef[i][2] = vec(i, 2);
    }

    vector<TreeNodeD> nodes;
    quadtree.get_regions(nodes);

    int * index = (int *) calloc(rect.width * rect.height, sizeof(int));
    Mat critical_points_mat = Mat::zeros(rect.height, rect.width, CV_8UC1);

    tbb::parallel_for(0, rect.height, [&](int i)
    {
        printf("%d ", i);

        for (int j = 0; j < rect.width; ++j)
        {
            CPoint2d p(original[0] + (i + 0.5) * step, original[1] + (j + 0.5) * step);

            CPoint2d pt(p);
            if (step != 1.0 && region.is_critical(p))
            {
                vector<int> idx = region.crossing_index(p);

                if (idx.empty())
                {
                    int edge_id = region.get_edge_id(p);

                    if (edge_id != 0)
                    {
                        CPoint2i p_ = region.find_closest_pixel(p, edge_id);
                        pt[0] = p_[0] + 0.5f;
                        pt[1] = p_[1] + 0.5f;

                        critical_points_mat.at<uchar>(i, j) = 255;
                    }
                    else
                    {
                        index[i * rect.width + j] = -1;
                    }
                }
                else
                {
                    CPoint2i p_ = region.find_nearby_point(p, idx);

                    if (p_ != CPoint2i(-1, -1))
                    {
                        pt[0] = p_[0] + 0.5f;
                        pt[1] = p_[1] + 0.5f;

                        critical_points_mat.at<uchar>(i, j) = 255;
                    }
                    else
                    {
                        index[i * rect.width + j] = -1;
                    }
                }
            }

            if ((step == 1.0 || index[i * rect.width + j] != -1) &&
                (region.is_boundary((int) pt[0], (int) pt[1]) || region.is_singular((int) pt[0], (int) pt[1])))
            {
                index[i * rect.width + j] = -1;
                result.at<Vec3f>(rect.row + i, rect.col + j) = laplacian_image.at<Vec3f>((int) pt[0], (int) pt[1]);
            }

            if (index[i * rect.width + j] == 0)
            {
                int id = quadtree.search(pt);

                if (id >= 0 && id < quadtree.get_number_of_inner_points())
                {
                    Vec3d val(0, 0, 0);

                    for (size_t k = 0; k < coef.size(); ++k)
                    {
                        val += coef[k] *
                               green_integral_double(pt, nodes[k].row, nodes[k].row + nodes[k].width, nodes[k].col,
                                                     nodes[k].col + nodes[k].width) / (nodes[k].width * nodes[k].width);
                    }

                    val *= 0.25 / CV_PI;
                    result.at<Vec3f>(rect.row + i, rect.col + j) = val;
                }
            }
        }
    });

    vector<CPoint2i> critical_points;

    for (int i = 0; i < critical_points_mat.rows; ++i)
    {
        for (int j = 0; j < critical_points_mat.cols; ++j)
        {
            if (critical_points_mat.at<uchar>(i, j) != 0)
            {
                critical_points.push_back(CPoint2i(rect.row + i, rect.col + j));
            }
        }
    }

    return critical_points;
}

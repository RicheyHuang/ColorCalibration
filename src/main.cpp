#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <typeinfo.h>
#include <math.h>
#include <numeric>
#include <map>
#include <algorithm>

cv::Point2f findCenter(std::vector<cv::Point> contour)
{
	cv::Point center_ = std::accumulate(contour.begin(), contour.end(), cv::Point(0, 0)) / (float)contour.size();

	cv::Point2f center = cv::Point2f(center_.x, center_.y);
	return center;
}

class Square
{
public:

	std::vector<std::vector<cv::Point>> m_contour;
	std::vector<cv::Point> m_points;
	std::vector<float> m_dists;
	float m_perimeter;
	std::vector<float> m_angles;
	cv::Vec3f m_color = cv::Vec3f(0, 0, 0);
	cv::Vec3f m_labcolor = cv::Vec3f(0, 0, 0);
	float m_gridX = 0;
	float m_gridY = 0;
	cv::Point2f m_center = cv::Point2f(0, 0);

};

class Possibility
{
public:
	float val_1 = 0;
	float val_2 = 0;
	cv::Vec3f val_3 = cv::Vec3f(0, 0, 0);
};

bool compare(float A, float B)
{
	return A > B;
}

bool compareGrid(Square A, Square B)
{
	float a_val = A.m_gridX * A.m_gridX + A.m_gridY * A.m_gridY;
	float b_val = B.m_gridX * B.m_gridX + B.m_gridY * B.m_gridY;

	return a_val < b_val;
}

bool compareGridX(Square A, Square B)
{
	float a_val = A.m_gridX;
	float b_val = B.m_gridX;

	return a_val < b_val;
}

bool compareGridY(Square A, Square B)
{
	float a_val = A.m_gridY;
	float b_val = B.m_gridY;

	return a_val < b_val;
}

bool comparePossibility(Possibility A, Possibility B)
{
	float a_val_1 = A.val_1;
	float a_val_2 = A.val_2;

	float b_val_1 = B.val_1;
	float b_val_2 = B.val_2;

	if (a_val_1 < b_val_1)
	{
		return true;
	}
	else if (a_val_1 == b_val_1 && a_val_2 < b_val_2)
	{
		return true;
	}
	else
	{
		return false;
	}
}

typedef std::map<int, Square> Squares;

Squares processContour(std::vector<std::vector<cv::Point>> Contours, std::vector<cv::Vec4i> hierarchy, float minWidth = 30)
{
	Squares squares;

	std::vector<std::vector<cv::Point>> contours;

	contours.assign(Contours.begin(), Contours.end());

	for (int index = 0; index < contours.size(); index++)
	{
		Square square;

		if (contours[index].empty())
		{
			continue;
		}

		if (cv::contourArea(contours[index])<20 * 20)
		{
			contours[index].clear();
			continue;
		}
		for (int eps = 2; eps < 20; eps++)
		{
			cv::approxPolyDP(contours[index], contours[index], float(eps), true);
			cv::approxPolyDP(contours[index], contours[index], float(eps), true);
		}
		if (!(cv::isContourConvex(contours[index]) && contours[index].size() == 4))
		{
			contours[index].clear();
			continue;
		}

		std::vector<float> dists;
		std::vector<float> angles;
		float angle;

		contours[index].push_back(contours[index][0]);

		for (int i = 0; i < contours[index].size() - 1; i++)
		{
			dists.push_back(cv::norm(contours[index][i + 1] - contours[index][i]));
			angle = atan2(contours[index][i].x - contours[index][i + 1].x, contours[index][i].y - contours[index][i + 1].y);
			while (angle < 0)
			{
				angle += CV_PI;
			}
			angles.push_back(angle);
		}
		if (!((abs(angles[2] - angles[0])<0.2 || abs(angles[2] - CV_PI - angles[0])<0.2 || abs(angles[2] + CV_PI - angles[0])<0.2)
			&& (abs(angles[3] - angles[1])<0.2 || abs(angles[3] - CV_PI - angles[1])<0.2 || abs(angles[3] + CV_PI - angles[1])<0.2)
			&& (abs(dists[0] - (dists[0] + dists[2]) / 2) + abs(dists[2] - (dists[0] + dists[2]) / 2)
				+ abs(dists[1] - (dists[1] + dists[3]) / 2) + abs(dists[3] - (dists[1] + dists[3]) / 2)<5.0)
			&& (abs((dists[1] + dists[3]) / 2 - (dists[0] + dists[2]) / 2)<30)
			))
		{
			contours[index].clear();
			continue;
		}
		int parentIndex = hierarchy[index][3];

		contours[index].pop_back();

		square.m_points.assign(contours[index].begin(), contours[index].end());

		square.m_contour.push_back(square.m_points);

		square.m_dists.assign(dists.begin(), dists.end());
		square.m_angles.assign(angles.begin(), angles.end());

		square.m_color = cv::Vec3f(255, 255, 255);
		square.m_labcolor = cv::Vec3f(0, 0, 0);
		square.m_center = findCenter(contours[index]);
		square.m_perimeter = std::accumulate(dists.begin(), dists.end(), 0);

		while (parentIndex > 0)
		{
			if (!contours[parentIndex].empty())
			{
				contours[index].clear();
				break;
			}
			else
			{
				parentIndex = hierarchy[parentIndex][3];
			}
		}
		squares[index] = square;
	}

	return squares;

}

float projectAonB(cv::Point A, cv::Point B)
{
	float dist = sqrtf(B.y * B.y + B.x * B.x);
	return (A.y * B.y + A.x * B.x) / dist;
};

std::vector<cv::Mat> setupStandardColor(cv::Mat labStandardColor)
{
	std::vector<cv::Mat> rotStandardColors;

	cv::Mat rotStandardColor = labStandardColor.clone();

	for (int i = 0; i< 4; i++)
	{
		cv::rotate(rotStandardColor, rotStandardColor, cv::ROTATE_90_CLOCKWISE);
		rotStandardColors.push_back(rotStandardColor);
	}

	return rotStandardColors;
}

double findMatMax(cv::Mat mat)
{
	double max, min;
	cv::minMaxIdx(mat, &min, &max);
	return max;
}

double findMatMin(cv::Mat mat)
{
	double max, min;
	cv::minMaxIdx(mat, &min, &max);
	return min;
}

cv::Mat findColorChecker(cv::Mat img, bool downsample = true)
{
	float MIN_DIST = 10.0;
	float MAX_VECT_ERROR = 0.97;
	float MAX_NORMALIZED_PERIMETER_ERROR = 0.4;
	int MAX_NUMBER_SQUARES_FROM_MEAN = 4;

	uchar arrStandardColor[6][4][3]
		= { { { 171,191,99 },
	{ 41,161,229 },
	{ 166,136,0 },
	{ 50,50,50 } },

	{ { 176, 129,130 },
	{ 62,189,160 },
	{ 150,84,188 },
	{ 85,84,83 } },

	{ { 65,108,90 },
	{ 105,59,91 },
	{ 22,200,238 },
	{ 121,121,120 } },

	{ { 157,123,93 },
	{ 98,84,195 },
	{ 56,48,176 },
	{ 161,161,160 } },


	{ { 129,149,196 },
	{ 168,92,72 },
	{ 72,149,71 },
	{ 201,201,200 } },


	{ { 67,81,115 },
	{ 45,123,220 },
	{ 147,62,43 },
	{ 240,245,245 } } };

	cv::Mat StandardColor = cv::Mat(cv::Size(4, 6), CV_8UC3, arrStandardColor);

	cv::Mat labStandardColor_ = cv::Mat(StandardColor.size(), CV_8UC3);
	cv::cvtColor(StandardColor, labStandardColor_, CV_BGR2Lab);

	cv::Mat labStandardColor = cv::Mat(labStandardColor_.size(), CV_32FC3);
	labStandardColor_.convertTo(labStandardColor, CV_32FC3);

	if (downsample)
	{
		cv::resize(img, img, cv::Size(img.size().width * 2560 / img.size().height, 2560));
	}

	cv::Mat total = cv::Mat::zeros(img.size(), CV_8UC3);

	cv::Mat labImg_ = cv::Mat(img.size(), CV_8UC3);

	cv::cvtColor(img, labImg_, CV_BGR2Lab);

	cv::Mat labImg = cv::Mat(labImg_.size(), CV_32FC3);
	labImg_.convertTo(labImg, CV_32FC3);

	cv::Mat normMap = cv::Mat(labImg.size(), CV_32FC1);
	cv::Mat NormMap = cv::Mat(normMap.size(), CV_8UC1);

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;

	std::vector<cv::Point2f> horizontalOffsets, verticalOffsets;

	std::vector<Square> square_vec;

	for (int i = 0; i < labStandardColor.rows; i++)
	{
		for (int j = 0; j < labStandardColor.cols; j++)
		{
			cv::Vec3f labColor = labStandardColor.at<cv::Vec3f>(i, j);

			for (int m = 0; m < labImg.rows; m++)
			{
				for (int n = 0; n < labImg.cols; n++)
				{
					normMap.at<float>(m, n) = cv::norm(labImg.at<cv::Vec3f>(m, n) - labColor);
				}
			}

			float max = findMatMax(normMap);

			normMap = normMap * 255 / max;

			normMap.convertTo(NormMap, CV_8UC1);

			cv::threshold(NormMap, NormMap, 50, 255, cv::THRESH_BINARY_INV);

			cv::morphologyEx(NormMap, NormMap, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

			cv::findContours(NormMap, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

			Squares squares = processContour(contours, hierarchy, labImg.cols / 100);

			Squares::iterator iter = squares.begin();

			cv::Mat labels = cv::Mat::zeros(img.size(), CV_8UC3);

			std::map<float, Square, std::greater<float>> means;

			while (iter != squares.end())
			{
				iter->second.m_labcolor = labColor;

				cv::Scalar t_color = cv::Scalar(iter->first + 100, 0, 0);

				cv::drawContours(labels, iter->second.m_contour, -1, t_color, -1);

				cv::Rect roi = cv::boundingRect(iter->second.m_points);

				cv::Mat mask = cv::Mat(roi.size(), CV_8UC1);
				for (int p = 0; p < mask.rows; p++)
				{
					for (int q = 0; q < mask.cols; q++)
					{
						if (labels(roi).at<cv::Vec3b>(i, j) == cv::Vec3b(iter->first + 100, 0, 0))
						{
							mask.at<uchar>(i, j) = 1;
						}
						else
						{
							mask.at<uchar>(i, j) = 0;
						}
					}
				}

				cv::Scalar mean = cv::mean(labImg(roi), mask);

				iter->second.m_labcolor = cv::Vec3f(mean[0], mean[1], mean[2]);

				means[cv::norm(iter->second.m_labcolor - labColor)] = iter->second;

				iter++;
			}

			std::map<float, Square, std::greater<float>>::iterator iter_means = means.begin();

			Square square;

			if (means.size()>0)
			{
				while (iter_means != means.end())
				{
					square = iter_means->second;

					bool atBegin = false;

					if (square_vec.size() == 0)
					{
						atBegin = true;
					}

					bool isFailed = false;

					for (int i = 0; i<square_vec.size(); i++)
					{
						if (cv::norm(iter_means->second.m_center - square_vec[i].m_center) < MIN_DIST)
						{
							isFailed = true;
							break;
						}
					}

					if (isFailed && !atBegin)
					{
						iter_means++;
						continue;
					}

					square_vec.push_back(square);
					std::vector<cv::Point> points;
					points.assign(square.m_points.begin(), square.m_points.end());
					cv::Point2f square_center = square.m_center;

					cv::Point2f horizontalOffset = ((points[1] - points[0]) / 2 + (points[2] - points[3]) / 2)*1.3;
					cv::Point2f verticalOffset = ((points[2] - points[1]) / 2 + (points[3] - points[0]) / 2) * 1.3;

					bool swap = false;

					if (horizontalOffsets.size() == 0)
					{
						swap = abs(horizontalOffset.x * 1 + horizontalOffset.y * 0) < abs(verticalOffset.x * 1 + verticalOffset.y * 0);
						if (swap)
						{
							horizontalOffsets.push_back(verticalOffset);
							verticalOffsets.push_back(horizontalOffset);
							horizontalOffset = horizontalOffsets.back();
							verticalOffset = verticalOffsets.back();
						}
						else
						{
							horizontalOffsets.push_back(horizontalOffset);
							verticalOffsets.push_back(verticalOffset);
						}
						if (horizontalOffset.x < 0)
						{
							horizontalOffset = -horizontalOffset;
							cv::Point2f temp_offset = -horizontalOffsets.back();
							horizontalOffsets.pop_back();
							horizontalOffsets.push_back(temp_offset);
						}
						if (verticalOffset.y<0)
						{
							verticalOffset = -verticalOffset;
							cv::Point2f temp_offset = -verticalOffsets.back();
							verticalOffsets.pop_back();
							verticalOffsets.push_back(temp_offset);
						}
					}
					else
					{
						swap = (abs(horizontalOffset.x * horizontalOffsets[0].x + horizontalOffset.y * horizontalOffsets[0].y) < abs(verticalOffset.x * horizontalOffsets[0].x + verticalOffset.y * horizontalOffsets[0].y));
							if (swap)
							{
								horizontalOffsets.push_back(verticalOffset);
								verticalOffsets.push_back(horizontalOffset);
								horizontalOffset = horizontalOffsets.back();
								verticalOffset = verticalOffsets.back();
							}
							else
							{
								horizontalOffsets.push_back(horizontalOffset);
								verticalOffsets.push_back(verticalOffset);
							}
						if (projectAonB(horizontalOffset, horizontalOffsets[0])<0)
						{
							horizontalOffset = -horizontalOffset;
							cv::Point2f temp_offset = -horizontalOffsets.back();
							horizontalOffsets.pop_back();
							horizontalOffsets.push_back(temp_offset);
						}
						if (projectAonB(verticalOffset, verticalOffsets[0]) <0)
						{
							verticalOffset = -verticalOffset;
							cv::Point2f temp_offset = -verticalOffsets.back();
							verticalOffsets.pop_back();
							verticalOffsets.push_back(temp_offset);
						}
					}

					iter_means++;
				}
			}
		}
	}

	cv::Point2f h = std::accumulate(horizontalOffsets.begin(), horizontalOffsets.end(), cv::Point2f(0, 0)) / (float)horizontalOffsets.size();
	cv::Point2f v = std::accumulate(verticalOffsets.begin(), verticalOffsets.end(), cv::Point2f(0, 0)) / (float)verticalOffsets.size();

	float diagonalOffsetDistance = std::max(cv::norm(h + v), cv::norm(v - h));

	std::vector<float> perimeters;
	std::vector<cv::Point2f> centers;

	for (int i = 0; i < square_vec.size(); i++)
	{
		perimeters.push_back(square_vec[i].m_perimeter);
		centers.push_back(square_vec[i].m_center);
	}

	float averPerimeter = std::accumulate(perimeters.begin(), perimeters.end(), 0) / (float)perimeters.size();
	cv::Point2f averPosition = std::accumulate(centers.begin(), centers.end(), cv::Point2f(0, 0)) / (float)centers.size();

	cv::circle(total, averPosition, 5, cv::Scalar(255, 128, 255), 5);

	std::vector<cv::Point2f> horizontalOffsets_selected, verticalOffsets_selected;
	std::vector<Square> square_vec_selected;

	for (int i = 0; i < square_vec.size(); i++)
	{
		if ((horizontalOffsets[i].dot(h) / cv::norm(horizontalOffsets[i]) / cv::norm(h) > MAX_VECT_ERROR)
			&& (verticalOffsets[i].dot(v) / cv::norm(verticalOffsets[i]) / cv::norm(v) > MAX_VECT_ERROR)
			&& (abs(square_vec[i].m_perimeter - averPerimeter) / averPerimeter < MAX_NORMALIZED_PERIMETER_ERROR)
			&& (cv::norm(averPosition - square_vec[i].m_center) < MAX_NUMBER_SQUARES_FROM_MEAN * diagonalOffsetDistance)
			)
		{
			horizontalOffsets_selected.push_back(horizontalOffsets[i]);
			verticalOffsets_selected.push_back(verticalOffsets[i]);
			square_vec_selected.push_back(square_vec[i]);
		}
	}

	if (square_vec_selected.size()>0)
	{
		cv::Point2f h_new = std::accumulate(horizontalOffsets_selected.begin(), horizontalOffsets_selected.end(), cv::Point2f(0, 0)) / (float)horizontalOffsets_selected.size();
		cv::Point2f v_new = std::accumulate(verticalOffsets_selected.begin(), verticalOffsets_selected.end(), cv::Point2f(0, 0)) / (float)verticalOffsets_selected.size();
		float hx = h_new.x;
		float hy = h_new.y;
		float vx = v_new.x;
		float vy = v_new.y;

		float hv_arr[4] = { hx,vx,hy,vy };
		cv::Mat hv_mat = cv::Mat(cv::Size(2, 2), CV_32FC1, hv_arr);
		cv::Mat basis = hv_mat.inv();

		for (int i = 0; i<square_vec_selected.size(); i++)
		{
			cv::circle(total, square_vec_selected[i].m_center, 5, cv::Scalar(255, 255, 255), 5);

			cv::Scalar temp_color = cv::Scalar(square_vec_selected[i].m_color[0], square_vec_selected[i].m_color[1], square_vec_selected[i].m_color[2]);

			cv::drawContours(total, square_vec_selected[i].m_contour, -1, temp_color, 5);

			float center_arr[2] = { square_vec_selected[i].m_center.x, square_vec_selected[i].m_center.y };
			cv::Mat center_mat = cv::Mat(cv::Size(1, 2), CV_32FC1, center_arr);
			cv::Mat out = basis * center_mat;

			square_vec_selected[i].m_gridX = out.at<float>(0, 0);
			square_vec_selected[i].m_gridY = out.at<float>(1, 0);
		}
		std::vector<Square> square_vec_sorted;
		square_vec_sorted.assign(square_vec_selected.begin(), square_vec_selected.end());
		std::sort(square_vec_sorted.begin(), square_vec_sorted.end(), compareGridX);
		float offsetX = square_vec_sorted[0].m_gridX;
		std::sort(square_vec_sorted.begin(), square_vec_sorted.end(), compareGridY);
		float offsetY = square_vec_sorted[0].m_gridY;
		std::sort(square_vec_sorted.begin(), square_vec_sorted.end(), compareGrid);

		float maxX = 6.0;
		float maxY = 4.0;

		float topLeft = 24.0;
		float topRight = 24.0;
		float bottomLeft = 24.0;
		float bottomRight = 24.0;

		float totalGX = 0.0;
		float totalGY = 0.0;
		float totalXOff = 0.0;
		float totalYOff = 0.0;

		for (int i = 0; i<square_vec_sorted.size(); i++)
		{
			square_vec_sorted[i].m_gridX -= offsetX;
			square_vec_sorted[i].m_gridY -= offsetY;
		}

		int count = 0;
		float residuals = 0.0;
		float bestResiduals = 1000000.0;
		float bestMaxX = 0.0;
		float bestMaxY = 0.0;

		typedef std::map<float, std::map<float, Square>> GridDict;
		GridDict SquareDict;

		std::map<float, Square> emptyDict;

		Square bottomRightSquare, topLeftSquare, topRightSquare, bottomLeftSquare;

		std::vector<Square> tsquares, bestSquares;

		std::vector<float> gridXs, gridYs;

		float minX = 0.0;
		float minY = 0.0;

		while (count < 4)
		{
			for (int i = 0; i<square_vec_sorted.size(); i++)
			{
				if ((square_vec_sorted[i].m_gridX >= count + 0.5) && (square_vec_sorted[i].m_gridX <= count + 1.5))
				{
					gridXs.push_back(square_vec_sorted[i].m_gridX);
				}
				if ((square_vec_sorted[i].m_gridY >= count + 0.5) && (square_vec_sorted[i].m_gridY <= count + 1.5))
				{
					gridYs.push_back(square_vec_sorted[i].m_gridY);
				}
			}
			minX = std::accumulate(gridXs.begin(), gridXs.end(), 0.0) / (float)gridXs.size() / (float)(count + 1);
			minY = std::accumulate(gridYs.begin(), gridYs.end(), 0.0) / (float)gridYs.size() / (float)(count + 1);
			count += 1;

			if (minX != minX || minY != minY)
			{
				continue;
			}

			maxX = 0.0;
			maxY = 0.0;
			residuals = 0.0;

			tsquares.assign(square_vec_sorted.begin(), square_vec_sorted.end());
			for (int i = 0; i < tsquares.size(); i++)
			{
				float tx = tsquares[i].m_gridX;
				float ty = tsquares[i].m_gridY;
				residuals += (abs(tsquares[i].m_gridX / minX - round(tsquares[i].m_gridX / minX)) + abs(tsquares[i].m_gridY / minY - round(tsquares[i].m_gridY / minY)));
				tsquares[i].m_gridX = round(tsquares[i].m_gridX / minX);
				tsquares[i].m_gridY = round(tsquares[i].m_gridY / minY);
				float gridX = tsquares[i].m_gridX;
				float gridY = tsquares[i].m_gridY;

				if (gridX > maxX)
				{
					maxX = gridX;
					totalXOff = tx;
				}
				if (gridY > maxY)
				{
					maxY = gridY;
					totalYOff = ty;
				}
				if (SquareDict.find(gridY) == SquareDict.end())
				{
					SquareDict[gridY] = emptyDict;
				}
				if (SquareDict[gridY].find(gridX) == SquareDict[gridY].end())
				{
					SquareDict[gridY][gridX] = tsquares[i];
				}
				if ((4 - gridX + 6 - gridY)<bottomRight)
				{
					bottomRight = 4 - gridX + 6 - gridY;
					bottomRightSquare = tsquares[i];
				}
				if (gridX + gridY<topLeft)
				{
					topLeft = gridX + gridY;
					topLeftSquare = tsquares[i];
				}
				if (4 - gridX + gridY<topRight)
				{
					topRight = 4 - gridX + gridY;
					topRightSquare = tsquares[i];
				}
				if (gridX + 6 - gridY<bottomLeft)
				{
					bottomLeft = gridX + 6 - gridY;
					bottomLeftSquare = tsquares[i];
				}
			}
			if ((residuals < bestResiduals) && (maxX<6 && maxY<4) || (maxX<4 && maxY<6))
			{
				bestResiduals = residuals;
				bestSquares.assign(tsquares.begin(), tsquares.end());
				bestMaxX = maxX;
				bestMaxY = maxY;
			}

		}
		maxX = 0.0;
		maxY = 0.0;
		residuals = 0.0;
		tsquares.assign(square_vec_sorted.begin(), square_vec_sorted.end());

		for (int i = 0; i<tsquares.size(); i++)
		{
			float tx = tsquares[i].m_gridX;
			float ty = tsquares[i].m_gridY;

			residuals += abs(tsquares[i].m_gridX - round(tsquares[i].m_gridX)) + abs(tsquares[i].m_gridY - round(tsquares[i].m_gridY));
			tsquares[i].m_gridX = round(tsquares[i].m_gridX);
			tsquares[i].m_gridY = round(tsquares[i].m_gridY);

			float gridX = tsquares[i].m_gridX;
			float gridY = tsquares[i].m_gridY;

			if (int(tsquares[i].m_gridX > maxX))
			{
				maxX = tsquares[i].m_gridX;
				totalXOff = tx;
			}
			if (int(tsquares[i].m_gridY>maxY))
			{
				maxY = tsquares[i].m_gridY;
				totalYOff = ty;
			}

			if (SquareDict.find(gridY) == SquareDict.end())
			{
				SquareDict[gridY] = emptyDict;
			}
			if (SquareDict[gridY].find(gridX) == SquareDict[gridY].end())
			{
				SquareDict[gridY][gridX] = tsquares[i];
			}

			if ((4 - gridX + 6 - gridY)<bottomRight)
			{
				bottomRight = 4 - gridX + 6 - gridY;
				bottomRightSquare = tsquares[i];
			}
			if ((gridX + gridY)<topLeft)
			{
				topLeft = gridX + gridY;
				topLeftSquare = tsquares[i];
			}
			if ((4 - gridX + gridY)<topRight)
			{
				topRight = 4 - gridX + gridY;
				topRightSquare = tsquares[i];
			}
			if ((gridX + 6 - gridY)<bottomLeft)
			{
				bottomLeft = gridX + 6 - gridY;
				bottomLeftSquare = tsquares[i];
			}
		}
		if ((residuals < bestResiduals) && (maxX<6 && maxY<4) || (maxX<4 && maxY<6))
		{
			bestResiduals = residuals;
			bestSquares.assign(tsquares.begin(), tsquares.end());
			bestMaxX = maxX;
			bestMaxY = maxY;
		}

		std::vector<Square> square_vec_new;
		square_vec_new.assign(bestSquares.begin(), bestSquares.end());
		maxX = bestMaxX;
		maxY = bestMaxY;

		float ax, ay;

		if (maxX != 0)
		{
			ax = totalXOff / float(maxX);
		}
		else
		{
			ax = 1;
		}

		if (maxY != 0)
		{
			ay = totalYOff / float(maxY);
		}
		else
		{
			ay = 1;
		}
		cv::Point2f recalculatedHorizontalOffset = ax * h_new;
		cv::Point2f recalculatedVerticalOffset = ay * v_new;

		for (int i = 0; i< square_vec_new.size(); i++)
		{
			for (int j = 0; j < square_vec_new.size(); j++)
			{
				if (abs(square_vec_new[j].m_gridX - square_vec_new[i].m_gridX) + abs(square_vec_new[j].m_gridY - square_vec_new[i].m_gridY) == 1)
				{
					cv::line(total, square_vec_new[i].m_center, square_vec_new[j].m_center, 255, 5);
				}
			}
		}

		for (int cy = -6; cy < 6; cy++)
		{
			for (int cx = -6; cx < 6; cx++)
			{
				Square tempSquare;
				std::map<float, int> indexMap;

				if ((SquareDict.find(cy) != SquareDict.end()) && (SquareDict[cy].find(cx) != SquareDict[cy].end()))
				{
					continue;
				}
				else
				{
					if (SquareDict.find(cy) == SquareDict.end())
					{
						SquareDict[cy] = emptyDict;
					}

					for (int i = 0; i < square_vec_new.size(); i++)
					{
						float val = abs(square_vec_new[i].m_gridX - cx) + abs(square_vec_new[i].m_gridY - cy);
						indexMap[val] = i;
					}

					int nearestIndex = indexMap.begin()->second;
					tempSquare.m_center = (cx - square_vec_new[nearestIndex].m_gridX)*recalculatedHorizontalOffset + (cy - square_vec_new[nearestIndex].m_gridY)*recalculatedVerticalOffset + square_vec_new[nearestIndex].m_center;

					if ((tempSquare.m_center.x>0) && (tempSquare.m_center.y > 0) && (tempSquare.m_center.x<labImg.cols) && (tempSquare.m_center.y<labImg.rows))
					{
						tempSquare.m_labcolor = labImg.at<cv::Vec3f>(tempSquare.m_center);
						SquareDict[cy][cx] = tempSquare;
						cv::circle(total, tempSquare.m_center, 3, cv::Scalar(255, 255, 255), 5);
					}
				}
			}
		}
		std::vector<Possibility> Possibilities;
		std::vector<cv::Mat> labRotated = setupStandardColor(labStandardColor);

		for (int i = 0; i<labRotated.size(); i++)
		{
			float width = labRotated[i].cols;
			float height = labRotated[i].rows;
			float tmaxX = maxX;
			float tmaxY = maxY;

			for (int y = 0; y <height - tmaxY; y++)
			{
				for (int x = 0; x<width - tmaxX; x++)
				{
					float tError = 0.0;
					int count = 0;
					for (int cy = 0; cy<tmaxY + 1; cy++)
					{
						for (int cx = 0; cx<tmaxX + 1; cx++)
						{
							if ((SquareDict.find(cy) != SquareDict.end()) && (SquareDict[cy].find(cx) != SquareDict[cy].end()))
							{
								Square t_square = SquareDict[cy][cx];
								cv::Vec3f t_labcolor = labRotated[i].at<float>(y + cy, x + cx);
								tError += cv::norm(t_square.m_labcolor - t_labcolor);
								count += 1;
							}
						}
					}
					Possibility possibility;
					possibility.val_1 = 1 / float(count);
					possibility.val_2 = tError / float(count);
					possibility.val_3 = cv::Vec3f(y, x, i);
					Possibilities.push_back(possibility);
				}
			}
		}
		std::sort(Possibilities.begin(), Possibilities.end(), comparePossibility);
		float rotArr[4] = { 0,-1,1,0 };
		cv::Mat rotMatrix = cv::Mat(cv::Size(2, 2), CV_32FC1, rotArr);

		if (Possibilities.size()>0)
		{
			cv::Vec3f ans = Possibilities[0].val_3;
			float colArr[8] = { 0.0,0.0,3.0,3.0,0.0,5.0,5.0,0.0 };
			cv::Mat regPoints = cv::Mat(cv::Size(4, 2), CV_32FC1, colArr);

			std::vector<cv::Point2f> v_regPoints;
			for (int i = 0; i<4; i++)
			{
				cv::Point2f t_points = cv::Point2f(colArr[i], colArr[i + 4]);
				v_regPoints.push_back(t_points);
			}

			cv::Mat basisMat = cv::Mat::eye(cv::Size(2, 2), CV_32FC1);

			for (int i = 0; i<ans[2] - 1; i++)
			{
				basisMat = basisMat * rotMatrix;
			}

			cv::Mat regPoints_new = (basisMat*regPoints).t();
			float colMin1 = findMatMin(regPoints_new.col(0));
			float colMin2 = findMatMin(regPoints_new.col(1));
			regPoints_new.col(0) -= colMin1;
			regPoints_new.col(1) -= colMin2;

			cv::Point2f position = square_vec_new[0].m_center;
			float xoff = square_vec_new[0].m_gridX;
			float yoff = square_vec_new[0].m_gridY;

			for (int i = 0; i<regPoints_new.rows; i++)
			{
				cv::Point2f regPoint = cv::Point(regPoints_new.at<float>(i, 0), regPoints_new.at<float>(i, 1));
				regPoint -= cv::Point2f(ans[1], ans[0]);
				regPoint -= cv::Point2f(xoff, yoff);
				regPoint = (regPoint.y*recalculatedVerticalOffset + regPoint.x*recalculatedHorizontalOffset) + position;
				cv::Scalar color = cv::Scalar(
					(float)StandardColor.at<cv::Vec3b>(v_regPoints[i]).val[0],
					(float)StandardColor.at<cv::Vec3b>(v_regPoints[i]).val[1],
					(float)StandardColor.at<cv::Vec3b>(v_regPoints[i]).val[2]);

				std::vector<Square> snap;

				for (int j = 0; j<square_vec_new.size(); j++)
				{
					if (cv::norm(square_vec_new[j].m_center - regPoint)<MIN_DIST * 5)
					{
						snap.push_back(square_vec_new[j]);
					}
				}

				std::map<float, std::map<float, Square>>::iterator iter_cy = SquareDict.begin();
				std::map<float, Square>::iterator iter_cx;
				while (iter_cy != SquareDict.end())
				{
					iter_cx = iter_cy->second.begin();

					while (iter_cx != iter_cy->second.end())
					{
						Square t_sqr = iter_cx->second;
						if (cv::norm(t_sqr.m_center - regPoint)<MIN_DIST * 5)
						{
							snap.push_back(t_sqr);
						}
						iter_cx++;
					}
					iter_cy++;
				}
				if (snap.size()>0)
				{
					std::map<float, Square> snap_new;
					for (int k = 0; k<snap.size(); k++)
					{
						float t_norm = cv::norm(snap[k].m_center - regPoint);
						snap_new[t_norm] = snap[k];
					}

					regPoint = snap_new.begin()->second.m_center;
				}
				regPoints_new.at<float>(i, 0) = regPoint.x;
				regPoints_new.at<float>(i, 1) = regPoint.y;
				cv::circle(total, regPoint, 20, color, 20);
			}

			std::vector<cv::Point2f> regMat;
			for (int u = 0; u<regPoints_new.rows; u++)
			{
				regMat.push_back(cv::Point2f((int)regPoints_new.at<float>(u, 0), (int)regPoints_new.at<float>(u, 1)));
			}

			std::vector<cv::Point2f> dstMat;
			dstMat.push_back(cv::Point2f(50, 50));
			dstMat.push_back(cv::Point2f(50, 550));
			dstMat.push_back(cv::Point2f(350, 550));
			dstMat.push_back(cv::Point2f(350, 50));

			cv::Mat pt = cv::getPerspectiveTransform(regMat, dstMat);

			cv::warpPerspective(img, total, pt, cv::Size(400, 600));
		}
		else
		{
			std::cout << "No Possibilities Found." << std::endl;
			cv::resize(total, total, cv::Size(0, 0), 0.2, 0.2);
		}

	}

	return total;
}

cv::Vec3b findROIColor(cv::Mat roi)
{
	cv::Mat ROI = cv::Mat(roi.size(), CV_32FC3);
	roi.convertTo(ROI, CV_32FC3);

	std::vector<cv::Vec3f> colors;

	for (int i = 0; i<ROI.rows; i++)
	{
		for (int j = 0; j<ROI.cols; j++)
		{
			colors.push_back(ROI.at<cv::Vec3f>(i, j));
		}
	}

	cv::Vec3f color_ = std::accumulate(colors.begin(), colors.end(), cv::Vec3f(0, 0, 0)) / (float)colors.size();

	cv::Vec3b color = cv::Vec3b(round(color_[0]), round(color_[1]), round(color_[2]));

	return color;
}

cv::Mat getColorChartFromImage(cv::Mat img, int spotsize = 10)
{
	cv::Mat ret = cv::Mat::zeros(cv::Size(4, 6), CV_8UC3);
	for (int y = 0; y<6; y++)
	{
		for (int x = 0; x<4; x++)
		{
			int py = int((y + 0.5) * 100);
			int px = int((x + 0.5) * 100);
			cv::Mat roi = img(cv::Rect(cv::Point(px - spotsize, py - spotsize), cv::Point(px + spotsize, py + spotsize)));
			cv::Vec3b color = findROIColor(roi);
			ret.at<cv::Vec3b>(y, x) = color;
		}
	}
	return ret;
}

cv::Mat getStandardColorChart()
{
	uchar arrStandardColor[6][4][3]
		= { { { 171,191,99 },
			{ 41,161,229 },
			{ 166,136,0 },
			{ 50,50,50 } },

			{ { 176, 129,130 },
			{ 62,189,160 },
			{ 150,84,188 },
			{ 85,84,83 } },

			{ { 65,108,90 },
			{ 105,59,91 },
			{ 22,200,238 },
			{ 121,121,120 } },

			{ { 157,123,93 },
			{ 98,84,195 },
			{ 56,48,176 },
			{ 161,161,160 } },


			{ { 129,149,196 },
			{ 168,92,72 },
			{ 72,149,71 },
			{ 201,201,200 } },


			{ { 67,81,115 },
			{ 45,123,220 },
			{ 147,62,43 },
			{ 240,245,245 } } };

	cv::Mat StandardColor = cv::Mat(cv::Size(4, 6), CV_8UC3, arrStandardColor);
	return StandardColor.clone();
}

cv::Mat applyWhiteBalancePR(cv::Mat img, float white_rate)
{
	cv::Mat processedImg = cv::Mat(img.size(), img.type());

	std::vector<float>RGBs;

	for (int i = 0; i<img.rows; i++)
	{
		for (int j = 0; j<img.cols; j++)
		{
			float rgb = static_cast<float>(img.at<cv::Vec3b>(i, j).val[0] + img.at<cv::Vec3b>(i, j).val[1] + img.at<cv::Vec3b>(i, j).val[2]);
			RGBs.push_back(rgb);
		}
	}

	int maxNum = round(RGBs.size()*white_rate);

	std::sort(RGBs.begin(), RGBs.end(), compare);

	float threshold = RGBs[maxNum - 1];

	int amount = 0;
	float redSum = 0, greenSum = 0, blueSum = 0;

	for (int i = 0; i<img.rows; i++)
	{
		for (int j = 0; j<img.cols; j++)
		{
			float rgb = static_cast<float>(img.at<cv::Vec3b>(i, j).val[0] + img.at<cv::Vec3b>(i, j).val[1] + img.at<cv::Vec3b>(i, j).val[2]);
			if (rgb >= threshold)
			{
				amount++;
				blueSum += static_cast<float>(img.at<cv::Vec3b>(i, j).val[0]);
				greenSum += static_cast<float>(img.at<cv::Vec3b>(i, j).val[1]);
				redSum += static_cast<float>(img.at<cv::Vec3b>(i, j).val[2]);
			}
		}
	}

	float blueAver = blueSum / amount;
	float greenAver = greenSum / amount;
	float redAver = redSum / amount;

	for (int i = 0; i<img.rows; i++)
	{
		for (int j = 0; j<img.cols; j++)
		{
			float blue = static_cast<float>(img.at<cv::Vec3b>(i, j).val[0] / blueAver * 255);
			float green = static_cast<float>(img.at<cv::Vec3b>(i, j).val[1] / greenAver * 255);
			float red = static_cast<float>(img.at<cv::Vec3b>(i, j).val[2] / redAver * 255);

			if (blue>255)
			{
				blue = 255;
			}
			else if (blue<0)
			{
				blue = 0;
			}
			if (green>255)
			{
				green = 255;
			}
			else if (green<0)
			{
				green = 0;
			}
			if (red>255)
			{
				red = 255;
			}
			else if (red<0)
			{
				red = 0;
			}

			processedImg.at<cv::Vec3b>(i, j).val[0] = static_cast<uchar>(blue);
			processedImg.at<cv::Vec3b>(i, j).val[1] = static_cast<uchar>(green);
			processedImg.at<cv::Vec3b>(i, j).val[2] = static_cast<uchar>(red);
		}
	}

	return processedImg;
}

cv::Mat applyGamma(cv::Mat img, float gamma_r = 2.2, float gamma_g = 2.2, float gamma_b = 2.2,
	float gain_r = 1.0, float gain_g = 1.0, float gain_b = 1.0)
{
	cv::Mat gammaImg = cv::Mat(img.size(), img.type());

	for (int i = 0; i<img.rows; i++)
	{
		for (int j = 0; j<img.cols; j++)
		{
			float blue = static_cast<float>(img.at<cv::Vec3b>(i, j).val[0]);
			float green = static_cast<float>(img.at<cv::Vec3b>(i, j).val[1]);
			float red = static_cast<float>(img.at<cv::Vec3b>(i, j).val[2]);

			// 恢复到和人眼特性一致的颜色空间中，即非线性RGB空间
			int blue_gamma = static_cast<int>(powf((blue / 255.0), 1.0 / gamma_b)*gain_b*255.0);
			int green_gamma = static_cast<int>(powf((green / 255.0), 1.0 / gamma_g)*gain_g*255.0);
			int red_gamma = static_cast<int>(powf((red / 255.0), 1.0 / gamma_r)*gain_r*255.0);

			blue_gamma = (blue_gamma>255) ? 255 : blue_gamma;
			green_gamma = (green_gamma>255) ? 255 : green_gamma;
			red_gamma = (red_gamma>255) ? 255 : red_gamma;

			gammaImg.at<cv::Vec3b>(i, j).val[0] = blue_gamma;
			gammaImg.at<cv::Vec3b>(i, j).val[1] = green_gamma;
			gammaImg.at<cv::Vec3b>(i, j).val[2] = red_gamma;
		}
	}

	return gammaImg;
}

cv::Mat deGamma(cv::Mat img, float gamma_r = 2.2, float gamma_g = 2.2, float gamma_b = 2.2,
	float gain_r = 1.0, float gain_g = 1.0, float gain_b = 1.0)
{
	// 由于相机读取到的数据和人眼特性一致，并不处在线性RGB空间中，即已经在线性基础上乘上了一个因子：1/2.2, 
	// 因此degamma需要将其恢复到线性RGB空间中，以便进行颜色校正和白平衡调节
	float inv_gamma_r = 1 / gamma_r;
	float inv_gamma_g = 1 / gamma_g;
	float inv_gamma_b = 1 / gamma_b;

	cv::Mat degammaImg = applyGamma(img, inv_gamma_r, inv_gamma_g, inv_gamma_b, gain_r, gain_g, gain_b);

	return degammaImg;
}

cv::Mat computeCCM(cv::Mat uncorrected, cv::Mat reference, float gamma_r = 2.2, float gamma_g = 2.2, float gamma_b = 2.2,
	float gain_r = 1.0, float gain_g = 1.0, float gain_b = 1.0)
{
	/* DeGamma */
	cv::Mat linearUncorrected = deGamma(uncorrected, gamma_r, gamma_g, gamma_b, gain_r, gain_g, gain_b);
	cv::Mat linearReference = deGamma(reference, gamma_r, gamma_g, gamma_b, gain_r, gain_g, gain_b);

	/* LinearBGR -> XYZ */
	cv::Mat uncorrectedXYZ = cv::Mat(linearUncorrected.size(), CV_8UC3);
	cv::Mat referenceXYZ = cv::Mat(linearReference.size(), CV_8UC3);
	cv::cvtColor(linearUncorrected, uncorrectedXYZ, cv::COLOR_BGR2XYZ);
	cv::cvtColor(linearReference, referenceXYZ, cv::COLOR_BGR2XYZ);

	/* Solve */
	std::vector<uchar> vUncorrectedXYZ, vReferenceXYZ;

	vUncorrectedXYZ.assign(uncorrectedXYZ.datastart, uncorrectedXYZ.dataend);
	vReferenceXYZ.assign(referenceXYZ.datastart, referenceXYZ.dataend);

	cv::Mat UncorrectedXYZ = cv::Mat(cv::Size(3, uncorrectedXYZ.rows*uncorrectedXYZ.cols), CV_8UC1, vUncorrectedXYZ.data());
	cv::Mat ReferenceXYZ = cv::Mat(cv::Size(3, referenceXYZ.rows*referenceXYZ.cols), CV_8UC1, vReferenceXYZ.data());

	UncorrectedXYZ.convertTo(UncorrectedXYZ, CV_32FC1);
	ReferenceXYZ.convertTo(ReferenceXYZ, CV_32FC1);

	cv::Mat homoUncorrectedXYZ = cv::Mat::ones(cv::Size(UncorrectedXYZ.cols + 1, UncorrectedXYZ.rows), CV_32FC1);
	UncorrectedXYZ.col(0).copyTo(homoUncorrectedXYZ.col(0));
	UncorrectedXYZ.col(1).copyTo(homoUncorrectedXYZ.col(1));
	UncorrectedXYZ.col(2).copyTo(homoUncorrectedXYZ.col(2));

	cv::Mat invHomoUncorrectedXYZ;
	cv::invert(homoUncorrectedXYZ, invHomoUncorrectedXYZ, cv::DECOMP_SVD);

	cv::Mat CCM = invHomoUncorrectedXYZ * ReferenceXYZ;

	return CCM;
}

cv::Mat correctColor(cv::Mat img, cv::Mat ccm)
{
	std::vector<uchar> img_vec;
	img_vec.assign(img.datastart, img.dataend);
	cv::Mat uncorrected = cv::Mat(cv::Size(3, img.rows*img.cols), CV_8UC1, img_vec.data());
	uncorrected.convertTo(uncorrected, CV_32FC1);

	cv::Mat homoUncorrected = cv::Mat::ones(cv::Size(uncorrected.cols + 1, uncorrected.rows), CV_32FC1);
	uncorrected.col(0).copyTo(homoUncorrected.col(0));
	uncorrected.col(1).copyTo(homoUncorrected.col(1));
	uncorrected.col(2).copyTo(homoUncorrected.col(2));

	cv::Mat corrected = homoUncorrected * ccm;

	return corrected;
}

cv::Mat colorCalibrate(cv::Mat img, cv::Mat ccm, float gamma_r = 2.2, float gamma_g = 2.2, float gamma_b = 2.2,
	float gain_r = 1.0, float gain_g = 1.0, float gain_b = 1.0)
{
	cv::Mat linearBGR = deGamma(img, gamma_r, gamma_g, gamma_b, gain_r, gain_g, gain_b);
	cv::Mat XYZ;
	cv::cvtColor(linearBGR, XYZ, cv::COLOR_BGR2XYZ);
	cv::Mat corrected = correctColor(XYZ, ccm);
	corrected.convertTo(corrected, CV_8UC1);

	std::vector<uchar> corrected_vec;
	corrected_vec.assign(corrected.datastart, corrected.dataend);
	cv::Mat correctedXYZ = cv::Mat(img.size(), img.type(), corrected_vec.data());

	cv::Mat correctedLinearBGR;
	cv::cvtColor(correctedXYZ, correctedLinearBGR, cv::COLOR_XYZ2BGR);

	// 以后需要加white rate参数
	cv::Mat wbCorrectedLinearBGR;
	wbCorrectedLinearBGR = applyWhiteBalancePR(correctedLinearBGR, 0.2);
	// 确定degamma和gamma的意义

	cv::Mat correctedBGR = applyGamma(wbCorrectedLinearBGR, gamma_r, gamma_g, gamma_b, gain_r, gain_g, gain_b);

	return correctedBGR;
}

int main()
{
	cv::Mat src = cv::imread("..//data//colorchecker_pic.jpg");
	cv::Mat colorchecker = findColorChecker(src);
	cv::Mat srcColor = getColorChartFromImage(colorchecker);
	/* Directly use a known color chart */
	cv::Mat dstColor = getStandardColorChart();
	/* Extract color chart from a standard colorchecker image */
	//cv::Mat dst = cv::imread("..//data//colorchecker.jpg");
	//cv::Mat dstColor = getColorChartFromImage(dst);

	float gamma = 2.2;
	cv::Mat ccm = computeCCM(srcColor, dstColor, gamma, gamma, gamma);
	std::cout << "CCM: " << std::endl;
	std::cout << ccm << std::endl;

	cv::Mat corrected = colorCalibrate(src, ccm, gamma, gamma, gamma);
	cv::imwrite("..//data//corrected_pic.jpg", corrected);

	return 0;
}


















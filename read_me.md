# read me

## 相机标定

先提取图片采集角点，设定真实世界坐标点，计算出内参和畸变系数，求出旋转向量和平移向量

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main()
{
    const Size  board_size(11, 8);   // 内角点
    const float square_size = 50.f;  // 方格现实尺寸(mm)
    const int   Need = 25;           // 照片数量

  
    Mat frame, gray, show;

    Size imageSize;//图片尺寸
    vector<vector<Point2f>> imgsPoints;//2维所有图的内角点
    namedWindow("camera", WINDOW_NORMAL);

    for(int i=0;i<Need;i++)
    {
        string path = cv::format("image_%06d.png",i+1);
        frame = imread(path);
        if (frame.empty())//丢帧跳过
        {
            std::cout << "frame empty" << endl;
            continue;
        }

        if (imageSize.width == 0)//第一帧决定图像尺寸
        {
            imageSize = frame.size();
        }

        frame.copyTo(show);
        putText(show, cv::format("captured %d/%d   [ESC=quit]",
            (int)imgsPoints.size(), Need),
            Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 255), 2);
        imshow("camera", show);

        int key = waitKey(10) & 0xFF;   
        if (key == 27 || key == 'q') break;

        //检测
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, board_size, corners,
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        if (!found)
        {
            std::cout << "this frame has no chessboard, not saved" << endl;
            continue;
        }

        // 亚像素细化
        cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
            TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001));

        imgsPoints.push_back(corners);
        //string name = cv::format("calib_%02d.png", (int)imgsPoints.size());
        //imwrite(name, frame);   
        //std::cout << "saved " << name << " (" << imgsPoints.size() << "/" << Need << ")" << endl;

        // 显示一下刚采集的角点
        drawChessboardCorners(frame, board_size, corners, true);
        imshow("camera", frame);
        waitKey(200);
        
    }

    cv::destroyAllWindows();   
    

    if ((int)imgsPoints.size() < 4)//如果不够，不要进行下面的程序
    {
        std::cout << "not enough images: " << imgsPoints.size() << endl;
        return -1;
    }

    vector<Point3f> oneBoard;
    oneBoard.reserve(board_size.area());              // width * height = 88 个点
    for (int r = 0; r < board_size.height; r++)       // 行
        for (int c = 0; c < board_size.width; c++)    // 列
            oneBoard.push_back(Point3f(c * square_size,   // x 沿宽度方向
                r * square_size,   // y 沿高度方向
                0.f));             // 平面板：z = 0

    vector<vector<Point3f>> objectPoints(imgsPoints.size(), oneBoard);

    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    Mat distCoeffs = Mat::zeros(1, 5, CV_64F);
    vector<Mat> rvecs, tvecs;

    double rms = calibrateCamera(objectPoints, imgsPoints, imageSize,
        cameraMatrix, distCoeffs, rvecs, tvecs, 0);

    std::cout << "overall RMS reprojection error = " << rms << " pixel" << endl;
    std::cout << "camera matrix = " << endl << cameraMatrix << endl;
    std::cout << "distortion (k1,k2,p1,p2,k3) = " << distCoeffs << endl;

    const string calibFile = "camera_params.xml";
    FileStorage fs(calibFile, FileStorage::WRITE);
    if (!fs.isOpened())
    {
        std::cout << "cannot open " << calibFile << " for writing" << endl;
        return -1;
    }

    fs << "CAMERA_MATRIX_1" << cameraMatrix;            // 3x3 内参
    fs << "DISTORTION_COEFF_1" << distCoeffs;           // 1x5 畸变 k1 k2 p1 p2 k3
    fs << "IMAGE_WIDTH" << imageSize.width;             // 标定时的分辨率
    fs << "IMAGE_HEIGHT" << imageSize.height;       
    fs << "SQUARE_SIZE_MM" << square_size;              // 棋盘格边长
    fs << "IMAGE_COUNT" << (int)imgsPoints.size();      // 用了几张图
    fs << "REPROJECTION_RMS" << rms;                    // 重投影误差，越小越好
    fs.release();

    std::cout << "\ncalibration saved to " << calibFile << endl;

    return 0;
}
```

![image-20261018152150425](C:\Users\2200\AppData\Roaming\Typora\typora-user-images\image-20261018152150425.png)





## 装甲板识别

用通道相减预处理后，膨胀补全灯条，提取轮廓筛选灯条，俩俩配对，用角度差，长度比，位置间隔的比较识别出装甲板。后面的svm数字识别我没有去训练过。在调试过程中，不断改变参数识别到灯塔上小的装甲板。我是通过debug运行的，延迟是20多。

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cstdio>

using namespace std;
using namespace cv;
using namespace cv::ml;   // SVM

//全局参数
struct ArmorParam
{
    int   color_threshold;      //阈值   
    int   min_area;             //轮廓面积
    float max_angle;            //相对竖直最大角度
    float max_angle_diff;       //最大角度差
    float max_lengthDiff_ratio; //最大长度差比 = |L1-L2| / max(L1,L2)
    float max_deviation_angle;  //两灯条中心连线相对水平线允许的最大夹角
    float max_y_diff_ratio;     // Y 方向错位比 = |Δy| / 平均灯条长度
    float max_x_diff_ratio;     // X 方向间距比 = |Δx| / 平均灯条长度
    float min_x_diff_ratio;     // X 方向间距比最小值

    ArmorParam()
        : color_threshold(80),
        min_area(50),
        max_angle(45.f),
        max_angle_diff(6.f),
        max_lengthDiff_ratio(0.5f),
        max_deviation_angle(50.f),
        max_y_diff_ratio(0.5f),
        max_x_diff_ratio(4.5f),
        min_x_diff_ratio(0.7f)
    {
    }
};

static ArmorParam armorParam;

//状态机
enum DetectorState
{
    LIGHTS_NOT_FOUND = 0,   // 灯条少于 2 条，退出
    LIGHTS_FOUND = 1,       // 找到 >= 2 条灯条，尝试配对
    ARMOR_NOT_FOUND = 2,    // 灯条配不出任何合法装甲板
    ARMOR_FOUND = 3         // 至少配出一块装甲板
};

enum ArmorType { SMALL_ARMOR = 0, BIG_ARMOR = 1 };  // 小/大装甲板
enum Color { BLUE = 0, GREEN = 1, RED = 2 };    // 敌方颜色

class LightBar
{
public:
    LightBar() : length(0.f), angle(0.f) {}

    explicit LightBar(const RotatedRect& lightRect)
    {
        this->lightRect = lightRect;

        //取最大为length
        length = MAX(lightRect.size.height, lightRect.size.width);

        center = lightRect.center;

        angle = (lightRect.angle > 90.f) ? lightRect.angle - 180.f : lightRect.angle;
    }

    RotatedRect lightRect;   // 拟合椭圆得到的旋转矩形
    float       length;      // 灯条长度
    Point2f     center;      // 灯条中心
    float       angle;       // 相对竖直方向的夹角
};

//灯条中心
static Point2f crossPointof(const Point2f& lower_l, const Point2f& upper_l,
    const Point2f& upper_r, const Point2f& lower_r)
{
    // 对角线 1：lower_l -> upper_r
    float a1 = upper_r.y - lower_l.y;                          // = y2 - y1
    float b1 = upper_r.x - lower_l.x;                          // = x2 - x1
    float c1 = lower_l.x * upper_r.y - upper_r.x * lower_l.y;  // = x1*y2 - x2*y1

    // 对角线 2：upper_l -> lower_r
    float a2 = lower_r.y - upper_l.y;
    float b2 = lower_r.x - upper_l.x;
    float c2 = upper_l.x * lower_r.y - lower_r.x * upper_l.y;

    float d = a1 * b2 - a2 * b1;    // 系数行列式

    //d ≈ 0 没有交点。
    if (fabs(d) < 1e-6f)
        return Point2f(FLT_MAX, FLT_MAX);

    return Point2f((b2 * c1 - b1 * c2) / d,      // x
        (c1 * a2 - c2 * a1) / d);     // y
}

//装甲板
class ArmorBox
{
public:
    //造一个空装甲板
    ArmorBox()
        : l_index(-1), r_index(-1), armorNum(0), type(SMALL_ARMOR),
        center(0.f, 0.f), armorAngle(0.f)
    {
        armorVertices.resize(4);   // 四个顶点，先占好位置
    }

    //用左右两条灯条拼出一块装甲板
    ArmorBox(const LightBar& l_light, const LightBar& r_light)
        : l_index(-1), r_index(-1), armorNum(0), type(SMALL_ARMOR),
        center(0.f, 0.f), armorAngle(0.f)
    {
        this->l_light = l_light;
        this->r_light = r_light;

        armorNum = 0;   // 装甲板数字

        // 装甲板倾角 = 两条灯条角度的平均值
        armorAngle = (l_light.angle + r_light.angle) / 2.f;

        
        Size exLSize(int(l_light.lightRect.size.width), int(l_light.lightRect.size.height * 2));
        Size exRSize(int(r_light.lightRect.size.width), int(r_light.lightRect.size.height * 2));
        RotatedRect exLLight(l_light.center, exLSize, armorAngle);   // 左灯条：中心不变，拉长
        RotatedRect exRLight(r_light.center, exRSize, armorAngle);   // 右灯条：同样处理

    
        Point2f pts_l[4]; exLLight.points(pts_l);
        Point2f pts_r[4]; exRLight.points(pts_r);

        armorVertices.resize(4);
        armorVertices[0] = pts_l[3];   // lower_left  = 左灯条右边缘的下端点
        armorVertices[1] = pts_l[2];   // upper_left  = 左灯条右边缘的上端点
        armorVertices[2] = pts_r[1];   // upper_right = 右灯条左边缘的上端点
        armorVertices[3] = pts_r[0];   // lower_right = 右灯条左边缘的下端点

        //中心 = 两条对角线的交点
        center = crossPointof(armorVertices[0], armorVertices[1], armorVertices[2], armorVertices[3]);

        //外接正矩形
        armorRect = boundingRect(armorVertices);

        // 大小装甲板判定：两灯条间距 / 灯条长度 > 2.5 就算大装甲板。
        if (getDislocationX() > 2.5f)
            type = BIG_ARMOR;
    }

    // 左右灯条的角度差
    float getAngleDiff() const { return fabs(l_light.angle - r_light.angle); }

    // 错位度角：两条灯条中心的连线，与水平线的夹角  
    float getDeviationAngle() const
    {
        float delta_x = r_light.center.x - l_light.center.x;
        float delta_y = r_light.center.y - l_light.center.y;
        return fabs(atan(delta_y / delta_x)) * 180.f / (float)CV_PI;   // 弧度转角度
    }

    // X 方向间距比 = 两灯条中心的水平距离 / 平均灯条长度。  
    float getDislocationX() const
    {
        float meanLen = (l_light.length + r_light.length) / 2.f;
        return fabs(l_light.center.x - r_light.center.x) / meanLen;
    }

    // Y 方向错位比 = 两灯条中心的竖直距离 / 平均灯条长度
    float getDislocationY() const
    {
        float meanLen = (l_light.length + r_light.length) / 2.f;
        return fabs(l_light.center.y - r_light.center.y) / meanLen;
    }

    // 长度差比 = |L1 - L2| / max(L1, L2)
    // 同一块装甲板两侧灯条几乎等长，所以这个值应该接近 0
    float getLengthRation() const
    {
        float length_diff = fabs(l_light.length - r_light.length);
        return length_diff / MAX(l_light.length, r_light.length);
    }

    
    bool isSuitableArmor() const
    {
        return getAngleDiff() < armorParam.max_angle_diff           // 平行
            && getDeviationAngle() < armorParam.max_deviation_angle // 连线接近水平
            && getDislocationX() < armorParam.max_x_diff_ratio      // 间距不能过大
            && getDislocationX() > armorParam.min_x_diff_ratio      // 间距不能过小
            && getDislocationY() < armorParam.max_y_diff_ratio      // 不能一上一下
            && getLengthRation() < armorParam.max_lengthDiff_ratio; // 长度接近
    }

    LightBar l_light, r_light;      // 组成装甲板的左右两条灯条
    int l_index, r_index;           // 它们在 lights 数组里的下标（-1 表示还没设置）
    int armorNum;                   // 装甲板上的数字
    vector<Point2f> armorVertices;  // 四个顶点，顺序 bl -> tl -> tr -> br
    ArmorType type;                 // 大/小装甲板
    Point2f center;                 // 装甲板中心（对角线交点）
    Rect armorRect;                 // 外接正矩形（打分用）
    float armorAngle;               // 装甲板倾角（两条灯条角度的平均）

    Mat armorImg;                   // roi透视变换
};

//数字识别
class ArmorNumClassifier
{
public:
    // 载入训练好的模型
    bool loadSvmModel(const char* modelPath, Size armorImgSize = Size(40, 40))
    {
        svm = StatModel::load<SVM>(modelPath);
        imgSize = armorImgSize;
        return !svm.empty();
    }

    bool ready() const { return !svm.empty(); }

    void getArmorImg(const Mat& grayImg, ArmorBox& armor)
    {
        if (grayImg.empty() || armor.armorVertices.size() != 4) return;

        float W = (float)imgSize.width;
        float H = (float)imgSize.height;

        Point2f src[4] = { armor.armorVertices[0], armor.armorVertices[1],
                           armor.armorVertices[2], armor.armorVertices[3] };
        Point2f dst[4] = { Point2f(0.f, H), Point2f(0.f, 0.f),
                           Point2f(W, 0.f), Point2f(W, H) };

        Mat M = getPerspectiveTransform(src, dst);
        warpPerspective(grayImg, armor.armorImg, M, imgSize, INTER_LINEAR, BORDER_REPLICATE);

        threshold(armor.armorImg, armor.armorImg, 0, 255, THRESH_BINARY | THRESH_OTSU);
    }

    // 送进 SVM 预测数字
    void setArmorNum(ArmorBox& armor)
    {
        if (svm.empty() || armor.armorImg.empty()) { armor.armorNum = 0; return; }

        // reshape(1,1)：40x40 -> 1 行 1600 列；predict 要求 CV_32F
        Mat p = armor.armorImg.reshape(1, 1);
        p.convertTo(p, CV_32F);

        armor.armorNum = (int)svm->predict(p);
    }

private:
    Ptr<SVM> svm;              // SVM 模型
    Size     imgSize = Size(40, 40);   // 模型训练时的图片尺寸
};


//欧式距离
static float getPointsDistance(const Point2f& a, const Point2f& b)
{
    float dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

//识别判断装甲板
class ArmorDetector
{
public:
    ArmorDetector() : enemyColor(BLUE), targetNum(0), state(LIGHTS_NOT_FOUND) {}

    void setEnemyColor(Color c) { enemyColor = c; }   // 设置打哪一方

    //载入数字识别模型
    void loadSVM(const char* modelPath) { classifier.loadSvmModel(modelPath); }

    //相机内参
    void setCameraParam(const Mat& K, const Mat& D)
    {
        cameraMatrix = K.clone();
        distCoeffs = D.clone();
    }

    //每帧开始时清空上一帧的结果
    void resetDetector()
    {
        state = LIGHTS_NOT_FOUND;
        lights.clear();
        armors.clear();
        rejected.clear();
    }

    // 图像预处理 setImg()
    void setImg(Mat& src)
    {
        src.copyTo(srcImg);   // 深拷贝一份

        // 二值图：和原图同样大小，单通道，先全填 0（黑）
        srcImg_binary = Mat::zeros(srcImg.size(), CV_8UC1);

        //通道相减
        uchar* pdata = (uchar*)srcImg.data;          // 指向原图首字节
        uchar* qdata = (uchar*)srcImg_binary.data;   // 指向二值图首字节
        int    srcData = srcImg.rows * srcImg.cols;  // 像素总个数

        if (enemyColor == RED)
        {
            for (int i = 0; i < srcData; ++i)
            {
                if (*(pdata + 2) - *pdata > armorParam.color_threshold)  // R - B
                    *qdata = 255;
                pdata += 3;    // +3 字节 = 下一个像素
                ++qdata;
            }
        }
        else  // BLUE
        {
            for (int i = 0; i < srcData; ++i)
            {
                if (*pdata - *(pdata + 2) > armorParam.color_threshold)  // B - R
                    *qdata = 255;
                pdata += 3;
                ++qdata;
            }
        }

        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        dilate(srcImg_binary, srcImg_binary, kernel);

        cvtColor(srcImg, srcImg_gray, COLOR_BGR2GRAY);
    }

    // 灯条检测 findLights()
    void findLights()
    {
        vector<vector<Point>> lightContours;   
        Mat contourImg;
        srcImg_binary.copyTo(contourImg);      // 复制一份给 findContours 用

        //只取外轮廓
        findContours(contourImg, lightContours, 0, 2);

        //筛轮廓
        for (size_t k = 0; k < lightContours.size(); ++k)
        {
            const vector<Point>& c = lightContours[k];
            double area = contourArea(c);   // 轮廓围出的面积（像素个数）
            RotatedRect lightRect;

            //轮廓点数
            if (c.size() < 6)
            {
                continue;
            }

            //面积
            if (area < armorParam.min_area)
            {
                continue;
            }

            lightRect = fitEllipse(c);
            LightBar light(lightRect);   // 构造灯条

            //角度
            if (fabs(light.angle) > armorParam.max_angle)
            {
                continue;
            }

            // 三道筛都过了，收进 lights
            lights.push_back(light);
        }

        // 少于 2 条灯条就不可能组成装甲板
        if (lights.size() < 2)
        {
            state = LIGHTS_NOT_FOUND;
            return;
        }

        // 按中心 x 坐标从小到大排序
        sort(lights.begin(), lights.end(),
            [](const LightBar& a, const LightBar& b) { return a.center.x < b.center.x; });

        state = LIGHTS_FOUND;
    }

    //匹配灯条
    void matchArmors()
    {
        for (size_t i = 0; i + 1 < lights.size(); ++i)          
        {
            for (size_t j = i + 1; j < lights.size(); ++j)      
            {
                //i 和 j 之间若还夹着别的同层灯条，不可能是同一块板
                bool inBetween = false;
                float yLow = MIN(lights[i].center.y, lights[j].center.y);
                float yHigh = MAX(lights[i].center.y, lights[j].center.y);
                float meanLen = (lights[i].length + lights[j].length) / 2.f;
                for (size_t k = i + 1; k < j; ++k)
                    if (lights[k].center.y > yLow - meanLen * 0.5f &&
                        lights[k].center.y < yHigh + meanLen * 0.5f) {
                        inBetween = true; break;
                    }
                if (inBetween) continue;

                // 用这一对灯条拼一块装甲板
                ArmorBox armor(lights[i], lights[j]);

                if (armor.isSuitableArmor())
                {
                   
                    
                    armor.l_index = (int)i;
                    armor.r_index = (int)j;

                   
                    if (classifier.ready())
                    {
                        classifier.getArmorImg(srcImg_gray, armor);
                        classifier.setArmorNum(armor);
                    }

                    armors.push_back(armor);
                }
                else
                {
                    
                    rejected.push_back(armor);
                }
            }

            // 每处理完一条灯条，就清理一次假装甲板。

            eraseErrorRepeatArmor();
        }

        if (armors.empty())
        {
            state = ARMOR_NOT_FOUND;
            return;
        }

        state = ARMOR_FOUND;

    }

    //选出目标装甲板 setTargetArmor()
    //一个打分算法
    void setTargetArmor()
    {
        if (state == ARMOR_FOUND)
        {
            ArmorBox mva = armors[0];   // mva = most valuable armor，先假设第一个最好
            for (size_t i = 1; i < armors.size(); ++i)
            {
                // armorCompare 返回 true 表示 armors[i] 比当前 mva 更值得打
                if (armorCompare(armors[i], mva)) mva = armors[i];
            }
            targetArmor = mva;
        }
        lastArmor = targetArmor;
    }

    //ArmorDetector::run()
    void run(Mat& src)
    {
        setImg(src);
        resetDetector();      // 清空上一帧的灯条和装甲板
        findLights();         // 找灯条
        if (state == LIGHTS_FOUND)
        {
            matchArmors();    // 两两配对成装甲板
            if (state == ARMOR_FOUND)
                setTargetArmor();   // 从候选里选出唯一的目标装甲板
        }
    }

   
    float solveDistance(const ArmorBox& armor) const
    {
        if (cameraMatrix.empty() || armor.armorVertices.size() != 4)
            return -1.f;

        // 真实尺寸：宽（两灯条内侧间距方向）、高（2 倍灯条长度方向）
        float w = (armor.type == BIG_ARMOR) ? 230.f : 135.f;
        float h = (armor.type == BIG_ARMOR) ? 127.f : 125.f;
        float hx = w / 2.f, hy = h / 2.f;

   
        vector<Point3f> obj;
        obj.push_back(Point3f(-hx, -hy, 0.f));   // bl 左下
        obj.push_back(Point3f(-hx, hy, 0.f));   // tl 左上
        obj.push_back(Point3f(hx, hy, 0.f));   // tr 右上
        obj.push_back(Point3f(hx, -hy, 0.f));   // br 右下

        Mat rvec, tvec;
        // SOLVEPNP_ITERATIVE：LM 迭代优化，平面 4 点最常用
        if (!solvePnP(obj, armor.armorVertices, cameraMatrix, distCoeffs,
            rvec, tvec, false, SOLVEPNP_ITERATIVE))
            return -1.f;

        return (float)norm(tvec);   // 模长 = 直线距离（单位 mm）
    }
  
    bool isFoundArmor() const { return state == ARMOR_FOUND; }
    DetectorState getState() const { return state; }
    const vector<LightBar>& getLights() const { return lights; }
    const vector<ArmorBox>& getArmors() const { return armors; }
    const ArmorBox& getTargetArmor() const { return targetArmor; }

   
    // 把二值图转成三通道，这样才能和彩色原图拼在一起显示 
    void drawBinary(Mat& canvas) const
    {
        cvtColor(srcImg_binary, canvas, COLOR_GRAY2BGR);
    }

    // 画灯条绿色旋转矩形 + 紫色中心点 + 角度/长度文字 
    void drawLights(Mat& canvas) const
    {
        for (size_t i = 0; i < lights.size(); ++i)
        {
            Point2f pts[4];
            lights[i].lightRect.points(pts);      // 取出旋转矩形的四个角
            for (int k = 0; k < 4; ++k)
                line(canvas, pts[k], pts[(k + 1) % 4], Scalar(0, 255, 0), 1, LINE_AA);  // 四点依次连线成框
            circle(canvas, lights[i].center, 2, Scalar(255, 0, 255), -1, LINE_AA);

            char buf[32];
            snprintf(buf, sizeof(buf), "a=%.0f L=%.0f", lights[i].angle, lights[i].length);
            putText(canvas, buf, lights[i].center + Point2f(-46, -30),
                FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 255), 1, LINE_AA);
        }
    }

    // 画被拒的细红框
    void drawRejected(Mat& canvas) const
    {
        for (size_t i = 0; i < rejected.size(); ++i)
        {
            vector<Point> poly;
            for (int k = 0; k < 4; ++k) poly.push_back(rejected[i].armorVertices[k]);
            polylines(canvas, poly, true, Scalar(80, 80, 200), 1, LINE_AA);
        }
    }

    // 画通过判据的装甲板：青色框；其中最终目标再用白色粗框描一遍 
    void drawArmors(Mat& canvas) const
    {
        for (size_t i = 0; i < armors.size(); ++i)
        {
            vector<Point> poly;
            for (int k = 0; k < 4; ++k) poly.push_back(armors[i].armorVertices[k]);
            polylines(canvas, poly, true, Scalar(200, 200, 0), 2, LINE_AA);
            circle(canvas, armors[i].center, 3, Scalar(0, 255, 255), -1, LINE_AA);

            // 识别到的数字标在装甲板
            char buf[32];
            snprintf(buf, sizeof(buf), "%s %d",
                armors[i].type == BIG_ARMOR ? "BIG" : "SMALL", armors[i].armorNum);
            putText(canvas, buf, armors[i].center + Point2f(-18, -12),
                FONT_HERSHEY_PLAIN, 1.1, Scalar(255, 255, 0), 1, LINE_AA);
        }
        if (state == ARMOR_FOUND)
        {
            vector<Point> poly;
            for (int k = 0; k < 4; ++k) poly.push_back(targetArmor.armorVertices[k]);
            polylines(canvas, poly, true, Scalar(255, 255, 255), 2, LINE_AA);   // 目标：白色粗框
        }
    }

private:
    
    //删掉假装甲板
    int eraseErrorRepeatArmor()
    {
        if (armors.size() < 2) return 0;   // 只有一块板时out

        vector<bool> drop(armors.size(), false);   // 标记数组，true = 这块要淘汰

        for (size_t i = 0; i + 1 < armors.size(); ++i)
        {
            for (size_t j = i + 1; j < armors.size(); ++j)
            {
                // 已经确定要删的就不再参与比较
                if (drop[i] || drop[j]) continue;

                //判断两块装甲板是否共用灯条：比较它们记录下的灯条下标。
                bool shareLight = (armors[i].l_index == armors[j].l_index ||
                    armors[i].l_index == armors[j].r_index ||
                    armors[i].r_index == armors[j].l_index ||
                    armors[i].r_index == armors[j].r_index);
                if (!shareLight) continue;   // 不共用灯条，两块可以共存

                // 错位角大的淘汰
                if (armors[i].getDeviationAngle() > armors[j].getDeviationAngle())
                    drop[i] = true;
                else
                    drop[j] = true;
            }
        }

        // 统一重建：把没被标记的搬进新数组，再换回来
        vector<ArmorBox> kept;
        int dropped = 0;
        for (size_t i = 0; i < armors.size(); ++i)
        {
            if (!drop[i]) kept.push_back(armors[i]);
            else ++dropped;
        }
        armors.swap(kept);   
        return dropped;
    }

  
    static void setNumScore(int armorNum, int targetNum, float& score)
    {
        if (targetNum == 0)   // 没指定按数字大小排优先级
        {
            if (armorNum == 1) score += 1000;
            else if (armorNum == 2) score += 2000;
            else if (armorNum == 3) score += 3000;
            else if (armorNum == 4) score += 4000;
            else if (armorNum == 5) score += 5000;
            else if (armorNum == 6) score += 6000;
        }
        if (armorNum == targetNum && targetNum != 0) score += 100000;   // 指定目标
    }

    
    bool armorCompare(const ArmorBox& a_armor, const ArmorBox& b_armor) const
    {
        float a_score = (float)a_armor.armorRect.area();
        float b_score = (float)b_armor.armorRect.area();

        setNumScore(a_armor.armorNum, targetNum, a_score);
        setNumScore(b_armor.armorNum, targetNum, b_score);

        // 只有上一帧确实有目标时才做距离惩罚
        // （默认装甲板的 area 是 0，用这个条件判断 lastArmor 是不是空的）
        if (lastArmor.armorNum != 0 || lastArmor.armorRect.area() > 0)
        {
            a_score -= getPointsDistance(a_armor.center, lastArmor.center) * 2.f;
            b_score -= getPointsDistance(b_armor.center, lastArmor.center) * 2.f;
        }
        return a_score > b_score;
    }

   
    Mat srcImg;                     // 原图
    Mat srcImg_binary;              // 二值图
    Mat srcImg_gray;                // 原图的灰度图
    Color enemyColor;               // 打红方还是蓝方
    int targetNum;                  // 操作手指定的目标数字
    vector<LightBar> lights;        // 本帧找到的所有灯条
    vector<ArmorBox> armors;        // 本帧匹配出的所有装甲板
    vector<ArmorBox> rejected;      // 没通过五条判据的候选
    ArmorBox targetArmor;           // 本帧最终目标
    ArmorBox lastArmor;             // 上一帧的目标（打分时的时间连续性约束）
    DetectorState state;            // 检测状态

    ArmorNumClassifier classifier;  // 数字识别
    Mat cameraMatrix;               // 相机内参 3x3
    Mat distCoeffs;                 // 畸变系数 1x5
};



//视频
int main(int argc, char** argv)
{
    
    string path = (argc > 1) ? argv[1] : "shibie.avi";

    VideoCapture cap;
    if (path == "0")
        cap.open(0, CAP_DSHOW);   
        cap.open(path);

    if (!cap.isOpened())
    {
        cout << "打开视频失败: " << path << endl;
        return -1;
    }

    ArmorDetector detector;
    detector.setEnemyColor(BLUE);
    //detector.loadSVM("model/123svm.xml");

    FileStorage fs("camera_params.xml", FileStorage::READ);
    Mat K, D;
    fs["CAMERA_MATRIX_1"] >> K;
    fs["DISTORTION_COEFF_1"] >> D;

    fs.release();

    detector.setCameraParam(K, D);

    VideoWriter writer;
 
    double sumMs = 0, maxMs = 0;   
    int frameCount = 0;

    Mat frame;
    while (true)
    {
        cap >> frame;              // 读一帧
        if (frame.empty()) break;  // 视频读完

        
        TickMeter tm;
        tm.start();
        detector.run(frame);      
        tm.stop();

        double ms = tm.getTimeMilli();
        sumMs += ms;
        maxMs = MAX(maxMs, ms);
        ++frameCount;

        
        Mat canvas = frame.clone();
        
        detector.drawArmors(canvas);     // 通过的装甲板 + 目标白框

        char info[160];
        if (detector.isFoundArmor())
        {
            const ArmorBox& target = detector.getTargetArmor();
            float distance = detector.solveDistance(target);

            circle(canvas, target.center, 4, Scalar(0, 0, 255), -1, LINE_AA);  // 中心点
            snprintf(info, sizeof(info), "num=%d  dist=%.0fmm", target.armorNum, distance);
            putText(canvas, info, Point(10, 28), FONT_HERSHEY_SIMPLEX, 0.7,
                Scalar(0, 255, 255), 2, LINE_AA);
        }
        else
        {
            putText(canvas, "ARMOR NOT FOUND", Point(10, 28), FONT_HERSHEY_SIMPLEX, 0.7,
                Scalar(0, 0, 255), 2, LINE_AA);
        }

        // 实时耗时 + 累计平均 + 最慢帧
        snprintf(info, sizeof(info), "%.2f ms | avg %.2f | max %.2f",
            ms, sumMs / frameCount, maxMs);
        putText(canvas, info, Point(10, 56), FONT_HERSHEY_PLAIN, 1.2,
            Scalar(0, 255, 255), 1, LINE_AA);

        
        if (!writer.isOpened())
            writer.open("output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'),
                30.0, canvas.size());

        writer.write(canvas);
        imshow("armor", canvas);

        
        int key = waitKey(1);
        if (key == 'q' || key == 27) break;              // q 或 Esc 退出
        
    }

    cap.release();
    writer.release();
    destroyAllWindows();

    
    if (frameCount > 0)
    {
        cout << "处理帧数: " << frameCount << endl;
        cout << "平均耗时: " << sumMs / frameCount << " ms" << endl;
        cout << "最大耗时: " << maxMs << " ms" << endl;
    }
    return 0;
}
```

![image-20261018171757543](C:\Users\2200\AppData\Roaming\Typora\typora-user-images\image-20261018171757543.png)


import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;

import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;

import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat img = Imgcodecs.imread("test_books_images/test_image_2.jpg");
        Mat imgHSV = new Mat();
        Imgproc.cvtColor(img, imgHSV, Imgproc.COLOR_BGR2HSV);

        // 设定颜色阈值并进行掩膜处理
        Scalar lower_red = new Scalar(0, 125, 180);
        Scalar upper_red = new Scalar(80, 235, 255);
        Mat mask = new Mat();
        Core.inRange(imgHSV, lower_red, upper_red, mask);
        Mat imgResult = new Mat();
        Core.bitwise_and(img, img, imgResult, mask);

        // 转换为灰度图并检测边缘
        Mat gray = new Mat();
        Imgproc.cvtColor(imgResult, gray, Imgproc.COLOR_BGR2GRAY);
        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, 50, 120);

        // 腐蚀和闭合操作
        Mat kernel = Mat.ones(new Size(1, 1), CvType.CV_8U);
        Mat eroded_edges = new Mat();
        Imgproc.erode(edges, eroded_edges, kernel, new Point(-1, -1), 2);
        Mat closing = new Mat();
        Imgproc.morphologyEx(eroded_edges, closing, Imgproc.MORPH_CLOSE, kernel);

        // 寻找轮廓
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(closing, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // 处理每个轮廓
        Tesseract tesseract = new Tesseract();
        tesseract.setDatapath("path_to_tessdata"); // 设置tesseract的语言包路径
        tesseract.setLanguage("chi_sim"); // 使用中文语言包

        for (int i = 0; i < contours.size(); i++) {
            Rect rect = Imgproc.boundingRect(contours.get(i));
            if (rect.width > 10 && rect.height > 10) {  // 过滤掉噪声
                Mat subMat = img.submat(rect);

                // 使用Tesseract OCR提取文字
                try {
                    String text = tesseract.doOCR(Mat2BufferedImage(subMat));
                    System.out.println("Text in rectangle " + i + ": " + text);
                } catch (TesseractException e) {
                    e.printStackTrace();
                }

                // 显示分割出来的矩形区域
                // 由于Java没有直接的imshow方法，可以使用其它库显示图像
                // 例如使用OpenCV的imshow或将图像保存后再查看
                Imgcodecs.imwrite("rect" + i + ".jpg", subMat);
            }
        }
    }

    // 工具方法：将Mat转换为BufferedImage
    private static BufferedImage Mat2BufferedImage(Mat matrix) {
        int cols = matrix.cols();
        int rows = matrix.rows();
        int elemSize = (int)matrix.elemSize();
        byte[] data = new byte[cols * rows * elemSize];
        int type;

        matrix.get(0, 0, data);

        switch (matrix.channels()) {
            case 1:
                type = BufferedImage.TYPE_BYTE_GRAY;
                break;
            case 3:
                type = BufferedImage.TYPE_3BYTE_BGR;
                // BGR to RGB conversion
                byte b;
                for(int i = 0; i < data.length; i = i + 3) {
                    b = data[i];
                    data[i] = data[i+2];
                    data[i+2] = b;
                }
                break;
            default:
                return null;
        }

        BufferedImage image = new BufferedImage(cols, rows, type);
        image.getRaster().setDataElements(0, 0, cols, rows, data);
        return image;
    }
}
